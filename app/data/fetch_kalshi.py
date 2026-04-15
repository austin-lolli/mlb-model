"""Read-only Kalshi Trade API helpers for market-implied probabilities."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from app.config import DATA_DIR, KALSHI_API_BASE
from app.data.normalize import fangraphs_team_abbrev

logger = logging.getLogger(__name__)

DEFAULT_KALSHI_MAP_PATH = DATA_DIR / "kalshi_game_map.csv"
KALSHI_GAME_SERIES_TICKER = "KXMLBGAME"

_KALSHI_TO_FG_ABBR = {
    "TB": "TBR",
    "SD": "SDP",
    "SF": "SFG",
    "WSH": "WSN",
    "CWS": "CHW",
    "AZ": "ARI",
}
_MONTH_TO_NUM = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _parse_dollar_field(val) -> float | None:
    if val is None:
        return None
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return None


def _yes_mid_prob(m: dict[str, Any]) -> float | None:
    """Mid of YES bid/ask in 0-1 contract dollars (Kalshi v2: *_dollars strings)."""
    bid = _parse_dollar_field(m.get("yes_bid_dollars"))
    ask = _parse_dollar_field(m.get("yes_ask_dollars"))
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    lp = _parse_dollar_field(m.get("last_price_dollars"))
    if lp is not None:
        return lp
    # Legacy field names (cents), if present.
    bid = m.get("yes_bid")
    ask = m.get("yes_ask")
    if bid is not None and ask is not None:
        try:
            return (float(bid) + float(ask)) / 2.0 / 100.0
        except (TypeError, ValueError):
            pass
    return None


def implied_prob_from_mid(mid: float | None) -> float:
    if mid is None:
        return float("nan")
    return float(mid)


def fetch_market(ticker: str, *, timeout: float = 30.0) -> dict[str, Any]:
    from urllib.parse import quote

    url = f"{KALSHI_API_BASE.rstrip('/')}/markets/{quote(str(ticker), safe='-_.~')}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    m = payload.get("market") or payload
    return m if isinstance(m, dict) else {}


def _fetch_event(event_ticker: str, *, timeout: float = 30.0) -> dict[str, Any]:
    from urllib.parse import quote

    url = f"{KALSHI_API_BASE.rstrip('/')}/events/{quote(str(event_ticker), safe='-_.~')}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    ev = payload.get("event") or payload
    return ev if isinstance(ev, dict) else {}


def _fetch_open_game_markets(*, timeout: float = 30.0, limit: int = 500) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {
            "series_ticker": KALSHI_GAME_SERIES_TICKER,
            "status": "open",
            "limit": min(max(int(limit), 1), 1000),
        }
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{KALSHI_API_BASE.rstrip('/')}/markets", params=params, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        batch = payload.get("markets") or []
        if not isinstance(batch, list) or not batch:
            break
        markets.extend([m for m in batch if isinstance(m, dict)])
        cursor = payload.get("cursor") or ""
        if not cursor:
            break
    return markets


def _fg_from_kalshi_code(code: str) -> str:
    c = str(code or "").strip().upper()
    return _KALSHI_TO_FG_ABBR.get(c, c)


def _event_date_from_ticker(event_ticker: str) -> pd.Timestamp | None:
    # KXMLBGAME-26APR141915MIAATL -> 2026-04-14
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})\d{4}", str(event_ticker))
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2), m.group(3)
    mm = _MONTH_TO_NUM.get(mon.upper())
    if mm is None:
        return None
    try:
        year = 2000 + int(yy)
        day = int(dd)
        return pd.Timestamp(year=year, month=mm, day=day).normalize()
    except ValueError:
        return None


def _away_home_codes_from_subtitle(sub_title: str | None) -> tuple[str, str] | None:
    # e.g. "MIA vs ATL (Apr 14)"
    m = re.search(r"\b([A-Z]{2,4})\s+vs\s+([A-Z]{2,4})\b", str(sub_title or ""))
    if not m:
        return None
    away = _fg_from_kalshi_code(m.group(1))
    home = _fg_from_kalshi_code(m.group(2))
    if not away or not home:
        return None
    return away, home


def _build_auto_kalshi_map_rows(games: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Auto-map only current/future MLB games by matching Kalshi game events:
    (game_date, away team, home team) -> home-winning market ticker.
    """
    if games.empty:
        return []
    need = {"game_pk", "game_date", "home_team_name", "away_team_name"}
    if not need.issubset(games.columns):
        return []

    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.normalize()
    today = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    g = g[g["game_date"] >= today].copy()
    if g.empty:
        return []

    g["home_fg"] = g["home_team_name"].map(lambda x: fangraphs_team_abbrev(str(x or "")))
    g["away_fg"] = g["away_team_name"].map(lambda x: fangraphs_team_abbrev(str(x or "")))
    g = g.dropna(subset=["game_pk", "game_date", "home_fg", "away_fg"])
    if g.empty:
        return []

    try:
        markets = _fetch_open_game_markets()
    except Exception as e:
        logger.warning("Kalshi auto-map: could not fetch open MLB game markets: %s", e)
        return []

    by_event: dict[str, list[dict[str, Any]]] = {}
    for m in markets:
        ev = str(m.get("event_ticker") or "").strip()
        if not ev:
            continue
        by_event.setdefault(ev, []).append(m)

    event_cache: dict[str, dict[str, Any]] = {}
    event_key_to_home_ticker: dict[tuple[pd.Timestamp, str, str], str] = {}

    for ev, ev_markets in by_event.items():
        if not ev.startswith(f"{KALSHI_GAME_SERIES_TICKER}-"):
            continue
        if ev not in event_cache:
            try:
                event_cache[ev] = _fetch_event(ev)
            except Exception as e:
                logger.debug("Kalshi auto-map: event %s lookup failed: %s", ev, e)
                continue
        eobj = event_cache.get(ev) or {}
        dt = _event_date_from_ticker(ev)
        ah = _away_home_codes_from_subtitle(eobj.get("sub_title"))
        if dt is None or ah is None:
            continue
        away_fg, home_fg = ah
        home_ticker = ""
        for m in ev_markets:
            t = str(m.get("ticker") or "")
            # Market suffix is usually a team code (e.g. -SD, -SEA). Normalize to FG code.
            suf = t.rsplit("-", 1)[-1].strip().upper() if "-" in t else ""
            if _fg_from_kalshi_code(suf) == home_fg:
                home_ticker = t
                break
        if home_ticker:
            event_key_to_home_ticker[(dt, away_fg, home_fg)] = home_ticker

    mapped_rows: list[dict[str, Any]] = []
    for _, row in g.iterrows():
        key = (pd.Timestamp(row["game_date"]).normalize(), str(row["away_fg"]), str(row["home_fg"]))
        ticker = event_key_to_home_ticker.get(key)
        if not ticker:
            continue
        gpk = int(row["game_pk"])
        mapped_rows.append(
            {
                "game_pk": gpk,
                "kalshi_ticker_home": ticker,
                "mapping_source": "auto_kxmlbgame",
                "mapped_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    return mapped_rows


def load_kalshi_game_map(path: str | Path | None = None) -> pd.DataFrame:
    """
    CSV columns: game_pk,kalshi_ticker_home (YES = home team wins).

    Extra columns are ignored.
    """
    p = Path(path) if path else DEFAULT_KALSHI_MAP_PATH
    if not p.is_file():
        logger.info("Kalshi map missing at %s; implied columns will be NaN", p)
        return pd.DataFrame(columns=["game_pk", "kalshi_ticker_home"])
    df = pd.read_csv(p, dtype={"game_pk": "Int64", "kalshi_ticker_home": str})
    if "game_pk" not in df.columns or "kalshi_ticker_home" not in df.columns:
        raise ValueError("kalshi_game_map.csv must include game_pk and kalshi_ticker_home")
    return df.dropna(subset=["game_pk"])


def refresh_kalshi_game_map_for_games(
    games: pd.DataFrame,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Auto-populate ``kalshi_game_map.csv`` for current/future games when possible.
    Existing mappings are preserved and take precedence.
    """
    p = Path(path) if path else DEFAULT_KALSHI_MAP_PATH
    existing = load_kalshi_game_map(p)
    auto_rows = _build_auto_kalshi_map_rows(games)
    if not auto_rows:
        return existing
    auto_df = pd.DataFrame(auto_rows)
    merged = pd.concat([existing, auto_df], ignore_index=True, sort=False)
    merged = merged.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"], keep="first")
    merged = merged.sort_values("game_pk")
    p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(p, index=False)
    existing_pks = set(existing["game_pk"].dropna().astype(int).tolist()) if not existing.empty else set()
    added = len(set(auto_df["game_pk"].tolist()) - existing_pks)
    if added > 0:
        logger.info("Kalshi auto-map wrote %s new game_pk->ticker mappings to %s", added, p)
    return merged


def build_kalshi_features_for_games(
    games: pd.DataFrame,
    *,
    map_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    One row per input game row (same order): game_pk, kalshi_home_implied, kalshi_away_implied, edge_vs_model (NaN).
    """
    mdf = refresh_kalshi_game_map_for_games(games, map_path)
    rows: list[dict[str, Any]] = []
    ticker_cache: dict[str, float] = {}

    def implied_for_ticker(t: str) -> float:
        t = str(t).strip()
        if not t:
            return float("nan")
        if t in ticker_cache:
            return ticker_cache[t]
        try:
            mkt = fetch_market(t)
        except Exception as e:
            logger.warning("Kalshi market %s: %s", t, e)
            ticker_cache[t] = float("nan")
            return float("nan")
        mid = _yes_mid_prob(mkt)
        prob = implied_prob_from_mid(mid)
        ticker_cache[t] = prob
        return prob

    map_by_pk = (
        mdf.drop_duplicates(subset=["game_pk"]).set_index("game_pk")["kalshi_ticker_home"].to_dict()
        if not mdf.empty
        else {}
    )

    for _, row in games.iterrows():
        gpk = row.get("game_pk")
        gpk_i = int(gpk) if gpk is not None and not pd.isna(gpk) else None
        ticker = map_by_pk.get(gpk_i) if gpk_i is not None else None
        home_p = implied_for_ticker(ticker) if ticker else float("nan")
        away_p = 1.0 - home_p if pd.notna(home_p) else float("nan")
        rows.append(
            {
                "game_pk": gpk_i,
                "kalshi_home_implied": home_p,
                "kalshi_away_implied": away_p,
                "edge_vs_model": float("nan"),
            }
        )

    return pd.DataFrame(rows)
