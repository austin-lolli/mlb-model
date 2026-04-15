"""FanGraphs-unavailable fallback: pitcher and team stats from statsapi.mlb.com."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from app.config import MLB_API_ABBR_TO_FG_ABBR, RELIEVER_GS_RATIO_MAX
from app.data.normalize import fangraphs_team_abbrev

logger = logging.getLogger(__name__)

MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"
FIP_C_F = 3.10  # league FIP constant (approximate)


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict:
    r = requests.get(url, params=params or {}, timeout=120)
    r.raise_for_status()
    return r.json()


def _parse_innings(ip: str | None) -> float:
    if not ip:
        return 0.0
    s = str(ip).strip()
    if "." in s:
        whole, frac = s.split(".", 1)
        try:
            w = int(whole)
            f = int(frac[0]) if frac else 0
            return w + f / 3.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_avg_ops(s: str | None) -> float:
    if not s:
        return float("nan")
    try:
        return float(str(s).strip())
    except ValueError:
        return float("nan")


def _fip(hr: int, bb: int, hbp: int, k: int, ip: float) -> float:
    if ip <= 0:
        return float("nan")
    return (13 * hr + 3 * (bb + hbp) - 2 * k) / ip + FIP_C_F


def fetch_pitcher_stats_mlb(season: int) -> pd.DataFrame:
    """
    Season pitching from MLB Stats API. Uses FIP (not xFIP) in the xFIP column
    for compatibility with build_features when FanGraphs is blocked.
    """
    url = f"{MLB_STATS_BASE}/stats"
    params: dict[str, Any] = {
        "stats": "season",
        "season": season,
        "group": "pitching",
        "sportIds": 1,
        "playerPool": "all",
        "limit": 5000,
    }
    logger.info("MLB fallback: pitching season stats (%s)", season)
    payload = _get_json(url, params)
    stats_block = (payload.get("stats") or [{}])[0]
    splits = stats_block.get("splits") or []
    rows: list[dict[str, Any]] = []
    for sp in splits:
        st = sp.get("stat") or {}
        pl = sp.get("player") or {}
        name = pl.get("fullName")
        if not name:
            continue
        pid = pl.get("id")
        bf = st.get("battersFaced") or 0
        so = st.get("strikeOuts") or 0
        bb = st.get("baseOnBalls") or 0
        hr = st.get("homeRuns") or 0
        hbp = st.get("hitBatsmen") or 0
        ip = _parse_innings(st.get("inningsPitched"))
        gs = int(st.get("gamesStarted") or 0)
        g = int(st.get("gamesPlayed") or st.get("games") or 0)
        k_pct = 100.0 * float(so) / float(bf) if bf else float("nan")
        bb_pct = 100.0 * float(bb) / float(bf) if bf else float("nan")
        fip = _fip(int(hr), int(bb), int(hbp), int(so), ip)
        team = sp.get("team") or {}
        tname = team.get("name")
        team_abbr = fangraphs_team_abbrev(str(tname)) if tname else None
        is_rel = bool(g >= 3 and (gs / g) < RELIEVER_GS_RATIO_MAX)
        rows.append(
            {
                "id": int(pid) if pid is not None else None,
                "Name": str(name),
                "Team": team_abbr,
                "K%": k_pct,
                "BB%": bb_pct,
                "xFIP": fip,
                "kbb": k_pct - bb_pct,
                "gamesStarted": gs,
                "gamesPlayed": g,
                "is_reliever_season": is_rel,
            }
        )
    out = pd.DataFrame(rows)
    logger.info("MLB fallback pitcher rows: %s", len(out))
    return out


def _team_list() -> list[dict[str, Any]]:
    payload = _get_json(f"{MLB_STATS_BASE}/teams", {"sportId": 1, "activeStatus": "Active"})
    teams = payload.get("teams") or []
    # MLB only (exclude complex / inactive without division)
    return [t for t in teams if t.get("division") and t.get("sport", {}).get("id") == 1]


def fetch_team_batting_stats_mlb(season: int) -> pd.DataFrame:
    """
    Team offense: no wRC+ in MLB API — use OPS indexed to league average (100 = avg).
    Column kept as wRC+ for build_features compatibility.
    """
    teams = _team_list()
    ops_list: list[tuple[str, float]] = []
    for t in teams:
        tid = t.get("id")
        abbr = t.get("abbreviation")
        if tid is None or not abbr:
            continue
        url = f"{MLB_STATS_BASE}/teams/{tid}/stats"
        try:
            payload = _get_json(
                url,
                {"stats": "season", "group": "hitting", "season": season},
            )
        except requests.HTTPError as e:
            logger.warning("Team %s stats failed: %s", tid, e)
            continue
        stats_block = (payload.get("stats") or [{}])[0]
        splits = stats_block.get("splits") or []
        if not splits:
            continue
        st = splits[0].get("stat") or {}
        ops = _parse_avg_ops(st.get("ops"))
        if pd.isna(ops):
            continue
        fg = MLB_API_ABBR_TO_FG_ABBR.get(str(abbr), str(abbr))
        ops_list.append((fg, float(ops)))

    if not ops_list:
        return pd.DataFrame(columns=["Team", "wRC+"])

    lg_ops = sum(o for _, o in ops_list) / len(ops_list)
    rows = [{"Team": abbr, "wRC+": 100.0 * (ops / lg_ops)} for abbr, ops in ops_list]
    out = pd.DataFrame(rows)
    logger.info("MLB fallback team batting rows: %s (OPS index as wRC+)", len(out))
    return out
