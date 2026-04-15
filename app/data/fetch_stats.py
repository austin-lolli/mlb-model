"""Pull pitcher and team stats: FanGraphs (optional) or MLB Stats API."""

from __future__ import annotations

import logging
import re

import pandas as pd
import requests
from pybaseball.datasources.html_table_processor import HTMLTableProcessor

from app.config import PITCHER_PRIOR_YEAR_FALLBACK, USE_FANGRAPHS
from app.data.normalize import normalize_player_name
from app.data.fetch_stats_mlb import fetch_pitcher_stats_mlb, fetch_team_batting_stats_mlb

logger = logging.getLogger(__name__)

# FanGraphs often returns 403 without a browser-like User-Agent.
_FG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _patch_fangraphs_http() -> None:
    if getattr(HTMLTableProcessor, "_pipeline_fg_headers", False):
        return

    def get_tabular_data_from_url(
        self,
        url: str,
        query_params=None,
        column_name_mapper=None,
        known_percentages=None,
        row_id_func=None,
        row_id_name=None,
    ):
        response = requests.get(
            self.root_url + url,
            params=query_params or {},
            headers=_FG_HEADERS,
            timeout=120,
        )
        if response.status_code > 399:
            raise requests.exceptions.HTTPError(
                f"Error accessing '{self.root_url + url}'. Received status code {response.status_code}"
            )
        return self.get_tabular_data_from_html(
            response.content,
            column_name_mapper=column_name_mapper,
            known_percentages=known_percentages,
            row_id_func=row_id_func,
            row_id_name=row_id_name,
        )

    HTMLTableProcessor.get_tabular_data_from_url = get_tabular_data_from_url  # type: ignore[method-assign]
    HTMLTableProcessor._pipeline_fg_headers = True


_PCT_RE = re.compile(r"^[\s]*([0-9]*\.?[0-9]+)\s*%?\s*$")


def _parse_pct(val) -> float | None:
    """FanGraphs percentages may be float or string '21.3%'. Return 0–100 scale float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    m = _PCT_RE.match(s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except ValueError:
        return None


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


_STATS_SEASON_COL = "_stats_season"


def _tag_pitcher_stats_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Mark each row with the season those rate stats came from (for prior-year fallback)."""
    if df.empty:
        return df
    out = df.copy()
    out[_STATS_SEASON_COL] = season
    return out


def _merge_prior_year_pitcher_rows(cur: pd.DataFrame, prev: pd.DataFrame, season: int) -> pd.DataFrame:
    """Append prior-season rows whose Name is absent from ``cur`` (both must include _stats_season)."""
    if cur.empty or prev.empty:
        return cur
    names_cur = set(cur["Name"].astype(str).str.strip())
    extra = prev[~prev["Name"].astype(str).str.strip().isin(names_cur)].copy()
    if extra.empty:
        return cur
    logger.info(
        "Pitcher stats: merged %s rows from season %s (names not in season %s)",
        len(extra),
        season - 1,
        season,
    )
    return pd.concat([cur, extra], ignore_index=True)


def _fetch_pitcher_stats_single_season(season: int) -> pd.DataFrame:
    """Single-season pitcher table only (no ``_stats_season``, no prior-year merge)."""
    if not USE_FANGRAPHS:
        logger.info("Pitcher stats: MLB Stats API (USE_FANGRAPHS unset/false)")
        return fetch_pitcher_stats_mlb(season)

    logger.info("Fetching pitching_stats(%s) via FanGraphs/pybaseball", season)
    _patch_fangraphs_http()
    from pybaseball import pitching_stats  # noqa: PLC0415

    try:
        df = pitching_stats(season)
    except requests.exceptions.HTTPError as e:
        logger.warning("FanGraphs pitching_stats failed (%s); using MLB Stats API fallback", e)
        return fetch_pitcher_stats_mlb(season)
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "Name",
                "Team",
                "K%",
                "BB%",
                "xFIP",
                "kbb",
                "id",
                "gamesStarted",
                "gamesPlayed",
                "is_reliever_season",
            ]
        )

    k_col = _find_column(df, ("K%",))
    bb_col = _find_column(df, ("BB%",))
    xfip_col = _find_column(df, ("xFIP",))
    name_col = _find_column(df, ("Name", "name"))
    if not name_col:
        logger.warning("pitching_stats: no Name column; returning empty frame")
        return pd.DataFrame(
            columns=[
                "Name",
                "Team",
                "K%",
                "BB%",
                "xFIP",
                "kbb",
                "id",
                "gamesStarted",
                "gamesPlayed",
                "is_reliever_season",
            ]
        )

    out = pd.DataFrame()
    out["Name"] = df[name_col] if name_col else None
    k_series = df[k_col].map(_parse_pct) if k_col else pd.Series(dtype=float)
    bb_series = df[bb_col].map(_parse_pct) if bb_col else pd.Series(dtype=float)
    out["K%"] = k_series
    out["BB%"] = bb_series
    out["xFIP"] = pd.to_numeric(df[xfip_col], errors="coerce") if xfip_col else None
    out["kbb"] = out["K%"] - out["BB%"]
    out = out.dropna(subset=["Name"])
    mlb = fetch_pitcher_stats_mlb(season)
    if not mlb.empty and "id" in mlb.columns:
        mlb_sub = mlb.dropna(subset=["id"]).copy()
        mlb_sub["_n"] = mlb_sub["Name"].astype(str).map(normalize_player_name)
        id_by_norm = mlb_sub.drop_duplicates(subset=["_n"], keep="first").set_index("_n")["id"]
        out = out.copy()
        out["_n"] = out["Name"].astype(str).map(normalize_player_name)
        out["id"] = out["_n"].map(id_by_norm)
        out = out.drop(columns=["_n"])
        id_to_extra = (
            mlb.dropna(subset=["id"])
            .drop_duplicates(subset=["id"], keep="first")
            .set_index("id")[["Team", "gamesStarted", "gamesPlayed", "is_reliever_season"]]
        )
        out["Team"] = out["id"].map(id_to_extra["Team"])
        out["gamesStarted"] = out["id"].map(id_to_extra["gamesStarted"])
        out["gamesPlayed"] = out["id"].map(id_to_extra["gamesPlayed"])
        out["is_reliever_season"] = out["id"].map(id_to_extra["is_reliever_season"])
    else:
        out["id"] = pd.NA
        out["Team"] = pd.NA
        out["gamesStarted"] = pd.NA
        out["gamesPlayed"] = pd.NA
        out["is_reliever_season"] = pd.NA
    logger.info("Pitcher rows: %s", len(out))
    return out


def fetch_pitcher_stats(season: int, *, prior_year_fallback: bool | None = None) -> pd.DataFrame:
    """
    Pitcher-season rows: Name, K%, BB%, xFIP, kbb, plus ``_stats_season`` after merge.

    When ``prior_year_fallback`` is true (default: ``PITCHER_PRIOR_YEAR_FALLBACK`` env / config),
    fetches ``season - 1`` and appends any ``Name`` not present in ``season`` so probables
    without current-year lines can still match. ``_stats_season`` records which year each row's
    rates came from. Set env ``PITCHER_PRIOR_YEAR_FALLBACK=0`` to disable.
    """
    use_fb = PITCHER_PRIOR_YEAR_FALLBACK if prior_year_fallback is None else prior_year_fallback
    cur_raw = _fetch_pitcher_stats_single_season(season)
    cur = _tag_pitcher_stats_season(cur_raw, season)
    if cur.empty or not use_fb or season <= 1876:
        return cur
    prev_raw = _fetch_pitcher_stats_single_season(season - 1)
    prev = _tag_pitcher_stats_season(prev_raw, season - 1)
    if prev.empty:
        return cur
    return _merge_prior_year_pitcher_rows(cur, prev, season)


def fetch_batting_stats(season: int) -> pd.DataFrame:
    """
    Team batting: Team and wRC+ for joining to schedule.

    FanGraphs path uses team_batting(); MLB path uses OPS indexed to league average in wRC+ column.
    """
    if not USE_FANGRAPHS:
        logger.info("Team batting: MLB Stats API (USE_FANGRAPHS unset/false)")
        return fetch_team_batting_stats_mlb(season)

    logger.info("Fetching team_batting(%s) via FanGraphs/pybaseball", season)
    _patch_fangraphs_http()
    from pybaseball import team_batting  # noqa: PLC0415

    try:
        df = team_batting(season)
    except requests.exceptions.HTTPError as e:
        logger.warning("FanGraphs team_batting failed (%s); using MLB Stats API fallback", e)
        return fetch_team_batting_stats_mlb(season)
    if df is None or df.empty:
        return pd.DataFrame(columns=["Team", "wRC+"])

    team_col = _find_column(df, ("Team", "team"))
    wrc_col = _find_column(df, ("wRC+", "wrc+", "wRC Plus"))

    out = pd.DataFrame()
    out["Team"] = df[team_col].astype(str) if team_col else None
    out["wRC+"] = pd.to_numeric(df[wrc_col], errors="coerce") if wrc_col else None
    out = out.dropna(subset=["Team"])
    logger.info("Team batting rows: %s", len(out))
    return out
