"""Merge schedule rows with pitcher and team batting stats into a model table."""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from app.data.normalize import fangraphs_team_abbrev
from app.utils.name_matching import match_player_names

logger = logging.getLogger(__name__)

_FINAL_STATES = frozenset({"Final", "Game Over", "Completed Early"})
# Set on pitcher rows when ``fetch_pitcher_stats`` merges prior-season fallback.
_STATS_SEASON_COL = "_stats_season"


def _to_float_stat(val: Any) -> float:
    """Coerce a pitcher stat cell to float, or NaN if missing or invalid."""
    if val is None:
        return float("nan")
    try:
        x = float(val)
    except (TypeError, ValueError):
        return float("nan")
    if pd.isna(x):
        return float("nan")
    return x


def _platoon_value(
    platoon_by_team: pd.DataFrame | None,
    team_abbr: str,
    opp_throws: str | None,
) -> float:
    if platoon_by_team is None or platoon_by_team.empty or team_abbr not in platoon_by_team.index:
        return float("nan")
    ot = str(opp_throws).strip().upper()[:1] if opp_throws else ""
    if ot == "R":
        return float(platoon_by_team.loc[team_abbr, "wRC_plus_vs_RHP"])
    if ot == "L":
        return float(platoon_by_team.loc[team_abbr, "wRC_plus_vs_LHP"])
    return float("nan")


def build_dataset(
    games: pd.DataFrame,
    pitchers: pd.DataFrame,
    batting: pd.DataFrame,
    *,
    final_only: bool = False,
    platoon_by_team: pd.DataFrame | None = None,
    roll_df: pd.DataFrame | None = None,
    pen_df: pd.DataFrame | None = None,
    kalshi_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge schedule games with pitcher and team stat rows.

    Games are kept even when a probable starter cannot be matched to the
    pitcher stats table or kbb/xFIP are missing; in that case ``home_sp_kbb``,
    ``away_sp_kbb``, ``home_sp_xfip``, ``away_sp_xfip``, and the derived diff
    columns are NaN. See ``sp_season_stats_complete``.

    When the pitcher table includes ``_stats_season`` (prior-year fallback),
    ``home_sp_stats_season`` / ``away_sp_stats_season`` record which season each
    side's kbb/xFIP row came from (nullable when unknown).

    Optional extended frames (merge on ``game_pk`` where applicable):
    rolling SP metrics, bullpen season/roll14, Kalshi implieds.
    Platoon offense uses ``platoon_by_team`` indexed by FG ``Team`` with
    ``wRC_plus_vs_RHP`` / ``wRC_plus_vs_LHP``.
    """
    if games.empty:
        return pd.DataFrame()

    g = games.copy()
    g["detailed_state"] = g["detailed_state"].astype(str)
    if final_only:
        g = g.loc[g["detailed_state"].isin(_FINAL_STATES)].copy()
    if g.empty:
        logger.warning("No games in range after filters.")
        return pd.DataFrame()

    pitch = pitchers.drop_duplicates(subset=["Name"], keep="first").set_index("Name")
    bat = batting.drop_duplicates(subset=["Team"], keep="first").set_index("Team")

    def row_stats_season(matched: str | None) -> Any:
        if matched is None or matched not in pitch.index:
            return pd.NA
        if _STATS_SEASON_COL not in pitch.columns:
            return pd.NA
        v = pitch.at[matched, _STATS_SEASON_COL]
        if pd.isna(v):
            return pd.NA
        try:
            return int(v)
        except (TypeError, ValueError):
            return pd.NA

    home_wrc: list[float | None] = []
    away_wrc: list[float | None] = []
    home_kbb: list[float | None] = []
    away_kbb: list[float | None] = []
    home_xfip: list[float | None] = []
    away_xfip: list[float | None] = []
    home_win: list[float | None] = []
    home_platoon: list[float] = []
    away_platoon: list[float] = []
    sp_season_stats_complete: list[bool] = []
    home_sp_stats_season: list[Any] = []
    away_sp_stats_season: list[Any] = []

    keep_idx: list[Any] = []
    incomplete_sp_season = 0

    for i, row in g.iterrows():
        hs = row.get("home_score")
        aws = row.get("away_score")
        has_scores = (
            hs is not None
            and aws is not None
            and not pd.isna(hs)
            and not pd.isna(aws)
        )
        state = str(row.get("detailed_state") or "")
        is_decided = state in _FINAL_STATES
        hi = fangraphs_team_abbrev(str(row.get("home_team_name") or ""))
        ai = fangraphs_team_abbrev(str(row.get("away_team_name") or ""))
        if hi is None or ai is None:
            logger.debug("Skip game_pk=%s: unknown team mapping", row.get("game_pk"))
            continue
        if hi not in bat.index or ai not in bat.index:
            logger.debug(
                "Skip game_pk=%s: missing team batting for %s or %s",
                row.get("game_pk"),
                hi,
                ai,
            )
            continue

        hm = match_player_names(row.get("home_probable_pitcher"), pitchers)
        am = match_player_names(row.get("away_probable_pitcher"), pitchers)

        hk = ak = hx = ax = None
        if hm is not None and hm in pitch.index:
            hk = pitch.loc[hm, "kbb"]
            hx = pitch.loc[hm, "xFIP"]
        if am is not None and am in pitch.index:
            ak = pitch.loc[am, "kbb"]
            ax = pitch.loc[am, "xFIP"]

        if hm is None or am is None:
            logger.debug(
                "game_pk=%s: pitcher name match incomplete (home=%r away=%r)",
                row.get("game_pk"),
                hm,
                am,
            )
        hk_f = _to_float_stat(hk)
        ak_f = _to_float_stat(ak)
        hx_f = _to_float_stat(hx)
        ax_f = _to_float_stat(ax)
        sp_ok = all(math.isfinite(x) for x in (hk_f, ak_f, hx_f, ax_f))
        if not sp_ok:
            incomplete_sp_season += 1
            logger.debug(
                "game_pk=%s: retaining row with incomplete SP season stats (kbb/xFIP)",
                row.get("game_pk"),
            )

        hw = float(bat.loc[hi, "wRC+"])
        aw = float(bat.loc[ai, "wRC+"])

        away_throws = row.get("away_sp_throws")
        home_throws = row.get("home_sp_throws")
        home_platoon.append(
            _platoon_value(platoon_by_team, hi, away_throws)
        )
        away_platoon.append(
            _platoon_value(platoon_by_team, ai, home_throws)
        )

        home_wrc.append(hw)
        away_wrc.append(aw)
        home_kbb.append(hk_f)
        away_kbb.append(ak_f)
        home_xfip.append(hx_f)
        away_xfip.append(ax_f)
        sp_season_stats_complete.append(sp_ok)
        home_sp_stats_season.append(row_stats_season(hm))
        away_sp_stats_season.append(row_stats_season(am))
        if has_scores and is_decided:
            home_win.append(1.0 if float(hs) > float(aws) else 0.0)
        else:
            home_win.append(float("nan"))
        keep_idx.append(i)

    if not keep_idx:
        return pd.DataFrame()

    out = g.loc[keep_idx].reset_index(drop=True)
    out["home_wrc_plus"] = home_wrc
    out["away_wrc_plus"] = away_wrc
    out["home_sp_kbb"] = home_kbb
    out["away_sp_kbb"] = away_kbb
    out["home_sp_xfip"] = home_xfip
    out["away_sp_xfip"] = away_xfip
    out["home_win"] = home_win
    out["sp_season_stats_complete"] = sp_season_stats_complete
    out["home_sp_stats_season"] = pd.array(home_sp_stats_season, dtype="Int64")
    out["away_sp_stats_season"] = pd.array(away_sp_stats_season, dtype="Int64")
    out["sp_kbb_diff"] = out["home_sp_kbb"] - out["away_sp_kbb"]
    out["sp_xfip_diff"] = out["home_sp_xfip"] - out["away_sp_xfip"]
    out["offense_diff"] = out["home_wrc_plus"] - out["away_wrc_plus"]
    out["home_offense_platoon"] = home_platoon
    out["away_offense_platoon"] = away_platoon
    out["offense_platoon_diff"] = out["home_offense_platoon"] - out["away_offense_platoon"]

    def _merge_extra(df: pd.DataFrame | None, how: str = "left") -> None:
        nonlocal out
        if df is None or df.empty:
            return
        key = "game_pk"
        if key not in out.columns or key not in df.columns:
            return
        extra_cols = [c for c in df.columns if c != key]
        out = out.merge(df[[key] + extra_cols], on=key, how=how)

    _merge_extra(roll_df)
    _merge_extra(pen_df)
    _merge_extra(kalshi_df)

    for col in (
        "home_sp_kbb_roll14",
        "away_sp_kbb_roll14",
        "home_sp_xfip_roll14",
        "away_sp_xfip_roll14",
        "home_pen_season_fip",
        "away_pen_season_fip",
        "home_pen_season_era",
        "away_pen_season_era",
        "home_pen_roll14_fip",
        "away_pen_roll14_fip",
        "home_pen_roll14_era",
        "away_pen_roll14_era",
        "home_pen_roll14_minus_season_fip",
        "away_pen_roll14_minus_season_fip",
        "kalshi_home_implied",
        "kalshi_away_implied",
        "edge_vs_model",
    ):
        if col not in out.columns:
            out[col] = float("nan")

    if incomplete_sp_season:
        logger.info(
            "Retained %s games with incomplete SP season kbb/xFIP (NaN pitcher columns; "
            "filter with sp_season_stats_complete where needed)",
            incomplete_sp_season,
        )
    logger.info("Dataset rows after merge: %s", len(out))
    return out
