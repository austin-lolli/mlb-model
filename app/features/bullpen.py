"""Team bullpen aggregates: season-to-date and rolling relief-only metrics (as-of game_date)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.config import PEN_ROLLING_DAYS
from app.data.fetch_stats_mlb import _fip
from app.data.fetch_team_pitching_mlb import all_team_relief_ids
from app.data.pitcher_gamelog import load_pitching_gamelog_df

logger = logging.getLogger(__name__)


def _relief_rows_for_team(df: pd.DataFrame, team_id: int, reliever_ids: set[int]) -> pd.DataFrame:
    if df.empty:
        return df
    m = (df["team_id"] == team_id) & (df["pitcher_id"].isin(reliever_ids))
    sub = df.loc[m]
    if sub.empty:
        return sub
    # Non-start appearances only when API provides per-game GS.
    gs = sub["games_started_game"]
    return sub.loc[(gs == 0) | gs.isna()]


def _agg_pen(sub: pd.DataFrame) -> tuple[float, float]:
    ip = float(sub["ip"].sum())
    if ip <= 0:
        return float("nan"), float("nan")
    er = float(sub["er"].sum())
    era = er / ip * 9.0
    hr = int(sub["hr"].sum())
    bb = int(sub["bb"].sum())
    hbp = int(sub["hbp"].sum())
    so = int(sub["so"].sum())
    fip = _fip(hr, bb, hbp, so, ip)
    return float(fip), float(era)


def _season_start_mask(dates: pd.Series, season: int, asof: pd.Timestamp) -> pd.Series:
    return (dates >= pd.Timestamp(f"{season}-01-01")) & (dates < asof)


def _roll_mask(dates: pd.Series, asof: pd.Timestamp, days: int) -> pd.Series:
    lo = asof - pd.Timedelta(days=days)
    return (dates >= lo) & (dates < asof)


def compute_bullpen_features_for_games(
    games: pd.DataFrame,
    season: int,
    *,
    pen_days: int = PEN_ROLLING_DAYS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Per game row: home/away pen FIP and ERA from relief appearances only,
    season-to-date before game_date and rolling ``pen_days`` before game_date.
    """
    cols = [
        "game_pk",
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
    ]
    if games.empty:
        return pd.DataFrame(columns=cols)

    team_rel = all_team_relief_ids(season)
    all_relief_pids: set[int] = set()
    for s in team_rel.values():
        all_relief_pids |= s

    logs: dict[int, pd.DataFrame] = {}
    for pid in sorted(all_relief_pids):
        try:
            logs[pid] = load_pitching_gamelog_df(pid, season, use_cache=use_cache)
        except Exception as e:
            logger.warning("Bullpen game log failed pitcher_id=%s: %s", pid, e)
            logs[pid] = pd.DataFrame()

    def team_pen(team_id: int | None, asof: pd.Timestamp) -> tuple[float, float, float, float]:
        if team_id is None or pd.isna(team_id):
            return (float("nan"),) * 4
        tid = int(team_id)
        rids = team_rel.get(tid, set())
        if not rids:
            return (float("nan"),) * 4
        season_parts: list[pd.DataFrame] = []
        roll_parts: list[pd.DataFrame] = []
        for pid in rids:
            df = logs.get(pid, pd.DataFrame())
            if df.empty:
                continue
            rel = _relief_rows_for_team(df, tid, rids)
            if rel.empty:
                continue
            sm = _season_start_mask(rel["game_date"], season, asof)
            rm = _roll_mask(rel["game_date"], asof, pen_days)
            season_parts.append(rel.loc[sm])
            roll_parts.append(rel.loc[rm])
        s_df = pd.concat(season_parts, ignore_index=True) if season_parts else pd.DataFrame()
        r_df = pd.concat(roll_parts, ignore_index=True) if roll_parts else pd.DataFrame()
        sfip, sera = _agg_pen(s_df)
        rfip, rera = _agg_pen(r_df)
        return sfip, sera, rfip, rera

    rows: list[dict[str, Any]] = []
    for _, row in games.iterrows():
        gpk = row.get("game_pk")
        gd = row.get("game_date")
        if gd is None or pd.isna(gd):
            continue
        asof = pd.Timestamp(gd).normalize()
        ht = row.get("home_team_id")
        at = row.get("away_team_id")
        hsf, hse, hrf, hre = team_pen(ht, asof)
        asf, ase, arf, are = team_pen(at, asof)
        hdiff = hrf - hsf if pd.notna(hrf) and pd.notna(hsf) else float("nan")
        adiff = arf - asf if pd.notna(arf) and pd.notna(asf) else float("nan")
        rows.append(
            {
                "game_pk": int(gpk) if gpk is not None else None,
                "home_pen_season_fip": hsf,
                "away_pen_season_fip": asf,
                "home_pen_season_era": hse,
                "away_pen_season_era": ase,
                "home_pen_roll14_fip": hrf,
                "away_pen_roll14_fip": arf,
                "home_pen_roll14_era": hre,
                "away_pen_roll14_era": are,
                "home_pen_roll14_minus_season_fip": hdiff,
                "away_pen_roll14_minus_season_fip": adiff,
            }
        )

    return pd.DataFrame(rows)
