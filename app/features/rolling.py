"""As-of rolling pitcher metrics from cached MLB game logs (no future leakage)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.config import ROLLING_DAYS
from app.data.fetch_stats_mlb import FIP_C_F, _fip
from app.data.pitcher_gamelog import load_pitching_gamelog_df

logger = logging.getLogger(__name__)


def _window_mask(dates: pd.Series, asof: pd.Timestamp, days: int) -> pd.Series:
    """Games strictly before asof, within the prior ``days`` calendar days (inclusive of earliest)."""
    lo = asof - pd.Timedelta(days=days)
    return (dates >= lo) & (dates < asof)


def _agg_sp_roll(sub: pd.DataFrame) -> dict[str, float]:
    ip = float(sub["ip"].sum())
    bf = int(sub["bf"].sum())
    so = int(sub["so"].sum())
    bb = int(sub["bb"].sum())
    hr = int(sub["hr"].sum())
    hbp = int(sub["hbp"].sum())
    k_pct = 100.0 * so / bf if bf else float("nan")
    bb_pct = 100.0 * bb / bf if bf else float("nan")
    kbb = k_pct - bb_pct if bf else float("nan")
    fip = _fip(hr, bb, hbp, so, ip)
    return {"kbb_roll14": kbb, "xfip_roll14": fip, "roll_ip": ip}


def compute_sp_rolling_for_games(
    games: pd.DataFrame,
    season: int,
    *,
    days: int = ROLLING_DAYS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    For each schedule row with probable pitcher id, compute rolling K-BB and FIP/xFIP-style
    column over games in [asof-days, asof) where asof is game_date (no same-day games).
    """
    if games.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "home_sp_kbb_roll14",
                "away_sp_kbb_roll14",
                "home_sp_xfip_roll14",
                "away_sp_xfip_roll14",
            ]
        )

    pids: set[int] = set()
    for c in ("home_probable_pitcher_id", "away_probable_pitcher_id"):
        if c in games.columns:
            s = games[c].dropna()
            pids.update(int(x) for x in s.astype(int))

    logs: dict[int, pd.DataFrame] = {}
    for pid in sorted(pids):
        try:
            logs[pid] = load_pitching_gamelog_df(pid, season, use_cache=use_cache)
        except Exception as e:
            logger.warning("Game log failed pitcher_id=%s: %s", pid, e)
            logs[pid] = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in games.iterrows():
        gpk = row.get("game_pk")
        gd = row.get("game_date")
        if gd is None or pd.isna(gd):
            continue
        asof = pd.Timestamp(gd).normalize()
        hpid = row.get("home_probable_pitcher_id")
        apid = row.get("away_probable_pitcher_id")

        def roll_for(pid) -> tuple[float, float]:
            if pid is None or pd.isna(pid):
                return float("nan"), float("nan")
            pid = int(pid)
            df = logs.get(pid)
            if df is None or df.empty:
                return float("nan"), float("nan")
            m = _window_mask(df["game_date"], asof, days)
            sub = df.loc[m]
            if sub.empty:
                return float("nan"), float("nan")
            a = _agg_sp_roll(sub)
            return float(a["kbb_roll14"]), float(a["xfip_roll14"])

        hk, hx = roll_for(hpid)
        ak, ax = roll_for(apid)
        rows.append(
            {
                "game_pk": int(gpk) if gpk is not None else None,
                "home_sp_kbb_roll14": hk,
                "away_sp_kbb_roll14": ak,
                "home_sp_xfip_roll14": hx,
                "away_sp_xfip_roll14": ax,
            }
        )

    return pd.DataFrame(rows)


def assert_no_future_in_window(
    log: pd.DataFrame,
    asof: pd.Timestamp,
    days: int = ROLLING_DAYS,
) -> None:
    """Sanity check: no game_date >= asof in the rolling window slice."""
    if log.empty:
        return
    m = _window_mask(log["game_date"], asof, days)
    sub = log.loc[m]
    if sub.empty:
        return
    assert (sub["game_date"] < asof).all(), "rolling window includes same-day or future games"
