"""Team batting platoon splits (vs RHP / vs LHP) from statsapi.mlb.com."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from app.config import MLB_API_ABBR_TO_FG_ABBR
from app.data.fetch_stats_mlb import MLB_STATS_BASE, _team_list

logger = logging.getLogger(__name__)


def _teams_for_fg_subset(fg_teams: list[str]) -> list[dict[str, Any]]:
    """MLB team dicts whose mapped FG abbrev is in ``fg_teams``; all teams if ``fg_teams`` is empty."""
    want = {x for x in fg_teams if x}
    teams = _team_list()
    if not want:
        return teams
    out: list[dict[str, Any]] = []
    for t in teams:
        abbr = t.get("abbreviation")
        if not abbr:
            continue
        fg = MLB_API_ABBR_TO_FG_ABBR.get(str(abbr), str(abbr))
        if fg in want:
            out.append(t)
    return out


def _split_ops_map(season: int, sit_code: str, mlb_teams: list[dict[str, Any]]) -> dict[str, float]:
    """
    Return FG team -> OPS map for one split code:
    - ``vr``: batting vs right-handed pitchers
    - ``vl``: batting vs left-handed pitchers

    Uses ``GET /api/v1/teams/{id}/stats`` with ``stats=statSplits`` (per-team aggregate).
    The global ``/stats`` endpoint rejects ``playerPool=team`` for ``statSplits`` (400).
    """
    out: dict[str, float] = {}
    for t in mlb_teams:
        tid = t.get("id")
        abbr = t.get("abbreviation")
        if tid is None or not abbr:
            continue
        url = f"{MLB_STATS_BASE}/teams/{tid}/stats"
        params: dict[str, Any] = {
            "stats": "statSplits",
            "group": "hitting",
            "season": season,
            "sitCodes": sit_code,
        }
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            payload = r.json()
        except requests.HTTPError as exc:
            logger.warning("MLB platoon team_id=%s sit=%s: %s", tid, sit_code, exc)
            continue
        except OSError as exc:
            logger.warning("MLB platoon team_id=%s sit=%s: %s", tid, sit_code, exc)
            continue
        stats_block = (payload.get("stats") or [{}])[0]
        splits = stats_block.get("splits") or []
        if not splits:
            continue
        stat = splits[0].get("stat") or {}
        ops_raw = stat.get("ops")
        if ops_raw in (None, ""):
            continue
        try:
            ops = float(str(ops_raw).strip())
        except ValueError:
            continue
        fg = MLB_API_ABBR_TO_FG_ABBR.get(str(abbr), str(abbr))
        out[fg] = ops
    return out


def fetch_team_platoon_ops_mlb(season: int, fg_teams: list[str]) -> pd.DataFrame:
    """
    Return FG-style Team abbrev with OPS vs RHP / vs LHP from MLB statSplits.
    """
    logger.info("MLB platoon split fetch (season=%s)", season)
    mlb_teams = _teams_for_fg_subset(fg_teams)
    ops_vs_r = _split_ops_map(season, "vr", mlb_teams)
    ops_vs_l = _split_ops_map(season, "vl", mlb_teams)
    wanted = sorted(set(fg_teams))
    if not wanted:
        wanted = sorted(set(ops_vs_r.keys()) | set(ops_vs_l.keys()))
    rows: list[dict[str, float | str]] = []
    for team in wanted:
        r = ops_vs_r.get(team)
        l = ops_vs_l.get(team)
        if r is None and l is None:
            continue
        rows.append(
            {
                "Team": team,
                "OPS_vs_RHP": float("nan") if r is None else float(r),
                "OPS_vs_LHP": float("nan") if l is None else float(l),
            }
        )
    out = pd.DataFrame(rows)
    logger.info("MLB platoon rows: %s", len(out))
    return out


def platoon_ops_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add wRC+ style columns (100 = league average) from OPS splits."""
    if df.empty:
        return pd.DataFrame(
            columns=["Team", "wRC_plus_vs_RHP", "wRC_plus_vs_LHP", "OPS_vs_RHP", "OPS_vs_LHP"]
        )
    work = df.copy()
    lg_r = work["OPS_vs_RHP"].mean()
    lg_l = work["OPS_vs_LHP"].mean()
    out = pd.DataFrame(
        {
            "Team": work["Team"],
            "OPS_vs_RHP": work["OPS_vs_RHP"],
            "OPS_vs_LHP": work["OPS_vs_LHP"],
            "wRC_plus_vs_RHP": 100.0 * work["OPS_vs_RHP"] / lg_r if lg_r else float("nan"),
            "wRC_plus_vs_LHP": 100.0 * work["OPS_vs_LHP"] / lg_l if lg_l else float("nan"),
        }
    )
    return out
