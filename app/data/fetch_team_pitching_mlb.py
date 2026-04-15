"""Per-team season pitching splits from MLB Stats API (for reliever classification)."""

from __future__ import annotations

import functools
import logging
from collections import defaultdict
from typing import Any

from app.config import RELIEVER_GS_RATIO_MAX
from app.data.fetch_stats_mlb import MLB_STATS_BASE, _get_json

logger = logging.getLogger(__name__)


def _team_list() -> list[dict[str, Any]]:
    payload = _get_json(f"{MLB_STATS_BASE}/teams", {"sportId": 1, "activeStatus": "Active"})
    teams = payload.get("teams") or []
    return [t for t in teams if t.get("division") and t.get("sport", {}).get("id") == 1]


def _split_row(sp: dict[str, Any]) -> dict[str, Any] | None:
    team = sp.get("team") or {}
    tid = team.get("id")
    pl = sp.get("player") or {}
    pid = pl.get("id")
    if tid is None or pid is None:
        return None
    st = sp.get("stat") or {}
    gs = int(st.get("gamesStarted") or 0)
    g = int(st.get("gamesPlayed") or st.get("games") or 0)
    return {
        "pitcher_id": int(pid),
        "team_id": int(tid),
        "gamesStarted": gs,
        "gamesPlayed": g,
    }


@functools.lru_cache(maxsize=8)
def season_pitching_player_rows_by_team(season: int) -> dict[int, list[dict[str, Any]]]:
    """
    One GET to ``/stats`` with ``playerPool=all`` (the ``playerPool=team`` query is rejected with HTTP 400).

    Each split includes ``team.id`` and per-player ``gamesStarted`` / ``gamesPlayed`` for that club.
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
    payload = _get_json(url, params)
    stats_block = (payload.get("stats") or [{}])[0]
    splits = stats_block.get("splits") or []
    if len(splits) >= 5000:
        logger.warning(
            "Pitching season splits hit limit=5000; some players may be missing for season %s",
            season,
        )
    by_team: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sp in splits:
        row = _split_row(sp)
        if row is not None:
            by_team[row["team_id"]].append(row)
    return dict(by_team)


def reliever_pitcher_ids_for_team(season: int, team_id: int) -> set[int]:
    """Deprecated path: one API call per team — use ``reliever_ids_from_rows`` with grouped data instead."""
    by_team = season_pitching_player_rows_by_team(season)
    return reliever_ids_from_rows(by_team.get(int(team_id), []))


def reliever_ids_from_rows(rows: list[dict[str, Any]]) -> set[int]:
    rel: set[int] = set()
    for r in rows:
        g = max(int(r["gamesPlayed"]), 1)
        gs = int(r["gamesStarted"])
        if g < 3:
            continue
        if gs / g < RELIEVER_GS_RATIO_MAX:
            rel.add(int(r["pitcher_id"]))
    return rel


def all_team_relief_ids(season: int) -> dict[int, set[int]]:
    # If logs still show per-team ``playerPool=team`` requests, the image is stale — rebuild with ``--no-cache``.
    logger.info(
        "Bullpen reliever pool: one MLB /stats request (playerPool=all), grouped by team — not per-team playerPool=team"
    )
    by_team = season_pitching_player_rows_by_team(season)
    logger.info(
        "Reliever classification: %s pitching splits across %s team keys (season=%s)",
        sum(len(v) for v in by_team.values()),
        len(by_team),
        season,
    )
    teams = _team_list()
    m: dict[int, set[int]] = {}
    for t in teams:
        tid = t.get("id")
        if tid is None:
            continue
        tid = int(tid)
        rows = by_team.get(tid, [])
        m[tid] = reliever_ids_from_rows(rows)
    return m
