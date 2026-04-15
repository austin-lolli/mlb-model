"""Cached MLB pitching game logs per player-season (statsapi)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from app.config import CACHE_DIR, GAMELOG_FETCH_DELAY_SEC, GAMELOG_LOG_EVERY
from app.data.fetch_stats_mlb import MLB_STATS_BASE, _parse_innings

logger = logging.getLogger(__name__)

# In-process cache: rolling + bullpen may request the same (pitcher_id, season).
_MEMO: dict[tuple[int, int], pd.DataFrame] = {}
_NET_FETCH_COUNT = 0


def _cache_path(person_id: int, season: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"gamelog_pitch_{season}_{person_id}.json"


def fetch_pitching_gamelog_json(person_id: int, season: int) -> dict[str, Any]:
    url = f"{MLB_STATS_BASE}/people/{person_id}/stats"
    params = {"stats": "gameLog", "group": "pitching", "season": season}
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def load_pitching_gamelog_df(person_id: int, season: int, *, use_cache: bool = True) -> pd.DataFrame:
    key = (int(person_id), int(season))
    if use_cache and key in _MEMO:
        return _MEMO[key].copy()

    path = _cache_path(person_id, season)
    if use_cache and path.is_file():
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        global _NET_FETCH_COUNT
        if GAMELOG_FETCH_DELAY_SEC > 0:
            time.sleep(GAMELOG_FETCH_DELAY_SEC)
        payload = fetch_pitching_gamelog_json(person_id, season)
        _NET_FETCH_COUNT += 1
        if GAMELOG_LOG_EVERY > 0 and _NET_FETCH_COUNT % GAMELOG_LOG_EVERY == 0:
            logger.info(
                "Game log fetch #%s (pitcher_id=%s season=%s)",
                _NET_FETCH_COUNT,
                person_id,
                season,
            )
        if use_cache:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

    stats_block = (payload.get("stats") or [{}])[0]
    splits = stats_block.get("splits") or []
    rows: list[dict[str, Any]] = []
    for sp in splits:
        st = sp.get("stat") or {}
        game = sp.get("game") or {}
        team = sp.get("team") or {}
        gpk = game.get("gamePk")
        gdate = sp.get("date") or game.get("gameDate", "")[:10]
        if not gdate:
            continue
        try:
            tid = int(team["id"]) if team.get("id") is not None else None
        except (TypeError, ValueError):
            tid = None
        ip = _parse_innings(st.get("inningsPitched"))
        so = int(st.get("strikeOuts") or 0)
        bb = int(st.get("baseOnBalls") or 0)
        hr = int(st.get("homeRuns") or 0)
        hbp = int(st.get("hitBatsmen") or 0)
        er = int(st.get("earnedRuns") or 0)
        bf = int(st.get("battersFaced") or 0)
        gs_game = st.get("gamesStarted")
        try:
            gs_game_i = int(gs_game) if gs_game is not None else 0
        except (TypeError, ValueError):
            gs_game_i = 0
        rows.append(
            {
                "pitcher_id": int(person_id),
                "game_pk": int(gpk) if gpk is not None else None,
                "game_date": str(gdate)[:10],
                "team_id": tid,
                "ip": ip,
                "so": so,
                "bb": bb,
                "hr": hr,
                "hbp": hbp,
                "er": er,
                "bf": bf,
                "games_started_game": gs_game_i,
            }
        )
    if not rows:
        empty = pd.DataFrame(
            columns=[
                "pitcher_id",
                "game_pk",
                "game_date",
                "team_id",
                "ip",
                "so",
                "bb",
                "hr",
                "hbp",
                "er",
                "bf",
                "games_started_game",
            ]
        )
        if use_cache:
            _MEMO[key] = empty
        return empty
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    out = df.sort_values("game_date").reset_index(drop=True)
    if use_cache:
        _MEMO[key] = out
    return out
