"""Fetch MLB schedule, scores, and probable pitchers from the MLB Stats API."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from app.config import MLB_SCHEDULE_URL, MLB_SPORT_ID_MLB

logger = logging.getLogger(__name__)


def _probable_full_name(team_side: dict[str, Any]) -> str | None:
    pitcher = team_side.get("probablePitcher") or {}
    if not pitcher:
        return None
    full = pitcher.get("fullName")
    if full:
        return str(full)
    return None


def _probable_pitcher_id(team_side: dict[str, Any]) -> int | None:
    pitcher = team_side.get("probablePitcher") or {}
    pid = pitcher.get("id")
    if pid is None:
        return None
    try:
        return int(pid)
    except (TypeError, ValueError):
        return None


def _team_name(team_side: dict[str, Any]) -> str:
    team = team_side.get("team") or {}
    return str(team.get("name") or "")


def _team_id(team_side: dict[str, Any]) -> int | None:
    team = team_side.get("team") or {}
    tid = team.get("id")
    if tid is None:
        return None
    try:
        return int(tid)
    except (TypeError, ValueError):
        return None


def _runs_from_linescore(game: dict[str, Any]) -> tuple[int | None, int | None]:
    ls = game.get("linescore") or {}
    teams_ls = ls.get("teams") or {}
    home_ls = teams_ls.get("home") or {}
    away_ls = teams_ls.get("away") or {}
    hr = home_ls.get("runs")
    ar = away_ls.get("runs")
    if hr is None and ar is None:
        return None, None
    return (
        int(hr) if hr is not None else None,
        int(ar) if ar is not None else None,
    )


def _parse_games_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date_block in payload.get("dates") or []:
        game_date = date_block.get("date")
        for game in date_block.get("games") or []:
            teams = game.get("teams") or {}
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            status = game.get("status") or {}
            detailed = str(status.get("detailedState") or "")
            home_score = home.get("score")
            away_score = away.get("score")
            if home_score is None and away_score is None:
                lh, la = _runs_from_linescore(game)
                home_score, away_score = lh, la
            rows.append(
                {
                    "game_pk": game.get("gamePk"),
                    "game_date": game_date,
                    "detailed_state": detailed,
                    "home_team_name": _team_name(home),
                    "away_team_name": _team_name(away),
                    "home_team_id": _team_id(home),
                    "away_team_id": _team_id(away),
                    "home_score": int(home_score) if home_score is not None else None,
                    "away_score": int(away_score) if away_score is not None else None,
                    "home_probable_pitcher": _probable_full_name(home),
                    "away_probable_pitcher": _probable_full_name(away),
                    "home_probable_pitcher_id": _probable_pitcher_id(home),
                    "away_probable_pitcher_id": _probable_pitcher_id(away),
                    "home_sp_throws": None,
                    "away_sp_throws": None,
                }
            )
    return rows


def _fetch_pitch_hands(person_ids: list[int]) -> dict[int, str]:
    """Map MLB personId -> throws code R / L / S (batting side sometimes used for switch)."""
    out: dict[int, str] = {}
    if not person_ids:
        return out
    # Batch in chunks (API accepts comma-separated personIds).
    chunk_size = 50
    for i in range(0, len(person_ids), chunk_size):
        chunk = person_ids[i : i + chunk_size]
        params = {"personIds": ",".join(str(x) for x in chunk)}
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/people",
            params=params,
            timeout=60,
        )
        r.raise_for_status()
        for p in r.json().get("people") or []:
            pid = p.get("id")
            if pid is None:
                continue
            ph = p.get("pitchHand") or {}
            code = ph.get("code")
            if code:
                try:
                    out[int(pid)] = str(code).strip().upper()[:1]
                except (TypeError, ValueError):
                    continue
    return out


def fetch_games(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull schedule rows for [start_date, end_date] inclusive (YYYY-MM-DD).

    Uses hydrate=probablePitcher. Scores are present when the API provides them
    (typically for final or in-progress games).
    """
    params = {
        "sportId": MLB_SPORT_ID_MLB,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "probablePitcher,linescore",
    }
    logger.info("GET schedule %s — %s", start_date, end_date)
    r = requests.get(MLB_SCHEDULE_URL, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    rows = _parse_games_payload(payload)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["game_pk"] = df["game_pk"].astype("Int64")
        for col in ("home_team_id", "away_team_id", "home_probable_pitcher_id", "away_probable_pitcher_id"):
            if col in df.columns:
                df[col] = df[col].astype("Int64")
        ids: list[int] = []
        for c in ("home_probable_pitcher_id", "away_probable_pitcher_id"):
            s = df[c].dropna().astype(int)
            ids.extend(s.tolist())
        id_set = sorted(set(ids))
        if id_set:
            hands = _fetch_pitch_hands(id_set)
            df["home_sp_throws"] = df["home_probable_pitcher_id"].map(
                lambda x: hands.get(int(x)) if pd.notna(x) else None
            )
            df["away_sp_throws"] = df["away_probable_pitcher_id"].map(
                lambda x: hands.get(int(x)) if pd.notna(x) else None
            )
    logger.info("Schedule rows: %s", len(df))
    return df
