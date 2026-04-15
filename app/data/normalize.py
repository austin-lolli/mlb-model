"""Normalize player and team strings for joins."""

from __future__ import annotations

import re
import unicodedata

from app.config import MLB_TEAM_NAME_TO_FG_ABBR

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_player_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _NON_ALNUM.sub("", s)
    return s


def _mlb_team_compact(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[\s\-]+", " ", s)
    s = _NON_ALNUM.sub("", s.replace(" ", ""))
    return s


def normalize_team_name(name: str) -> str:
    """
    Known MLB API names map to FanGraphs team abbreviations; otherwise return
    a compact alphanumeric key for debugging.
    """
    compact = _mlb_team_compact(name)
    return _apply_team_alias(compact)


def fangraphs_team_abbrev(mlb_team_name: str) -> str | None:
    """FanGraphs `Team` column value for this MLB API team name, or None."""
    compact = _mlb_team_compact(mlb_team_name)
    return MLB_TEAM_NAME_TO_FG_ABBR.get(compact)


def _apply_team_alias(compact: str) -> str:
    if compact in MLB_TEAM_NAME_TO_FG_ABBR:
        return MLB_TEAM_NAME_TO_FG_ABBR[compact]
    return compact
