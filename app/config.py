"""Application defaults and constants."""

import os
from datetime import datetime
from pathlib import Path


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# When False (default), pitcher/team stats use MLB Stats API only — no FanGraphs request.
# Set USE_FANGRAPHS=1 to try pybaseball first (xFIP, team wRC+); many networks get HTTP 403.
USE_FANGRAPHS = _env_truthy("USE_FANGRAPHS", default=False)

# When True (default), ``fetch_pitcher_stats`` appends prior-season rows for Names missing from
# the current year so schedule probables still match early in the season. Set to 0 to disable.
PITCHER_PRIOR_YEAR_FALLBACK = _env_truthy("PITCHER_PRIOR_YEAR_FALLBACK", default=True)

# Season and paths
DEFAULT_SEASON = 2026
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATS_TABLES_DIR = DATA_DIR / "stats_tables"


def games_parquet_path(season: int) -> Path:
    """Per-season games dataset: ``data/games_{season}.parquet``."""
    return DATA_DIR / f"games_{season}.parquet"


def default_live_season() -> int:
    """Season for ``live`` when ``--season`` omitted: ``PIPELINE_SEASON`` env or calendar year."""
    v = os.environ.get("PIPELINE_SEASON")
    if v and str(v).strip().isdigit():
        return int(str(v).strip())
    return datetime.now().year


# Legacy default filename (prefer ``games_parquet_path(season)``).
DEFAULT_PARQUET_NAME = f"games_{DEFAULT_SEASON}.parquet"

# MLB Stats API
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_SPORT_ID_MLB = 1

# Fuzzy player name matching (0-100 scale)
FUZZY_MATCH_THRESHOLD = 90

# MLB full team name (lowercase, collapsed) -> FanGraphs Team abbreviation
# Used to join schedule rows to team_batting.
MLB_TEAM_NAME_TO_FG_ABBR: dict[str, str] = {
    "arizonadiamondbacks": "ARI",
    "atlantabraves": "ATL",
    "baltimoreorioles": "BAL",
    "bostonredsox": "BOS",
    "chicagocubs": "CHC",
    "chicagowhitesox": "CHW",
    "cincinnatireds": "CIN",
    "clevelandguardians": "CLE",
    "coloradorockies": "COL",
    "detroittigers": "DET",
    "houstonastros": "HOU",
    "kansascityroyals": "KC",
    "losangelesangels": "LAA",
    "losangelesdodgers": "LAD",
    "miamimarlins": "MIA",
    "milwaukeebrewers": "MIL",
    "minnesotatwins": "MIN",
    "newyorkmets": "NYM",
    "newyorkyankees": "NYY",
    "athletics": "ATH",
    "philadelphiaphillies": "PHI",
    "pittsburghpirates": "PIT",
    "sanfranciscogiants": "SFG",
    "seattlemariners": "SEA",
    "stlouiscardinals": "STL",
    "tampabayrays": "TBR",
    "texasrangers": "TEX",
    "torontobluejays": "TOR",
    "washingtonnationals": "WSN",
    # Historical / alternate strings sometimes seen in API
    "oaklandathletics": "ATH",
    "sandiegopadres": "SDP",
}

# statsapi.mlb.com team abbreviations sometimes differ from FanGraphs `Team` codes.
MLB_API_ABBR_TO_FG_ABBR: dict[str, str] = {
    "SD": "SDP",
    "SF": "SFG",
    "TB": "TBR",
    "WSH": "WSN",
    "CWS": "CHW",
    "AZ": "ARI",
    "ATH": "ATH",
}

# FG-style team code -> Baseball-Reference `team=` param for split.cgi (only where different).
FG_TEAM_TO_BR_TEAM: dict[str, str] = {
    "CHW": "CWS",
}

# Kalshi production API (public market data).
KALSHI_API_BASE = os.environ.get(
    "KALSHI_API_BASE",
    "https://api.elections.kalshi.com/trade-api/v2",
)

# Rolling / bullpen cache under data dir
CACHE_DIR = DATA_DIR / "cache"

# Reliever if GS / max(G,1) below this (season classification).
RELIEVER_GS_RATIO_MAX = 0.15

# Rolling window (calendar days before game date, exclusive of game day).
ROLLING_DAYS = 14
PEN_ROLLING_DAYS = 14


def _env_float(name: str, default: float = 0.0) -> float:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(str(v).strip())
    except ValueError:
        return default


def _env_int(name: str, default: int = 0) -> int:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(str(v).strip())
    except ValueError:
        return default


# Throttle MLB Stats API when filling game-log cache (seconds between requests on cache miss).
# Set GAMELOG_FETCH_DELAY_SEC=0.05–0.15 if you see HTTP 429 during large backfills.
GAMELOG_FETCH_DELAY_SEC = _env_float("GAMELOG_FETCH_DELAY_SEC", 0.0)

# Log every N network game-log fetches at INFO (0 = off). Useful for long multi-year runs.
GAMELOG_LOG_EVERY = _env_int("GAMELOG_LOG_EVERY", 0)
