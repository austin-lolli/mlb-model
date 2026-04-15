"""Team batting platoon splits (vs RHP / vs LHP) from Baseball-Reference HTML."""

from __future__ import annotations

import logging
import os
import time
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.config import FG_TEAM_TO_BR_TEAM

logger = logging.getLogger(__name__)

# Sports Reference often returns 403 for naive clients; mimic a desktop Chrome request.
_BR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Sec-CH-UA": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
}

_BR_DELAY_SEC = float(os.environ.get("BR_PLATOON_DELAY_SEC", "0.6"))
_BR_MAX_RETRIES = int(os.environ.get("BR_PLATOON_MAX_RETRIES", "5"))
_BR_429_BASE_SLEEP_SEC = float(os.environ.get("BR_PLATOON_429_BASE_SLEEP_SEC", "2.0"))
_BR_RETRY_403 = os.environ.get("BR_PLATOON_RETRY_403", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
_BR_CURL_CFFI = os.environ.get("BR_PLATOON_CURL_CFFI", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
_BR_CURL_IMPERSONATE = os.environ.get("BR_PLATOON_CURL_IMPERSONATE", "chrome131").strip() or "chrome131"


def _retryable_status(code: int) -> bool:
    if code in (429, 503):
        return True
    if code == 403 and _BR_RETRY_403:
        return True
    return False


def _requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_BR_HEADERS)
    return s


def _prime_br_session(session: requests.Session) -> None:
    """Fetch homepage so the session may receive cookies before split.cgi."""
    try:
        r = session.get(
            "https://www.baseball-reference.com/",
            headers={
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            },
            timeout=60,
        )
        logger.debug("BR session prime: HTTP %s", r.status_code)
    except OSError as exc:
        logger.warning("BR session prime failed: %s", exc)


def _get_br_requests(session: requests.Session, url: str, referer: str) -> requests.Response:
    """GET with exponential backoff on HTTP 403 / 429 / 503 (403 often bot-filter)."""
    extra = {
        "Referer": referer,
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
    }
    attempt = 0
    while True:
        r = session.get(url, headers=extra, timeout=90)
        if _retryable_status(r.status_code) and attempt < _BR_MAX_RETRIES:
            wait = _BR_429_BASE_SLEEP_SEC * (2**attempt)
            logger.warning(
                "BR platoon GET retry: HTTP %s; sleeping %.1fs (%s/%s) %s",
                r.status_code,
                wait,
                attempt + 1,
                _BR_MAX_RETRIES,
                url,
            )
            time.sleep(wait)
            attempt += 1
            continue
        return r


def _get_br_curl(curl_requests_mod, url: str, referer: str):
    """TLS/browser impersonation via curl_cffi (module from ``import curl_cffi.requests``)."""
    headers = {**_BR_HEADERS, "Referer": referer}
    attempt = 0
    while True:
        r = curl_requests_mod.get(
            url,
            headers=headers,
            impersonate=_BR_CURL_IMPERSONATE,
            timeout=90,
        )
        if _retryable_status(r.status_code) and attempt < _BR_MAX_RETRIES:
            wait = _BR_429_BASE_SLEEP_SEC * (2**attempt)
            logger.warning(
                "BR platoon GET (curl_cffi) retry: HTTP %s; sleeping %.1fs (%s/%s) %s",
                r.status_code,
                wait,
                attempt + 1,
                _BR_MAX_RETRIES,
                url,
            )
            time.sleep(wait)
            attempt += 1
            continue
        return r


def _br_team_param(fg_team: str) -> str:
    return FG_TEAM_TO_BR_TEAM.get(fg_team, fg_team)


def _parse_ops(val) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return float("nan")
    s = str(val).strip()
    if not s or s in (".", "-"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _platoon_ops_from_html(html: str) -> tuple[float, float] | None:
    soup = BeautifulSoup(html, "lxml")
    # "Platoon Splits" caption; table follows standard BR layout.
    table = None
    for cap in soup.find_all("caption"):
        text = cap.get_text(strip=True).lower()
        if "platoon" in text:
            table = cap.find_parent("table")
            break
    if table is None:
        for tbl in soup.find_all("table", id=lambda x: x and "team_batting" in str(x)):
            cap = tbl.find("caption")
            if cap and "platoon" in cap.get_text(strip=True).lower():
                table = tbl
                break
    if table is None:
        return None

    try:
        dfs = pd.read_html(StringIO(str(table)))
    except ValueError:
        return None
    if not dfs:
        return None
    df = dfs[0]
    # Normalize column names (BR may use multi-index).
    df.columns = [str(c).strip() for c in df.columns.get_level_values(-1)]
    split_col = None
    for cand in ("Split", "split", "Unnamed: 0"):
        if cand in df.columns:
            split_col = cand
            break
    if split_col is None:
        return None
    ops_col = None
    for cand in ("OPS", "ops"):
        if cand in df.columns:
            ops_col = cand
            break
    if ops_col is None:
        return None

    def row_ops(label: str) -> float:
        m = df[df[split_col].astype(str).str.contains(label, case=False, na=False)]
        if m.empty:
            return float("nan")
        return _parse_ops(m.iloc[0][ops_col])

    vs_rhp = row_ops("vs RHP")
    vs_lhp = row_ops("vs LHP")
    if pd.isna(vs_rhp) and pd.isna(vs_lhp):
        return None
    return (vs_rhp, vs_lhp)


def fetch_team_platoon_ops_br(season: int, fg_teams: list[str]) -> pd.DataFrame:
    """
    Return FG-style Team abbrev with OPS vs RHP / vs LHP (from BR platoon table).

    wRC+ style indices are applied in build_features against league means per split.
    """
    curl_mod = None
    if _BR_CURL_CFFI:
        try:
            from curl_cffi import requests as curl_mod  # type: ignore[no-redef]
        except ImportError:
            logger.warning(
                "BR_PLATOON_CURL_CFFI is set but curl_cffi is not installed. "
                "Use a pre-built wheel, or unset BR_PLATOON_CURL_CFFI and try from a non-datacenter IP, "
                "or pass --no-platoon-br."
            )
    session = _requests_session()
    if curl_mod is None:
        _prime_br_session(session)

    rows: list[dict] = []
    for team in sorted(set(fg_teams)):
        br = _br_team_param(team)
        url = f"https://www.baseball-reference.com/teams/split.cgi?t=b&team={br}&year={season}"
        referer = f"https://www.baseball-reference.com/teams/{br}/{season}.shtml"
        logger.debug("BR platoon GET %s", url)
        if curl_mod is not None:
            r = _get_br_curl(curl_mod, url, referer)
        else:
            r = _get_br_requests(session, url, referer)
        if r.status_code != 200:
            logger.warning("BR platoon %s %s: HTTP %s", team, season, r.status_code)
            time.sleep(_BR_DELAY_SEC)
            continue
        parsed = _platoon_ops_from_html(r.text)
        if parsed is None:
            logger.warning("BR platoon parse failed: %s %s", team, season)
        else:
            vs_rhp, vs_lhp = parsed
            rows.append({"Team": team, "OPS_vs_RHP": vs_rhp, "OPS_vs_LHP": vs_lhp})
        time.sleep(_BR_DELAY_SEC)

    if not rows:
        return pd.DataFrame(columns=["Team", "OPS_vs_RHP", "OPS_vs_LHP"])
    return pd.DataFrame(rows)


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
