"""Match probable-pitcher strings from the schedule to FanGraphs stat rows."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz

from app.config import FUZZY_MATCH_THRESHOLD
from app.data.normalize import normalize_player_name

logger = logging.getLogger(__name__)


def match_player_names(
    game_name: str | None,
    stats_df: pd.DataFrame,
    name_col: str = "Name",
) -> Optional[str]:
    """
    Return the canonical Name from stats_df for this pitcher, or None.

    Exact match on normalized name first; else rapidfuzz token_sort_ratio
    against name_col (threshold FUZZY_MATCH_THRESHOLD).
    """
    if not game_name or stats_df is None or stats_df.empty or name_col not in stats_df.columns:
        return None
    target = normalize_player_name(game_name)
    if not target:
        return None

    names = stats_df[name_col].astype(str)
    norm_col = "_norm_name"
    work = stats_df.assign(**{norm_col: names.map(normalize_player_name)})
    exact = work.loc[work[norm_col] == target]
    if not exact.empty:
        return str(exact.iloc[0][name_col])

    best_name: str | None = None
    best_score = -1.0
    for raw, norm in zip(names, work[norm_col]):
        if not norm:
            continue
        score = float(fuzz.token_sort_ratio(target, norm))
        if score > best_score:
            best_score = score
            best_name = raw
    if best_name is not None and best_score >= FUZZY_MATCH_THRESHOLD:
        return str(best_name)
    logger.debug("No fuzzy match for %r (best=%.1f)", game_name, best_score)
    return None
