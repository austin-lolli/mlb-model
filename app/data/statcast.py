"""
Statcast integration (placeholder).

Future: use pybaseball Statcast helpers (e.g. statcast_pitcher) to compute
rolling xwOBA, hard-hit %, barrel rate by pitcher_id and game date, then merge
into the main dataset on pitcher id + date.
"""


def fetch_statcast_pitcher_features(*_args, **_kwargs):  # noqa: ANN001, ANN002
    raise NotImplementedError("Statcast features are not implemented yet.")
