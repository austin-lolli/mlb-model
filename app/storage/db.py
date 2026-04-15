"""Parquet persistence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_to_parquet(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
