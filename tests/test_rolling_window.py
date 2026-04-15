"""Rolling window uses only games strictly before as-of date (no leakage)."""

from __future__ import annotations

import unittest

import pandas as pd

from app.config import ROLLING_DAYS
from app.features.rolling import assert_no_future_in_window, _window_mask


class TestRollingWindow(unittest.TestCase):
    def test_window_excludes_game_day_and_future(self) -> None:
        asof = pd.Timestamp("2024-06-15")
        dates = pd.Series(
            pd.to_datetime(
                ["2024-06-01", "2024-06-14", "2024-06-15", "2024-06-16"]
            )
        )
        m = _window_mask(dates, asof, ROLLING_DAYS)
        self.assertTrue(m.iloc[0])
        self.assertTrue(m.iloc[1])
        self.assertFalse(m.iloc[2])
        self.assertFalse(m.iloc[3])

    def test_assert_no_future_in_window_passes(self) -> None:
        asof = pd.Timestamp("2024-06-15")
        log = pd.DataFrame(
            {
                "game_date": pd.to_datetime(["2024-06-01", "2024-06-10"]),
            }
        )
        assert_no_future_in_window(log, asof, days=ROLLING_DAYS)


if __name__ == "__main__":
    unittest.main()
