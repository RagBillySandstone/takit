"""Unit tests for takit.volume."""

from __future__ import annotations

import pytest
import polars as pl
from datetime import datetime, timezone

from takit.volume import vwap


def _make_ohlcv(n: int = 10) -> pl.DataFrame:
    """Build a minimal OHLC+volume DataFrame for testing."""
    return pl.DataFrame({
        "high":   [float(i + 1) for i in range(n)],
        "low":    [float(i) for i in range(n)],
        "close":  [float(i) + 0.5 for i in range(n)],
        "volume": [100] * n,
    })


class TestVWAP:
    def test_output_length_matches_input(self) -> None:
        df = _make_ohlcv(10)
        result = vwap(df)
        assert len(result) == 10

    def test_single_bar_equals_typical_price(self) -> None:
        df = pl.DataFrame({"high": [12.0], "low": [10.0], "close": [11.0], "volume": [100]})
        result = vwap(df)
        # typical_price = (12 + 10 + 11) / 3 = 11.0; VWAP of one bar = 11.0
        assert result[0] == pytest.approx(11.0)

    def test_equal_volume_equals_average_tp(self) -> None:
        # With equal volume each bar, VWAP = cumulative mean of typical price.
        df = pl.DataFrame({
            "high":   [11.0, 12.0],
            "low":    [ 9.0, 10.0],
            "close":  [10.0, 11.0],
            "volume": [100,  100],
        })
        result = vwap(df)
        tp0 = (11.0 + 9.0 + 10.0) / 3.0   # 10.0
        tp1 = (12.0 + 10.0 + 11.0) / 3.0  # 11.0
        assert result[0] == pytest.approx(tp0)
        assert result[1] == pytest.approx((tp0 + tp1) / 2.0)

    def test_session_reset_with_time_column(self) -> None:
        # Two sessions of 3 bars each, reset at hour 22.
        times = [
            datetime(2024, 1, 1, 22, 0, tzinfo=timezone.utc),  # session 1 start
            datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 2,  0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 2, 22, 0, tzinfo=timezone.utc),  # session 2 start
            datetime(2024, 1, 2, 23, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 3,  0, 0, tzinfo=timezone.utc),
        ]
        df = pl.DataFrame({
            "time":   pl.Series(times).cast(pl.Datetime("us", "UTC")),
            "high":   [11.0] * 6,
            "low":    [ 9.0] * 6,
            "close":  [10.0] * 6,
            "volume": [100]  * 6,
        })
        result = vwap(df, session_start_hour=22)
        # All typical prices are 10.0, so VWAP is always 10.0.
        for v in result.to_list():
            assert v == pytest.approx(10.0)
        # After reset, bar 3 should restart accumulation — same value here,
        # but we verify no NaN leaks through from the prior session.
        assert not any(v != v for v in result.to_list())  # NaN check
