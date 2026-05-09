"""Unit tests for takit.trend."""

from __future__ import annotations

import pytest
import polars as pl

from takit.trend import donchian_channels


OHLC = pl.DataFrame({
    "high":  [11.0, 11.5, 11.8, 11.5, 12.0, 12.2, 11.8, 11.2, 11.5, 12.0],
    "low":   [ 9.5, 10.0, 10.5, 10.2, 10.8, 11.0, 10.5, 10.0, 10.2, 10.8],
})


class TestDonchianChannels:
    def test_returns_three_columns(self) -> None:
        result = donchian_channels(OHLC, 3)
        assert "dc_upper_3" in result.columns
        assert "dc_lower_3" in result.columns
        assert "dc_middle_3" in result.columns

    def test_upper_gte_lower(self) -> None:
        result = donchian_channels(OHLC, 3)
        for upper, lower in zip(
            result["dc_upper_3"].drop_nulls().to_list(),
            result["dc_lower_3"].drop_nulls().to_list(),
        ):
            assert upper >= lower

    def test_middle_is_midpoint(self) -> None:
        result = donchian_channels(OHLC, 3)
        for upper, lower, mid in zip(
            result["dc_upper_3"].drop_nulls().to_list(),
            result["dc_lower_3"].drop_nulls().to_list(),
            result["dc_middle_3"].drop_nulls().to_list(),
        ):
            assert mid == pytest.approx((upper + lower) / 2.0)

    def test_warm_up_is_null(self) -> None:
        result = donchian_channels(OHLC, 3)
        assert result["dc_upper_3"][0] is None
        assert result["dc_upper_3"][1] is None

    def test_output_length_matches_input(self) -> None:
        result = donchian_channels(OHLC, 3)
        assert len(result) == len(OHLC)
