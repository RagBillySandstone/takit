"""Unit tests for takit.volatility."""

from __future__ import annotations

import pytest
import polars as pl

from takit.volatility import true_range, atr, bollinger_bands, keltner_channels


OHLC = pl.DataFrame({
    "open":  [10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.0, 10.5, 10.8, 11.2],
    "high":  [11.0, 11.5, 11.8, 11.5, 12.0, 12.2, 11.8, 11.2, 11.5, 12.0],
    "low":   [ 9.5, 10.0, 10.5, 10.2, 10.8, 11.0, 10.5, 10.0, 10.2, 10.8],
    "close": [10.5, 11.0, 10.8, 11.2, 11.5, 11.0, 10.5, 10.8, 11.2, 11.5],
})
CLOSE = OHLC["close"]


class TestTrueRange:
    def test_first_bar_equals_hl(self) -> None:
        # First bar: no prev_close, so TR = high - low.
        result = true_range(OHLC)
        assert result[0] == pytest.approx(OHLC["high"][0] - OHLC["low"][0])

    def test_all_non_negative(self) -> None:
        result = true_range(OHLC)
        assert all(v >= 0.0 for v in result.to_list())

    def test_output_length_matches_input(self) -> None:
        assert len(true_range(OHLC)) == len(OHLC)


class TestATR:
    def test_warm_up_is_null(self) -> None:
        result = atr(OHLC, 5)
        for i in range(4):
            assert result[i] is None

    def test_all_valid_values_non_negative(self) -> None:
        result = atr(OHLC, 3)
        for v in result.drop_nulls().to_list():
            assert v >= 0.0

    def test_output_length_matches_input(self) -> None:
        assert len(atr(OHLC, 3)) == len(OHLC)


class TestBollingerBands:
    def test_returns_five_columns(self) -> None:
        result = bollinger_bands(CLOSE, 3)
        assert f"bb_middle_3" in result.columns
        assert f"bb_upper_3" in result.columns
        assert f"bb_lower_3" in result.columns
        assert f"bb_pct_b_3" in result.columns
        assert f"bb_width_3" in result.columns

    def test_upper_gte_middle_gte_lower(self) -> None:
        result = bollinger_bands(CLOSE, 3)
        for upper, mid, lower in zip(
            result["bb_upper_3"].drop_nulls().to_list(),
            result["bb_middle_3"].drop_nulls().to_list(),
            result["bb_lower_3"].drop_nulls().to_list(),
        ):
            assert upper >= mid >= lower

    def test_output_length_matches_input(self) -> None:
        result = bollinger_bands(CLOSE, 3)
        assert len(result) == len(CLOSE)

    def test_period_1_raises(self) -> None:
        with pytest.raises(ValueError):
            bollinger_bands(CLOSE, 1)


class TestKeltnerChannels:
    def test_returns_three_columns(self) -> None:
        result = keltner_channels(OHLC, ema_period=3, atr_period=3)
        assert "kc_middle" in result.columns
        assert "kc_upper" in result.columns
        assert "kc_lower" in result.columns

    def test_upper_gte_middle_gte_lower(self) -> None:
        result = keltner_channels(OHLC, ema_period=3, atr_period=3)
        for upper, mid, lower in zip(
            result["kc_upper"].drop_nulls().to_list(),
            result["kc_middle"].drop_nulls().to_list(),
            result["kc_lower"].drop_nulls().to_list(),
        ):
            assert upper >= mid >= lower

    def test_output_length_matches_input(self) -> None:
        result = keltner_channels(OHLC, ema_period=3, atr_period=3)
        assert len(result) == len(OHLC)
