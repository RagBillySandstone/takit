"""Unit tests for polarticks.volatility."""

from __future__ import annotations

import polars as pl
import pytest

from polarticks.volatility import (
    atr,
    bollinger_bands,
    chaikin_volatility,
    historical_volatility,
    keltner_channels,
    true_range,
    ulcer_index,
)

OHLC = pl.DataFrame(
    {
        "open": [10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.0, 10.5, 10.8, 11.2],
        "high": [11.0, 11.5, 11.8, 11.5, 12.0, 12.2, 11.8, 11.2, 11.5, 12.0],
        "low": [9.5, 10.0, 10.5, 10.2, 10.8, 11.0, 10.5, 10.0, 10.2, 10.8],
        "close": [10.5, 11.0, 10.8, 11.2, 11.5, 11.0, 10.5, 10.8, 11.2, 11.5],
    }
)
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
        assert "bb_middle_3" in result.columns
        assert "bb_upper_3" in result.columns
        assert "bb_lower_3" in result.columns
        assert "bb_pct_b_3" in result.columns
        assert "bb_width_3" in result.columns

    def test_upper_gte_middle_gte_lower(self) -> None:
        result = bollinger_bands(CLOSE, 3)
        for upper, mid, lower in zip(
            result["bb_upper_3"].drop_nulls().to_list(),
            result["bb_middle_3"].drop_nulls().to_list(),
            result["bb_lower_3"].drop_nulls().to_list(),
            strict=True,
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
            strict=True,
        ):
            assert upper >= mid >= lower

    def test_output_length_matches_input(self) -> None:
        result = keltner_channels(OHLC, ema_period=3, atr_period=3)
        assert len(result) == len(OHLC)


class TestChaikinVolatility:
    def test_output_length_matches_input(self) -> None:
        assert len(chaikin_volatility(OHLC, 3, 3)) == len(OHLC)

    def test_warm_up_is_null(self) -> None:
        result = chaikin_volatility(OHLC, 3, 3)
        assert result[0] is None

    def test_alias_contains_periods(self) -> None:
        assert chaikin_volatility(OHLC, 3, 3).name == "chaikin_vol_3_3"

    def test_invalid_ema_period_raises(self) -> None:
        with pytest.raises(ValueError):
            chaikin_volatility(OHLC, 0, 3)

    def test_expanding_range_yields_positive_cv(self) -> None:
        # High-low range that grows each bar should produce positive CV.
        expanding = pl.DataFrame(
            {
                "high": [float(i) for i in range(1, 25)],
                "low": [0.0] * 24,
                "close": [float(i) * 0.5 for i in range(1, 25)],
            }
        )
        result = chaikin_volatility(expanding, 3, 3)
        valid = result.drop_nulls().to_list()
        assert any(v > 0 for v in valid)


class TestHistoricalVolatility:
    def test_output_length_matches_input(self) -> None:
        close = OHLC["close"]
        assert len(historical_volatility(close, 5)) == len(close)

    def test_warm_up_is_null(self) -> None:
        result = historical_volatility(OHLC["close"], 5)
        assert result[0] is None

    def test_alias_contains_period(self) -> None:
        assert historical_volatility(OHLC["close"], 5).name == "hv_5"

    def test_non_negative(self) -> None:
        result = historical_volatility(OHLC["close"], 5)
        for val in result.drop_nulls().to_list():
            assert val >= 0.0

    def test_flat_series_has_zero_volatility(self) -> None:
        flat = pl.Series([10.0] * 20)
        result = historical_volatility(flat, 5, annualise=False)
        for val in result.drop_nulls().to_list():
            assert val == pytest.approx(0.0, abs=1e-12)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            historical_volatility(OHLC["close"], 1)


class TestUlcerIndex:
    def test_output_length_matches_input(self) -> None:
        assert len(ulcer_index(OHLC["close"], 5)) == len(OHLC)

    def test_warm_up_is_null(self) -> None:
        result = ulcer_index(OHLC["close"], 5)
        assert result[0] is None

    def test_alias_contains_period(self) -> None:
        assert ulcer_index(OHLC["close"], 5).name == "ulcer_5"

    def test_non_negative(self) -> None:
        result = ulcer_index(OHLC["close"], 5)
        for val in result.drop_nulls().to_list():
            assert val >= 0.0

    def test_monotone_up_series_has_zero_ulcer(self) -> None:
        # A series that only goes up has no drawdowns → UI = 0.
        rising = pl.Series([float(i) for i in range(1, 20)])
        result = ulcer_index(rising, 5)
        for val in result.drop_nulls().to_list():
            assert val == pytest.approx(0.0, abs=1e-9)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            ulcer_index(OHLC["close"], 0)
