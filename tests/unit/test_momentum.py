"""Unit tests for polarticks.momentum."""

from __future__ import annotations

import polars as pl
import pytest

from polarticks.momentum import (
    cci,
    cmf,
    macd,
    mfi,
    roc,
    rsi,
    stochastic,
    tsi,
    ultimate_oscillator,
    williams_r,
)

# A longer series so Wilder-based indicators have enough warm-up bars.
CLOSE = pl.Series(
    "close",
    [
        44.34,
        44.09,
        44.15,
        43.61,
        44.33,
        44.83,
        45.10,
        45.15,
        43.61,
        44.33,
        44.83,
        45.10,
        45.15,
        45.98,
        45.77,
        45.54,
        45.41,
        44.83,
        45.10,
        45.15,
    ],
)
OHLC = pl.DataFrame(
    {
        "open": [44.0] * 20,
        "high": [v + 0.5 for v in CLOSE.to_list()],
        "low": [v - 0.5 for v in CLOSE.to_list()],
        "close": CLOSE.to_list(),
    }
)


class TestRSI:
    def test_warm_up_is_null(self) -> None:
        result = rsi(CLOSE, 14)
        # First 14 bars are null (Wilder smoothing needs period bars to seed).
        for i in range(14):
            assert result[i] is None

    def test_values_in_range(self) -> None:
        result = rsi(CLOSE, 14)
        valid = [v for v in result.to_list() if v is not None]
        assert all(0.0 <= v <= 100.0 for v in valid)

    def test_output_length_matches_input(self) -> None:
        assert len(rsi(CLOSE, 14)) == len(CLOSE)

    def test_all_up_bars_returns_100(self) -> None:
        # A constantly rising series should produce RSI near 100
        # after warm-up (no losses means avg_loss → 0).
        rising = pl.Series("rising", [float(i) for i in range(1, 30)])
        result = rsi(rising, 14)
        valid = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(100.0) for v in valid)

    def test_period_1_raises(self) -> None:
        with pytest.raises(ValueError):
            rsi(CLOSE, 1)


class TestMACD:
    def test_returns_three_columns(self) -> None:
        result = macd(CLOSE)
        assert set(result.columns) == {"macd_line", "macd_signal", "macd_histogram"}

    def test_output_length_matches_input(self) -> None:
        result = macd(CLOSE)
        assert len(result) == len(CLOSE)

    def test_histogram_equals_line_minus_signal(self) -> None:
        result = macd(CLOSE)
        for line, sig, hist in zip(
            result["macd_line"].to_list(),
            result["macd_signal"].to_list(),
            result["macd_histogram"].to_list(),
            strict=True,
        ):
            if line is not None and sig is not None and hist is not None:
                assert hist == pytest.approx(line - sig, abs=1e-9)

    def test_fast_gte_slow_raises(self) -> None:
        with pytest.raises(ValueError):
            macd(CLOSE, fast=26, slow=12)


class TestStochastic:
    def test_returns_two_columns(self) -> None:
        result = stochastic(OHLC)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

    def test_values_in_range(self) -> None:
        result = stochastic(OHLC)
        for v in result["stoch_k"].drop_nulls().to_list():
            assert 0.0 <= v <= 100.0

    def test_output_length_matches_input(self) -> None:
        result = stochastic(OHLC)
        assert len(result) == len(OHLC)


class TestWilliamsR:
    def test_values_in_range(self) -> None:
        result = williams_r(OHLC, 14)
        for v in result.drop_nulls().to_list():
            assert -100.0 <= v <= 0.0

    def test_output_length_matches_input(self) -> None:
        assert len(williams_r(OHLC, 14)) == len(OHLC)

    def test_warm_up_is_null(self) -> None:
        result = williams_r(OHLC, 5)
        for i in range(4):
            assert result[i] is None


class TestCCI:
    def test_output_length_matches_input(self) -> None:
        assert len(cci(OHLC, 10)) == len(OHLC)

    def test_warm_up_is_null(self) -> None:
        result = cci(OHLC, 5)
        for i in range(4):
            assert result[i] is None


class TestROC:
    def test_output_length_matches_input(self) -> None:
        assert len(roc(CLOSE, 5)) == len(CLOSE)

    def test_warm_up_is_null(self) -> None:
        result = roc(CLOSE, 5)
        for i in range(5):
            assert result[i] is None

    def test_known_value(self) -> None:
        # roc at index 5 = 100 * (CLOSE[5] - CLOSE[0]) / CLOSE[0]
        result = roc(CLOSE, 5)
        expected = 100.0 * (CLOSE[5] - CLOSE[0]) / CLOSE[0]
        assert result[5] == pytest.approx(expected)


# OHLCV fixture shared across MFI and CMF tests.
OHLCV = pl.DataFrame(
    {
        "high": [v + 0.5 for v in CLOSE.to_list()],
        "low": [v - 0.5 for v in CLOSE.to_list()],
        "close": CLOSE.to_list(),
        "volume": [1000.0 + i * 10 for i in range(20)],
    }
)


class TestMFI:
    def test_output_length_matches_input(self) -> None:
        assert len(mfi(OHLCV, 5)) == len(OHLCV)

    def test_warm_up_is_null(self) -> None:
        result = mfi(OHLCV, 5)
        assert result[0] is None

    def test_values_in_range(self) -> None:
        result = mfi(OHLCV, 5)
        for val in result.drop_nulls().to_list():
            assert 0.0 <= val <= 100.0

    def test_alias_contains_period(self) -> None:
        assert mfi(OHLCV, 5).name == "mfi_5"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            mfi(OHLCV, 1)


class TestCMF:
    def test_output_length_matches_input(self) -> None:
        assert len(cmf(OHLCV, 5)) == len(OHLCV)

    def test_warm_up_is_null(self) -> None:
        result = cmf(OHLCV, 5)
        assert result[0] is None

    def test_values_in_valid_range(self) -> None:
        result = cmf(OHLCV, 5)
        for val in result.drop_nulls().to_list():
            assert -1.0 <= val <= 1.0

    def test_alias_contains_period(self) -> None:
        assert cmf(OHLCV, 5).name == "cmf_5"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            cmf(OHLCV, 0)


class TestTSI:
    def test_output_length_matches_input(self) -> None:
        assert len(tsi(CLOSE, 5, 3)) == len(CLOSE)

    def test_warm_up_is_null(self) -> None:
        result = tsi(CLOSE, 5, 3)
        assert result[0] is None

    def test_values_in_range(self) -> None:
        result = tsi(CLOSE, 5, 3)
        for val in result.drop_nulls().to_list():
            assert -100.0 <= val <= 100.0

    def test_alias_contains_periods(self) -> None:
        assert tsi(CLOSE, 5, 3).name == "tsi_5_3"

    def test_invalid_slow_raises(self) -> None:
        with pytest.raises(ValueError):
            tsi(CLOSE, 0, 3)


class TestUltimateOscillator:
    def test_output_length_matches_input(self) -> None:
        assert len(ultimate_oscillator(OHLC, 3, 7, 14)) == len(OHLC)

    def test_warm_up_is_null(self) -> None:
        result = ultimate_oscillator(OHLC, 3, 7, 14)
        assert result[0] is None

    def test_values_in_range(self) -> None:
        result = ultimate_oscillator(OHLC, 3, 7, 14)
        for val in result.drop_nulls().to_list():
            assert 0.0 <= val <= 100.0

    def test_alias_contains_periods(self) -> None:
        assert ultimate_oscillator(OHLC, 3, 7, 14).name == "uo_3_7_14"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            ultimate_oscillator(OHLC, 0, 7, 14)
