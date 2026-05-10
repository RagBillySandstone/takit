"""Unit tests for StochRSI in polarticks.momentum."""

from __future__ import annotations

import polars as pl
import pytest

from polarticks.momentum import stoch_rsi

# A long enough series to warm up RSI + stochastic window.
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
        45.80,
        46.20,
        46.00,
        45.70,
        45.90,
        46.30,
        46.10,
        45.80,
        46.00,
        46.40,
        46.20,
        45.90,
        46.10,
        46.50,
        46.30,
        46.00,
        46.20,
        46.60,
        46.40,
        46.10,
        46.30,
        46.70,
        46.50,
        46.20,
        46.40,
        46.80,
        46.60,
        46.30,
        46.50,
        46.90,
        46.70,
        46.40,
        46.60,
        47.00,
        46.80,
        46.50,
        46.70,
        47.10,
        46.90,
        46.60,
    ],
)


class TestStochRSI:
    """Tests for the Stochastic RSI oscillator."""

    def test_output_columns(self) -> None:
        """Result DataFrame must contain stoch_rsi_k and stoch_rsi_d."""
        result = stoch_rsi(CLOSE)
        assert set(result.columns) == {"stoch_rsi_k", "stoch_rsi_d"}

    def test_output_length_matches_input(self) -> None:
        """All columns must have the same length as the input series."""
        result = stoch_rsi(CLOSE)
        assert len(result) == len(CLOSE)

    def test_leading_nulls_k(self) -> None:
        """stoch_rsi_k leading nulls = rsi_period + stoch_period + k_period - 2."""
        rsi_p, stoch_p, k_p, d_p = 5, 5, 3, 3
        expected_k_nulls = rsi_p + stoch_p + k_p - 2  # = 11
        result = stoch_rsi(
            CLOSE, rsi_period=rsi_p, stoch_period=stoch_p, k_period=k_p, d_period=d_p
        )
        k = result["stoch_rsi_k"]
        for idx in range(expected_k_nulls):
            assert k[idx] is None, f"Expected null at index {idx}"
        assert k[expected_k_nulls] is not None

    def test_leading_nulls_d(self) -> None:
        """stoch_rsi_d leading nulls = rsi_period + stoch_period + k_period + d_period - 3."""
        rsi_p, stoch_p, k_p, d_p = 5, 5, 3, 3
        expected_d_nulls = rsi_p + stoch_p + k_p + d_p - 3  # = 13
        result = stoch_rsi(
            CLOSE, rsi_period=rsi_p, stoch_period=stoch_p, k_period=k_p, d_period=d_p
        )
        d = result["stoch_rsi_d"]
        for idx in range(expected_d_nulls):
            assert d[idx] is None, f"Expected null at index {idx}"
        assert d[expected_d_nulls] is not None

    def test_k_values_in_range(self) -> None:
        """Valid %K values must be in [0, 100]."""
        result = stoch_rsi(CLOSE)
        valid = [v for v in result["stoch_rsi_k"].to_list() if v is not None]
        assert all(0.0 <= v <= 100.0 for v in valid), "K values outside [0, 100]"

    def test_d_values_in_range(self) -> None:
        """Valid %D values must be in [0, 100]."""
        result = stoch_rsi(CLOSE)
        valid = [v for v in result["stoch_rsi_d"].to_list() if v is not None]
        assert all(0.0 <= v <= 100.0 for v in valid), "D values outside [0, 100]"

    def test_flat_series_returns_neutral(self) -> None:
        """A flat close series produces RSI of NaN/100 or 50 — no crash."""
        flat = pl.Series("close", [50.0] * 50)
        result = stoch_rsi(flat, rsi_period=5, stoch_period=5, k_period=3, d_period=3)
        # Should not raise; valid values must still be in range.
        valid = [v for v in result["stoch_rsi_k"].to_list() if v is not None]
        assert all(0.0 <= v <= 100.0 for v in valid)

    def test_invalid_rsi_period_raises(self) -> None:
        """rsi_period < 2 must raise ValueError."""
        with pytest.raises(ValueError):
            stoch_rsi(CLOSE, rsi_period=1)

    def test_invalid_stoch_period_raises(self) -> None:
        """stoch_period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            stoch_rsi(CLOSE, stoch_period=0)
