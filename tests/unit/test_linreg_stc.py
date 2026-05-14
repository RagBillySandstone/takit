"""Unit tests for linreg_slope and stc in polarticks.trend."""

from __future__ import annotations

import math

import polars as pl
import pytest

from polarticks.trend import linreg_slope, stc

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_N = 60
_closes = [100.0 + math.sin(i * 0.3) * 10 + i * 0.5 for i in range(_N)]
_highs = [c + 1.5 for c in _closes]
_lows = [c - 1.5 for c in _closes]
_DF = pl.DataFrame({"high": _highs, "low": _lows, "close": _closes})

CLOSE = pl.Series("close", _closes)


# ---------------------------------------------------------------------------
# Linear Regression Slope
# ---------------------------------------------------------------------------


class TestLinregSlope:
    """Tests for the rolling linear regression slope."""

    def test_output_length_matches_input(self) -> None:
        """linreg_slope output must have the same length as the input."""
        assert len(linreg_slope(CLOSE, period=5)) == len(CLOSE)

    def test_leading_nulls_count(self) -> None:
        """First period-1 values must be null."""
        period = 5
        result = linreg_slope(CLOSE, period=period)
        for idx in range(period - 1):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[period - 1] is not None

    def test_alias_includes_period(self) -> None:
        """Output series name must embed the period."""
        assert linreg_slope(CLOSE, period=5).name == "linreg_slope_5"

    def test_linear_series_returns_exact_slope(self) -> None:
        """For price = a + b*t the slope should be exactly b."""
        a, b = 100.0, 2.5
        linear = pl.Series("close", [a + b * t for t in range(40)])
        result = linreg_slope(linear, period=5)
        valid = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(b, rel=1e-9) for v in valid)

    def test_flat_series_returns_zero(self) -> None:
        """For a constant price series the slope must be zero."""
        flat = pl.Series("close", [50.0] * 30)
        result = linreg_slope(flat, period=5)
        valid = [v for v in result.to_list() if v is not None]
        assert all(abs(v) < 1e-9 for v in valid)

    def test_period_2_equals_price_diff(self) -> None:
        """With period=2 the slope should equal price[t] - price[t-1]."""
        result = linreg_slope(CLOSE, period=2)
        diff = CLOSE.diff(1)
        for sv, dv in zip(result.to_list(), diff.to_list(), strict=True):
            if sv is not None and dv is not None:
                assert sv == pytest.approx(dv, rel=1e-9)

    def test_period_1_raises(self) -> None:
        """period < 2 must raise ValueError."""
        with pytest.raises(ValueError):
            linreg_slope(CLOSE, period=1)


# ---------------------------------------------------------------------------
# Schaff Trend Cycle
# ---------------------------------------------------------------------------


class TestSTC:
    """Tests for the Schaff Trend Cycle."""

    def test_output_length_matches_input(self) -> None:
        """STC output must have the same length as the input."""
        assert len(stc(_DF, fast=5, slow=10, stoch_period=5, smooth=2)) == _N

    def test_leading_nulls_count(self) -> None:
        """Null prefix must be slow + 2*stoch + 2*smooth - 5."""
        fast, slow, stoch_period, smooth = 5, 10, 5, 2
        expected_nulls = slow + 2 * stoch_period + 2 * smooth - 5
        result = stc(_DF, fast=fast, slow=slow, stoch_period=stoch_period, smooth=smooth)
        for idx in range(expected_nulls):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[expected_nulls] is not None

    def test_alias_is_stc(self) -> None:
        """Output series name must be 'stc'."""
        assert stc(_DF, fast=5, slow=10, stoch_period=5, smooth=2).name == "stc"

    def test_values_clamped_to_0_100(self) -> None:
        """All valid STC values must lie within [0, 100]."""
        result = stc(_DF, fast=5, slow=10, stoch_period=5, smooth=2)
        valid = [v for v in result.to_list() if v is not None]
        assert all(0.0 <= v <= 100.0 for v in valid)

    def test_fast_ge_slow_raises(self) -> None:
        """fast >= slow must raise ValueError."""
        with pytest.raises(ValueError, match="fast"):
            stc(_DF, fast=10, slow=10)

    def test_invalid_period_raises(self) -> None:
        """Any period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            stc(_DF, fast=0, slow=10)
