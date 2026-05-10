"""Unit tests for aroon and vortex in polarticks.trend."""

from __future__ import annotations

import polars as pl
import pytest

from polarticks.trend import aroon, vortex

# Synthetic OHLC — 50 bars with clear trend structure.
_N = 50
_closes = [100.0 + i * 0.4 + (i % 7) * 0.2 for i in range(_N)]
OHLC = pl.DataFrame(
    {
        "open": [c - 0.2 for c in _closes],
        "high": [c + 1.0 + (i % 3) * 0.3 for i, c in enumerate(_closes)],
        "low": [c - 1.0 - (i % 3) * 0.3 for i, c in enumerate(_closes)],
        "close": _closes,
    }
)


# ---------------------------------------------------------------------------
# Aroon
# ---------------------------------------------------------------------------


class TestAroon:
    """Tests for the Aroon Up/Down/Oscillator indicator."""

    def test_output_columns(self) -> None:
        """Result DataFrame must contain all three aroon columns."""
        result = aroon(OHLC, period=14)
        assert "aroon_up_14" in result.columns
        assert "aroon_down_14" in result.columns
        assert "aroon_osc_14" in result.columns

    def test_output_length_matches_input(self) -> None:
        """All columns must have the same length as the input."""
        result = aroon(OHLC, period=14)
        assert len(result) == len(OHLC)

    def test_leading_nulls_count(self) -> None:
        """First period bars must be null (window_size = period + 1)."""
        period = 10
        result = aroon(OHLC, period=period)
        for col in result.columns:
            for idx in range(period):
                assert result[col][idx] is None, f"Expected null at index {idx} in {col}"
        assert result[f"aroon_up_{period}"][period] is not None

    def test_up_and_down_in_range(self) -> None:
        """Aroon Up and Down must both be in [0, 100]."""
        result = aroon(OHLC, period=14)
        for col in ["aroon_up_14", "aroon_down_14"]:
            valid = [v for v in result[col].to_list() if v is not None]
            assert all(0.0 <= v <= 100.0 for v in valid), f"{col} outside [0, 100]"

    def test_oscillator_in_range(self) -> None:
        """Aroon oscillator must be in [−100, 100]."""
        result = aroon(OHLC, period=14)
        valid = [v for v in result["aroon_osc_14"].to_list() if v is not None]
        assert all(-100.0 <= v <= 100.0 for v in valid)

    def test_oscillator_equals_up_minus_down(self) -> None:
        """aroon_osc must equal aroon_up − aroon_down at every bar."""
        period = 14
        result = aroon(OHLC, period=period)
        up = result[f"aroon_up_{period}"].to_list()
        down = result[f"aroon_down_{period}"].to_list()
        osc = result[f"aroon_osc_{period}"].to_list()
        for u, d, o in zip(up, down, osc, strict=True):
            if u is not None and d is not None:
                assert o == pytest.approx(u - d, abs=1e-10)

    def test_sustained_high_gives_aroon_up_100(self) -> None:
        """If the highest high is always the current bar, aroon_up should be 100."""
        # Monotonically rising highs ensure each new bar sets the period high.
        rising_highs = pl.Series([float(i) for i in range(1, 51)])
        df = pl.DataFrame(
            {
                "open": [1.0] * 50,
                "high": rising_highs.to_list(),
                "low": [0.5] * 50,
                "close": [1.0] * 50,
            }
        )
        result = aroon(df, period=10)
        valid_up = [v for v in result["aroon_up_10"].to_list() if v is not None]
        assert all(v == pytest.approx(100.0) for v in valid_up)

    def test_invalid_period_raises(self) -> None:
        """period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            aroon(OHLC, period=0)


# ---------------------------------------------------------------------------
# Vortex Indicator
# ---------------------------------------------------------------------------


class TestVortex:
    """Tests for the Vortex Indicator (VI+ and VI−)."""

    def test_output_columns(self) -> None:
        """Result DataFrame must contain vi_plus and vi_minus columns."""
        result = vortex(OHLC, period=14)
        assert "vi_plus_14" in result.columns
        assert "vi_minus_14" in result.columns

    def test_output_length_matches_input(self) -> None:
        """Both columns must have the same length as the input."""
        result = vortex(OHLC, period=14)
        assert len(result) == len(OHLC)

    def test_leading_nulls_count(self) -> None:
        """First period-1 bars must be null."""
        period = 10
        result = vortex(OHLC, period=period)
        for col in result.columns:
            for idx in range(period - 1):
                assert result[col][idx] is None, f"Expected null at index {idx} in {col}"
            assert result[col][period - 1] is not None

    def test_values_positive(self) -> None:
        """Both VI+ and VI- must be strictly positive where valid."""
        result = vortex(OHLC, period=14)
        for col in result.columns:
            valid = [v for v in result[col].to_list() if v is not None]
            assert all(v > 0.0 for v in valid), f"{col} has non-positive value"

    def test_uptrend_vi_plus_dominant(self) -> None:
        """In a strong uptrend VI+ should generally exceed VI−."""
        # Build a series of strongly rising bars to produce a clear VI+ > VI- signal.
        n = 40
        highs = [100.0 + i * 2.0 for i in range(n)]
        lows = [100.0 + i * 2.0 - 0.5 for i in range(n)]
        closes = [100.0 + i * 2.0 - 0.1 for i in range(n)]
        df = pl.DataFrame({"open": lows, "high": highs, "low": lows, "close": closes})
        result = vortex(df, period=10)
        valid_plus = [v for v in result["vi_plus_10"].to_list() if v is not None]
        valid_minus = [v for v in result["vi_minus_10"].to_list() if v is not None]
        # On a steadily rising series VI+ should consistently dominate.
        dominant = sum(p > m for p, m in zip(valid_plus, valid_minus, strict=True))
        assert dominant >= len(valid_plus) * 0.8, "Expected VI+ > VI− in most uptrend bars"

    def test_invalid_period_raises(self) -> None:
        """period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            vortex(OHLC, period=0)
