"""Unit tests for chandelier_exit and ichimoku in polarticks."""

from __future__ import annotations

import polars as pl
import pytest

from polarticks.trend import ichimoku
from polarticks.volatility import chandelier_exit

# Synthetic OHLC — 80 bars so Ichimoku span-B has enough warm-up.
_N = 80
_closes = [100.0 + i * 0.5 + (i % 5) * 0.1 for i in range(_N)]
OHLC = pl.DataFrame(
    {
        "open": [c - 0.3 for c in _closes],
        "high": [c + 1.0 for c in _closes],
        "low": [c - 1.0 for c in _closes],
        "close": _closes,
    }
)


# ---------------------------------------------------------------------------
# Chandelier Exit
# ---------------------------------------------------------------------------


class TestChandelierExit:
    """Tests for the Chandelier Exit trailing-stop indicator."""

    def test_output_columns(self) -> None:
        """Result DataFrame must contain ce_long and ce_short columns."""
        result = chandelier_exit(OHLC, period=22)
        assert "ce_long_22" in result.columns
        assert "ce_short_22" in result.columns

    def test_output_length_matches_input(self) -> None:
        """All output columns must match the input length."""
        result = chandelier_exit(OHLC, period=22)
        assert len(result) == len(OHLC)

    def test_leading_nulls_count(self) -> None:
        """First period-1 bars must be null (ATR warm-up)."""
        period = 10
        result = chandelier_exit(OHLC, period=period)
        for idx in range(period - 1):
            assert result[f"ce_long_{period}"][idx] is None
            assert result[f"ce_short_{period}"][idx] is None
        assert result[f"ce_long_{period}"][period - 1] is not None

    def test_long_exit_below_highest_high(self) -> None:
        """Long exit must be strictly below the highest high in the window."""
        period = 10
        result = chandelier_exit(OHLC, period=period, multiplier=3.0)
        highest = OHLC["high"].rolling_max(window_size=period, min_samples=period)
        long_col = f"ce_long_{period}"
        for hh, ce in zip(highest.to_list(), result[long_col].to_list(), strict=True):
            if hh is not None and ce is not None:
                assert ce < hh, f"Long exit {ce} should be below highest high {hh}"

    def test_short_exit_above_lowest_low(self) -> None:
        """Short exit must be strictly above the lowest low in the window."""
        period = 10
        result = chandelier_exit(OHLC, period=period, multiplier=3.0)
        lowest = OHLC["low"].rolling_min(window_size=period, min_samples=period)
        short_col = f"ce_short_{period}"
        for ll, ce in zip(lowest.to_list(), result[short_col].to_list(), strict=True):
            if ll is not None and ce is not None:
                assert ce > ll, f"Short exit {ce} should be above lowest low {ll}"

    def test_invalid_period_raises(self) -> None:
        """period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            chandelier_exit(OHLC, period=0)


# ---------------------------------------------------------------------------
# Ichimoku Cloud
# ---------------------------------------------------------------------------


class TestIchimoku:
    """Tests for the Ichimoku Cloud indicator."""

    def test_output_columns(self) -> None:
        """Result DataFrame must contain all five Ichimoku components."""
        result = ichimoku(OHLC)
        assert set(result.columns) == {
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
        }

    def test_output_length_matches_input(self) -> None:
        """All five columns must match the input length."""
        result = ichimoku(OHLC)
        assert len(result) == len(OHLC)

    def test_tenkan_sen_leading_nulls(self) -> None:
        """tenkan_sen must have tenkan_period-1 leading nulls."""
        tenkan_p = 5
        result = ichimoku(OHLC, tenkan_period=tenkan_p, kijun_period=10, senkou_b_period=20)
        col = result["tenkan_sen"]
        for idx in range(tenkan_p - 1):
            assert col[idx] is None
        assert col[tenkan_p - 1] is not None

    def test_kijun_sen_leading_nulls(self) -> None:
        """kijun_sen must have kijun_period-1 leading nulls."""
        kijun_p = 10
        result = ichimoku(OHLC, tenkan_period=5, kijun_period=kijun_p, senkou_b_period=20)
        col = result["kijun_sen"]
        for idx in range(kijun_p - 1):
            assert col[idx] is None
        assert col[kijun_p - 1] is not None

    def test_senkou_span_a_leading_nulls(self) -> None:
        """senkou_span_a leading nulls limited by kijun_period (the slower component)."""
        tenkan_p, kijun_p = 5, 10
        result = ichimoku(OHLC, tenkan_period=tenkan_p, kijun_period=kijun_p, senkou_b_period=20)
        col = result["senkou_span_a"]
        for idx in range(kijun_p - 1):
            assert col[idx] is None
        assert col[kijun_p - 1] is not None

    def test_senkou_span_b_leading_nulls(self) -> None:
        """senkou_span_b must have senkou_b_period-1 leading nulls."""
        senkou_b_p = 20
        result = ichimoku(OHLC, tenkan_period=5, kijun_period=10, senkou_b_period=senkou_b_p)
        col = result["senkou_span_b"]
        for idx in range(senkou_b_p - 1):
            assert col[idx] is None
        assert col[senkou_b_p - 1] is not None

    def test_chikou_span_trailing_nulls(self) -> None:
        """chikou_span must have chikou_period trailing nulls and no leading nulls."""
        chikou_p = 5
        result = ichimoku(
            OHLC, tenkan_period=3, kijun_period=5, senkou_b_period=10, chikou_period=chikou_p
        )
        col = result["chikou_span"]
        # First value must not be null (shift(-n) moves values forward, nulls go to tail).
        assert col[0] is not None
        # Last chikou_p values must be null.
        for idx in range(len(col) - chikou_p, len(col)):
            assert col[idx] is None, f"Expected trailing null at index {idx}"

    def test_tenkan_midpoint_formula(self) -> None:
        """tenkan_sen value at first valid bar == (highest_high + lowest_low) / 2."""
        period = 5
        result = ichimoku(OHLC, tenkan_period=period, kijun_period=10, senkou_b_period=20)
        idx = period - 1
        expected = (
            OHLC["high"][:period].max() + OHLC["low"][:period].min()  # type: ignore[operator]
        ) / 2.0
        assert result["tenkan_sen"][idx] == pytest.approx(expected)

    def test_invalid_period_raises(self) -> None:
        """Any period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            ichimoku(OHLC, tenkan_period=0)
