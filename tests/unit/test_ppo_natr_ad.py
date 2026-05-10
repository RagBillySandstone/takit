"""Unit tests for PPO, NATR, and A/D Line in polarticks."""

from __future__ import annotations

import polars as pl
import pytest

from polarticks.momentum import ppo
from polarticks.volatility import natr
from polarticks.volume import ad_line

# Shared synthetic OHLCV fixture — 40 bars.
_N = 40
_closes = [100.0 + i * 0.5 + (i % 5) * 0.2 for i in range(_N)]
OHLCV = pl.DataFrame(
    {
        "open": [c - 0.3 for c in _closes],
        "high": [c + 1.0 for c in _closes],
        "low": [c - 1.0 for c in _closes],
        "close": _closes,
        "volume": [1000.0 + (i % 10) * 100.0 for i in range(_N)],
    }
)
CLOSE = OHLCV["close"]


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


class TestPPO:
    """Tests for the Percentage Price Oscillator."""

    def test_output_columns(self) -> None:
        """Result must contain ppo_line, ppo_signal, ppo_histogram."""
        result = ppo(CLOSE)
        assert set(result.columns) == {"ppo_line", "ppo_signal", "ppo_histogram"}

    def test_output_length_matches_input(self) -> None:
        """All columns must match the input length."""
        result = ppo(CLOSE, fast=5, slow=10, signal=3)
        assert len(result) == len(CLOSE)

    def test_ppo_line_leading_nulls(self) -> None:
        """ppo_line must have slow-1 leading nulls."""
        slow = 10
        result = ppo(CLOSE, fast=5, slow=slow, signal=3)
        col = result["ppo_line"]
        for idx in range(slow - 1):
            assert col[idx] is None, f"Expected null at index {idx}"
        assert col[slow - 1] is not None

    def test_signal_leading_nulls(self) -> None:
        """ppo_signal has (slow-1)+(signal-1) leading nulls."""
        slow, signal = 10, 3
        expected = (slow - 1) + (signal - 1)
        result = ppo(CLOSE, fast=5, slow=slow, signal=signal)
        col = result["ppo_signal"]
        for idx in range(expected):
            assert col[idx] is None
        assert col[expected] is not None

    def test_histogram_equals_line_minus_signal(self) -> None:
        """Histogram must equal ppo_line - ppo_signal at every valid bar."""
        result = ppo(CLOSE, fast=5, slow=10, signal=3)
        for line, sig, hist in zip(
            result["ppo_line"].to_list(),
            result["ppo_signal"].to_list(),
            result["ppo_histogram"].to_list(),
            strict=True,
        ):
            if line is not None and sig is not None:
                assert hist == pytest.approx(line - sig, abs=1e-10)

    def test_rising_series_ppo_positive(self) -> None:
        """On a steadily rising series the PPO line should be positive."""
        rising = pl.Series("close", [float(i) for i in range(1, 41)])
        result = ppo(rising, fast=3, slow=8, signal=3)
        valid = [v for v in result["ppo_line"].to_list() if v is not None]
        assert all(v > 0.0 for v in valid)

    def test_fast_ge_slow_raises(self) -> None:
        """fast >= slow must raise ValueError."""
        with pytest.raises(ValueError, match="fast"):
            ppo(CLOSE, fast=26, slow=12)


# ---------------------------------------------------------------------------
# NATR
# ---------------------------------------------------------------------------


class TestNATR:
    """Tests for Normalised Average True Range."""

    def test_output_length_matches_input(self) -> None:
        """NATR output must match the input length."""
        result = natr(OHLCV, period=14)
        assert len(result) == len(OHLCV)

    def test_leading_nulls_count(self) -> None:
        """First period-1 bars must be null (inherited from ATR)."""
        period = 10
        result = natr(OHLCV, period=period)
        for idx in range(period - 1):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[period - 1] is not None

    def test_values_positive(self) -> None:
        """Valid NATR values must be strictly positive."""
        result = natr(OHLCV, period=10)
        valid = [v for v in result.to_list() if v is not None]
        assert all(v > 0.0 for v in valid)

    def test_proportional_to_atr(self) -> None:
        """NATR should equal 100 * ATR / close at each valid bar."""
        from polarticks.volatility import atr

        period = 10
        atr_vals = atr(OHLCV, period)
        natr_vals = natr(OHLCV, period)
        close = OHLCV["close"]
        for a, n, c in zip(atr_vals.to_list(), natr_vals.to_list(), close.to_list(), strict=True):
            if a is not None and n is not None:
                assert n == pytest.approx(100.0 * a / c, rel=1e-9)

    def test_alias_includes_period(self) -> None:
        """Output series name should embed the period."""
        result = natr(OHLCV, period=14)
        assert result.name == "natr_14"

    def test_invalid_period_raises(self) -> None:
        """period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            natr(OHLCV, period=0)


# ---------------------------------------------------------------------------
# A/D Line
# ---------------------------------------------------------------------------


class TestADLine:
    """Tests for the Accumulation/Distribution Line."""

    def test_output_length_matches_input(self) -> None:
        """A/D Line output must match the input length."""
        result = ad_line(OHLCV)
        assert len(result) == len(OHLCV)

    def test_no_leading_nulls(self) -> None:
        """A/D Line starts accumulating from bar 0 — no leading nulls."""
        result = ad_line(OHLCV)
        assert result[0] is not None

    def test_alias(self) -> None:
        """Output series name must be 'ad_line'."""
        assert ad_line(OHLCV).name == "ad_line"

    def test_bullish_close_increases_ad(self) -> None:
        """A close at the top of the bar range should increase the A/D line."""
        # Each bar: high=110, low=100, close=110 → MFM = 1.0 → full positive volume.
        df = pl.DataFrame(
            {
                "open": [105.0] * 5,
                "high": [110.0] * 5,
                "low": [100.0] * 5,
                "close": [110.0] * 5,
                "volume": [100.0] * 5,
            }
        )
        result = ad_line(df).to_list()
        # Each bar contributes +100 (full positive MFV).
        assert result == pytest.approx([100.0, 200.0, 300.0, 400.0, 500.0])

    def test_bearish_close_decreases_ad(self) -> None:
        """A close at the bottom of the bar range should decrease the A/D line."""
        df = pl.DataFrame(
            {
                "open": [105.0] * 5,
                "high": [110.0] * 5,
                "low": [100.0] * 5,
                "close": [100.0] * 5,
                "volume": [100.0] * 5,
            }
        )
        result = ad_line(df).to_list()
        assert result == pytest.approx([-100.0, -200.0, -300.0, -400.0, -500.0])

    def test_doji_bar_contributes_zero(self) -> None:
        """A zero-range (doji) bar should not change the A/D line."""
        df = pl.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [100.0, 101.0],
                "low": [100.0, 99.0],
                "close": [100.0, 101.0],
                "volume": [500.0, 100.0],
            }
        )
        result = ad_line(df).to_list()
        # Bar 0: doji → MFM=NaN → filled to 0 → contributes 0.
        # Bar 1: close=101, high=101, low=99 → MFM=1 → +100.
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(100.0)
