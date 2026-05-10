"""Unit tests for KAMA and TRIX in polarticks.moving_averages."""

from __future__ import annotations

import polars as pl
import pytest

from polarticks.moving_averages import kama, trix

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# 40-bar synthetic close series with a mix of trend and chop.
CLOSE = pl.Series(
    "close",
    [
        100.0,
        101.0,
        102.5,
        101.0,
        101.5,
        102.0,
        103.0,
        104.0,
        103.5,
        104.5,
        105.0,
        106.0,
        105.5,
        106.5,
        107.0,
        108.0,
        107.5,
        108.5,
        109.0,
        110.0,
        109.5,
        110.5,
        111.0,
        112.0,
        111.5,
        112.5,
        113.0,
        114.0,
        113.5,
        114.5,
        115.0,
        116.0,
        115.5,
        116.5,
        117.0,
        118.0,
        117.5,
        118.5,
        119.0,
        120.0,
    ],
)


# ---------------------------------------------------------------------------
# KAMA
# ---------------------------------------------------------------------------


class TestKAMA:
    """Tests for the Kaufman Adaptive Moving Average."""

    def test_output_length_matches_input(self) -> None:
        """KAMA output length must equal the input length."""
        result = kama(CLOSE, period=10)
        assert len(result) == len(CLOSE)

    def test_leading_nulls_count(self) -> None:
        """First period-1 values must be null; bar period-1 is the seed."""
        period = 10
        result = kama(CLOSE, period=period)
        for idx in range(period - 1):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[period - 1] is not None

    def test_seed_equals_raw_price(self) -> None:
        """The seeded value at index period-1 equals the raw closing price."""
        period = 10
        result = kama(CLOSE, period=period)
        assert result[period - 1] == pytest.approx(CLOSE[period - 1])

    def test_values_stay_near_price_on_trend(self) -> None:
        """On a steadily rising series KAMA should not deviate far from price."""
        rising = pl.Series("close", [float(i) for i in range(1, 41)])
        result = kama(rising, period=5)
        valid_kama = [v for v in result.to_list() if v is not None]
        valid_price = rising.to_list()[4:]
        for kv, pv in zip(valid_kama, valid_price, strict=True):
            assert abs(kv - pv) < 2.0, f"KAMA {kv} drifted far from price {pv}"

    def test_flat_series_kama_stays_constant(self) -> None:
        """On a perfectly flat series KAMA should not drift from the seed."""
        flat = pl.Series("close", [50.0] * 30)
        result = kama(flat, period=5)
        valid = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(50.0) for v in valid)

    def test_period_1_valid(self) -> None:
        """period=1 should produce no leading nulls (seed at bar 0)."""
        result = kama(CLOSE, period=1)
        assert result[0] is not None

    def test_fast_period_ge_slow_raises(self) -> None:
        """fast_period >= slow_period should raise ValueError."""
        with pytest.raises(ValueError, match="fast_period"):
            kama(CLOSE, period=10, fast_period=30, slow_period=10)

    def test_invalid_period_raises(self) -> None:
        """period < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            kama(CLOSE, period=0)

    def test_alias_includes_period(self) -> None:
        """Output series name should embed the period."""
        result = kama(CLOSE, period=10)
        assert result.name == "kama_10"


# ---------------------------------------------------------------------------
# TRIX
# ---------------------------------------------------------------------------


class TestTRIX:
    """Tests for the triple-smoothed EMA oscillator."""

    def test_output_columns(self) -> None:
        """Result DataFrame must contain the three expected columns."""
        result = trix(CLOSE, period=5, signal=3)
        assert set(result.columns) == {"trix_line", "trix_signal", "trix_histogram"}

    def test_output_length_matches_input(self) -> None:
        """All output columns must be as long as the input series."""
        result = trix(CLOSE, period=5, signal=3)
        assert len(result) == len(CLOSE)

    def test_trix_line_leading_nulls(self) -> None:
        """trix_line should have 3*(period-1)+1 leading nulls."""
        period = 5
        expected_nulls = 3 * (period - 1) + 1  # = 13
        result = trix(CLOSE, period=period, signal=3)
        trix_line = result["trix_line"]
        for idx in range(expected_nulls):
            assert trix_line[idx] is None, f"Expected null at index {idx}"
        assert trix_line[expected_nulls] is not None

    def test_histogram_equals_line_minus_signal(self) -> None:
        """Histogram must equal trix_line - trix_signal at every valid bar."""
        result = trix(CLOSE, period=5, signal=3)
        trix_line = result["trix_line"]
        trix_signal = result["trix_signal"]
        histogram = result["trix_histogram"]
        for line_val, sig_val, hist_val in zip(
            trix_line.to_list(), trix_signal.to_list(), histogram.to_list(), strict=True
        ):
            if line_val is not None and sig_val is not None:
                assert hist_val == pytest.approx(line_val - sig_val, abs=1e-10)

    def test_rising_series_trix_positive(self) -> None:
        """On a steadily rising series the TRIX line should be positive."""
        rising = pl.Series("close", [float(i) for i in range(1, 41)])
        result = trix(rising, period=3, signal=3)
        valid = [v for v in result["trix_line"].to_list() if v is not None]
        assert all(v > 0.0 for v in valid)

    def test_invalid_period_raises(self) -> None:
        """period < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            trix(CLOSE, period=0)

    def test_invalid_signal_raises(self) -> None:
        """signal < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            trix(CLOSE, period=5, signal=0)
