"""Unit tests for CMO, DPO, KST, and Coppock in polarticks.momentum."""

from __future__ import annotations

import polars as pl
import pytest

from polarticks.momentum import cmo, coppock, dpo, kst

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

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
# CMO
# ---------------------------------------------------------------------------


class TestCMO:
    """Tests for the Chande Momentum Oscillator."""

    def test_output_length_matches_input(self) -> None:
        """CMO output must have the same length as the input."""
        assert len(cmo(CLOSE, period=5)) == len(CLOSE)

    def test_leading_nulls_count(self) -> None:
        """First period values must be null (diff adds 1; rolling sum adds period-1)."""
        period = 5
        result = cmo(CLOSE, period=period)
        for idx in range(period):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[period] is not None

    def test_alias_includes_period(self) -> None:
        """Output series name must embed the period."""
        assert cmo(CLOSE, period=5).name == "cmo_5"

    def test_all_up_series_returns_100(self) -> None:
        """A monotonically rising series must produce CMO = +100."""
        rising = pl.Series("close", [float(i) for i in range(1, 30)])
        result = cmo(rising, period=5)
        valid = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(100.0, abs=1e-9) for v in valid)

    def test_all_down_series_returns_minus_100(self) -> None:
        """A monotonically falling series must produce CMO = −100."""
        falling = pl.Series("close", [float(30 - i) for i in range(30)])
        result = cmo(falling, period=5)
        valid = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(-100.0, abs=1e-9) for v in valid)

    def test_invalid_period_raises(self) -> None:
        """period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            cmo(CLOSE, period=0)

    def test_values_bounded(self) -> None:
        """CMO values must lie within [−100, +100]."""
        result = cmo(CLOSE, period=7)
        valid = [v for v in result.to_list() if v is not None]
        assert all(-100.0 <= v <= 100.0 for v in valid)


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


class TestDPO:
    """Tests for the Detrended Price Oscillator."""

    def test_output_length_matches_input(self) -> None:
        """DPO output must have the same length as the input."""
        assert len(dpo(CLOSE, period=10)) == len(CLOSE)

    def test_leading_nulls_count(self) -> None:
        """First (period-1) + (period//2+1) values must be null."""
        period = 10
        displacement = period // 2 + 1
        expected_nulls = (period - 1) + displacement
        result = dpo(CLOSE, period=period)
        for idx in range(expected_nulls):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[expected_nulls] is not None

    def test_alias_includes_period(self) -> None:
        """Output series name must embed the period."""
        assert dpo(CLOSE, period=10).name == "dpo_10"

    def test_flat_series_near_zero(self) -> None:
        """DPO of a flat series should be near zero (same price minus same SMA)."""
        flat = pl.Series("close", [50.0] * 40)
        result = dpo(flat, period=10)
        valid = [v for v in result.to_list() if v is not None]
        assert all(abs(v) < 1e-9 for v in valid)

    def test_invalid_period_raises(self) -> None:
        """period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            dpo(CLOSE, period=0)


# ---------------------------------------------------------------------------
# KST
# ---------------------------------------------------------------------------


class TestKST:
    """Tests for the Know Sure Thing indicator."""

    def test_output_columns(self) -> None:
        """KST result must contain kst_line and kst_signal."""
        result = kst(
            CLOSE, roc1=3, roc2=4, roc3=5, roc4=6, sma1=3, sma2=4, sma3=5, sma4=6, signal=3
        )
        assert set(result.columns) == {"kst_line", "kst_signal"}

    def test_output_length_matches_input(self) -> None:
        """All output columns must be as long as the input."""
        result = kst(
            CLOSE, roc1=3, roc2=4, roc3=5, roc4=6, sma1=3, sma2=4, sma3=5, sma4=6, signal=3
        )
        assert len(result) == len(CLOSE)

    def test_kst_line_leading_nulls(self) -> None:
        """kst_line null count must be roc4 + sma4 - 1."""
        r4, s4 = 6, 6
        expected_nulls = r4 + s4 - 1
        result = kst(
            CLOSE, roc1=3, roc2=4, roc3=5, roc4=r4, sma1=3, sma2=4, sma3=5, sma4=s4, signal=3
        )
        kst_line = result["kst_line"]
        for idx in range(expected_nulls):
            assert kst_line[idx] is None, f"Expected null at {idx}"
        assert kst_line[expected_nulls] is not None

    def test_kst_signal_more_nulls_than_line(self) -> None:
        """kst_signal must have more leading nulls than kst_line."""
        result = kst(
            CLOSE, roc1=3, roc2=4, roc3=5, roc4=6, sma1=3, sma2=4, sma3=5, sma4=6, signal=3
        )
        kst_line_nulls = sum(1 for v in result["kst_line"].to_list() if v is None)
        kst_signal_nulls = sum(1 for v in result["kst_signal"].to_list() if v is None)
        assert kst_signal_nulls > kst_line_nulls


# ---------------------------------------------------------------------------
# Coppock Curve
# ---------------------------------------------------------------------------


class TestCoppock:
    """Tests for the Coppock Curve."""

    def test_output_length_matches_input(self) -> None:
        """Coppock output must have the same length as the input."""
        assert len(coppock(CLOSE, long_roc=5, short_roc=4, wma_period=3)) == len(CLOSE)

    def test_leading_nulls_count(self) -> None:
        """First long_roc + wma_period - 1 values must be null."""
        long_roc, wma_period = 5, 3
        expected_nulls = long_roc + wma_period - 1
        result = coppock(CLOSE, long_roc=long_roc, short_roc=4, wma_period=wma_period)
        for idx in range(expected_nulls):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[expected_nulls] is not None

    def test_alias_is_coppock(self) -> None:
        """Output series name must be 'coppock'."""
        assert coppock(CLOSE, long_roc=5, short_roc=4, wma_period=3).name == "coppock"

    def test_invalid_period_raises(self) -> None:
        """long_roc < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            coppock(CLOSE, long_roc=0)
