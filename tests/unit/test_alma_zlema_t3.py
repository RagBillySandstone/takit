"""Unit tests for ALMA, ZLEMA, and T3 in polarticks.moving_averages."""

from __future__ import annotations

import polars as pl
import pytest

from polarticks.moving_averages import alma, t3, zlema

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
# ZLEMA
# ---------------------------------------------------------------------------


class TestZLEMA:
    """Tests for Zero Lag EMA."""

    def test_output_length_matches_input(self) -> None:
        """ZLEMA output length must equal the input length."""
        assert len(zlema(CLOSE, period=5)) == len(CLOSE)

    def test_leading_nulls_count(self) -> None:
        """First lag + period - 1 values must be null."""
        period = 5
        lag = (period - 1) // 2
        expected_nulls = lag + period - 1
        result = zlema(CLOSE, period=period)
        for idx in range(expected_nulls):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[expected_nulls] is not None

    def test_alias_includes_period(self) -> None:
        """Output series name must embed the period."""
        assert zlema(CLOSE, period=5).name == "zlema_5"

    def test_period_1_no_lag(self) -> None:
        """With period=1 lag=0 so first value is non-null."""
        result = zlema(CLOSE, period=1)
        assert result[0] is not None

    def test_rising_series_tracks_above_ema(self) -> None:
        """On a steadily rising series ZLEMA should be at or above the EMA."""
        rising = pl.Series("close", [float(i) for i in range(1, 41)])
        from polarticks.moving_averages import ema

        z = zlema(rising, period=5)
        e = ema(rising, period=5)
        for zv, ev in zip(z.to_list(), e.to_list(), strict=True):
            if zv is not None and ev is not None:
                assert zv >= ev - 1e-9, f"ZLEMA {zv} unexpectedly below EMA {ev}"

    def test_invalid_period_raises(self) -> None:
        """period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            zlema(CLOSE, period=0)


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------


class TestT3:
    """Tests for the Tillson T3 Moving Average."""

    def test_output_length_matches_input(self) -> None:
        """T3 output length must equal the input length."""
        assert len(t3(CLOSE, period=3)) == len(CLOSE)

    def test_leading_nulls_count(self) -> None:
        """First 6*(period-1) values must be null."""
        period = 3
        expected_nulls = 6 * (period - 1)
        result = t3(CLOSE, period=period)
        for idx in range(expected_nulls):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[expected_nulls] is not None

    def test_alias_includes_period(self) -> None:
        """Output series name must embed the period."""
        assert t3(CLOSE, period=5).name == "t3_5"

    def test_vfactor_zero_equals_triple_ema(self) -> None:
        """With vfactor=0 T3 should equal the plain triple EMA."""
        from polarticks.moving_averages import ema

        period = 3
        result_t3 = t3(CLOSE, period=period, vfactor=0.0)
        e1 = ema(CLOSE, period)
        e2 = ema(e1, period)
        e3 = ema(e2, period)
        for tv, e3v in zip(result_t3.to_list(), e3.to_list(), strict=True):
            if tv is not None and e3v is not None:
                assert tv == pytest.approx(e3v, rel=1e-9)

    def test_period_1_yields_no_nulls(self) -> None:
        """With period=1 there is no warm-up so the first value is non-null."""
        result = t3(CLOSE, period=1)
        assert result[0] is not None

    def test_invalid_period_raises(self) -> None:
        """period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            t3(CLOSE, period=0)

    def test_rising_series_positive(self) -> None:
        """On a steadily rising series T3 values should be positive."""
        rising = pl.Series("close", [float(i + 1) for i in range(40)])
        result = t3(rising, period=2)
        valid = [v for v in result.to_list() if v is not None]
        assert all(v > 0.0 for v in valid)


# ---------------------------------------------------------------------------
# ALMA
# ---------------------------------------------------------------------------


class TestALMA:
    """Tests for the Arnaud Legoux Moving Average."""

    def test_output_length_matches_input(self) -> None:
        """ALMA output length must equal the input length."""
        assert len(alma(CLOSE, period=9)) == len(CLOSE)

    def test_leading_nulls_count(self) -> None:
        """First period-1 values must be null."""
        period = 9
        result = alma(CLOSE, period=period)
        for idx in range(period - 1):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[period - 1] is not None

    def test_alias_includes_period(self) -> None:
        """Output series name must embed the period."""
        assert alma(CLOSE, period=9).name == "alma_9"

    def test_flat_series_returns_constant(self) -> None:
        """ALMA of a constant series should return that constant."""
        flat = pl.Series("close", [50.0] * 30)
        result = alma(flat, period=5)
        valid = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(50.0) for v in valid)

    def test_period_1_no_nulls(self) -> None:
        """With period=1 the first value is non-null."""
        result = alma(CLOSE, period=1)
        assert result[0] is not None

    def test_invalid_period_raises(self) -> None:
        """period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            alma(CLOSE, period=0)

    def test_offset_1_large_sigma_equals_current_price(self) -> None:
        """With offset=1.0 and large sigma the Gaussian is very narrow at the
        current bar (k = period-1), so ALMA ≈ the current price."""
        # sigma=100 → s = period/sigma = 0.09 → very tight bell at k=period-1
        result = alma(CLOSE, period=5, offset=1.0, sigma=100.0)
        valid_pairs = [
            (rv, cv)
            for rv, cv in zip(result.to_list(), CLOSE.to_list(), strict=True)
            if rv is not None
        ]
        for rv, cv in valid_pairs:
            assert rv == pytest.approx(cv, rel=1e-3)
