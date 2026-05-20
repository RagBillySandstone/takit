"""Unit tests for v0.6.0 indicators.

Covers:
    moving_averages: ehma, pwma
    momentum:        disparity_index, apo, asi, pmo, chande_trend_score
    trend:           ma_envelope, linreg_intercept, standard_error_bands, cog, rwi
    volatility:      coefficient_of_variation, efficiency_ratio, standard_error
    volume:          vzo, mfi_bw, volume_delta
    patterns:        is_marubozu_bullish, is_marubozu_bearish

Each test class verifies:
  - Output length matches input length.
  - Correct number of leading nulls.
  - First valid value is finite (not nan/inf).
  - Sensible domain constraints where applicable.
  - At least one edge case or error condition.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from polarticks.momentum import apo, asi, chande_trend_score, disparity_index, pmo
from polarticks.moving_averages import ehma, pwma
from polarticks.patterns import is_marubozu_bearish, is_marubozu_bullish
from polarticks.trend import cog, linreg_intercept, ma_envelope, rwi, standard_error_bands
from polarticks.volatility import coefficient_of_variation, efficiency_ratio, standard_error
from polarticks.volume import mfi_bw, volume_delta, vzo

# ---------------------------------------------------------------------------
# Shared synthetic OHLCV fixture (N = 120 bars)
# ---------------------------------------------------------------------------

_N = 120

_closes = [100.0 + i * 0.5 + math.sin(i * 0.4) * 2 for i in range(_N)]
_highs = [c + 1.5 + math.cos(i * 0.3) * 0.3 for i, c in enumerate(_closes)]
_lows = [c - 1.5 - math.cos(i * 0.3) * 0.3 for i, c in enumerate(_closes)]
_opens = [c - 0.3 + math.sin(i * 0.5) * 0.2 for i, c in enumerate(_closes)]
_volumes = [1000.0 + math.sin(i * 0.6) * 300 + i * 5 for i in range(_N)]

OHLCV = pl.DataFrame(
    {
        "open": _opens,
        "high": _highs,
        "low": _lows,
        "close": _closes,
        "volume": _volumes,
    }
)
CLOSE = OHLCV["close"]
OHLC = OHLCV.select(["open", "high", "low", "close"])

# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


class TestEHMA:
    """Exponential Hull Moving Average."""

    def test_output_length(self) -> None:
        result = ehma(CLOSE, 10)
        assert len(result) == _N

    def test_leading_nulls(self) -> None:
        period = 10
        sqrt_p = max(2, round(math.sqrt(period)))
        expected_nulls = period + sqrt_p - 2
        result = ehma(CLOSE, period)
        assert result[:expected_nulls].null_count() == expected_nulls
        assert result[expected_nulls] is not None

    def test_first_valid_finite(self) -> None:
        period = 10
        sqrt_p = max(2, round(math.sqrt(period)))
        idx = period + sqrt_p - 2
        val = ehma(CLOSE, period)[idx]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert ehma(CLOSE, 14).name == "ehma_14"

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            ehma(CLOSE, 1)

    def test_period_2(self) -> None:
        result = ehma(CLOSE, 2)
        assert result.null_count() < _N


class TestPWMA:
    """Pascal's Weighted Moving Average."""

    def test_output_length(self) -> None:
        assert len(pwma(CLOSE, 5)) == _N

    def test_leading_nulls(self) -> None:
        period = 5
        result = pwma(CLOSE, period)
        assert result[: period - 1].null_count() == period - 1
        assert result[period - 1] is not None

    def test_first_valid_finite(self) -> None:
        val = pwma(CLOSE, 5)[4]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert pwma(CLOSE, 8).name == "pwma_8"

    def test_period_1_no_nulls(self) -> None:
        result = pwma(CLOSE, 1)
        assert result.null_count() == 0

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            pwma(CLOSE, 0)

    def test_constant_series(self) -> None:
        """PWMA of a constant equals that constant."""
        flat = pl.Series([5.0] * 30)
        result = pwma(flat, 5)
        valid = result.drop_nulls()
        assert all(abs(v - 5.0) < 1e-10 for v in valid.to_list())


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


class TestDisparityIndex:
    """Disparity Index."""

    def test_output_length(self) -> None:
        assert len(disparity_index(CLOSE, 14)) == _N

    def test_leading_nulls(self) -> None:
        period = 14
        result = disparity_index(CLOSE, period)
        assert result[: period - 1].null_count() == period - 1
        assert result[period - 1] is not None

    def test_first_valid_finite(self) -> None:
        val = disparity_index(CLOSE, 14)[13]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert disparity_index(CLOSE, 10).name == "disparity_10"

    def test_constant_series_is_zero(self) -> None:
        """DI of flat price = 0 (price = SMA)."""
        flat = pl.Series([50.0] * 30)
        result = disparity_index(flat, 5).drop_nulls()
        assert all(abs(v) < 1e-10 for v in result.to_list())

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            disparity_index(CLOSE, 0)


class TestAPO:
    """Absolute Price Oscillator."""

    def test_output_length(self) -> None:
        assert len(apo(CLOSE)) == _N

    def test_leading_nulls(self) -> None:
        fast, slow = 5, 10
        result = apo(CLOSE, fast=fast, slow=slow)
        assert result[: slow - 1].null_count() == slow - 1
        assert result[slow - 1] is not None

    def test_first_valid_finite(self) -> None:
        val = apo(CLOSE, fast=5, slow=10)[9]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert apo(CLOSE, fast=5, slow=10).name == "apo_5_10"

    def test_fast_ge_slow_raises(self) -> None:
        with pytest.raises(ValueError):
            apo(CLOSE, fast=10, slow=10)

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            apo(CLOSE, fast=0, slow=10)


class TestASI:
    """Accumulative Swing Index."""

    def test_output_length(self) -> None:
        assert len(asi(OHLC)) == _N

    def test_leading_nulls(self) -> None:
        result = asi(OHLC)
        assert result[0] is None
        assert result[1] is not None

    def test_first_valid_finite(self) -> None:
        val = asi(OHLC)[1]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert asi(OHLC).name == "asi"

    def test_monotone_rising_market(self) -> None:
        """ASI accumulates positive SI for consistently rising closes."""
        closes = list(range(100, 160))
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        opens = [c - 0.5 for c in closes]
        df = pl.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})
        result = asi(df).drop_nulls()
        # ASI should be positive (net upward accumulation).
        assert result[-1] > 0.0


class TestPMO:
    """Price Momentum Oscillator."""

    def test_output_length(self) -> None:
        assert len(pmo(CLOSE)) == _N

    def test_leading_nulls(self) -> None:
        fast, slow = 35, 20
        expected = fast + slow - 1
        result = pmo(CLOSE, fast=fast, slow=slow)
        assert result[:expected].null_count() == expected
        assert result[expected] is not None

    def test_first_valid_finite(self) -> None:
        val = pmo(CLOSE, fast=10, slow=5)[14]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert pmo(CLOSE, fast=10, slow=5).name == "pmo_10_5"

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            pmo(CLOSE, fast=0, slow=5)

    def test_short_periods(self) -> None:
        result = pmo(CLOSE, fast=3, slow=2)
        assert result.drop_nulls().len() > 0


class TestChandeTrendScore:
    """Chande Trend Score."""

    def test_output_length(self) -> None:
        assert len(chande_trend_score(CLOSE, 10)) == _N

    def test_leading_nulls(self) -> None:
        period = 10
        result = chande_trend_score(CLOSE, period)
        assert result[:period].null_count() == period
        assert result[period] is not None

    def test_range_0_to_100(self) -> None:
        result = chande_trend_score(CLOSE, 10).drop_nulls()
        assert result.min() >= 0.0
        assert result.max() <= 100.0

    def test_alias(self) -> None:
        assert chande_trend_score(CLOSE, 10).name == "cts_10"

    def test_rising_market_high_score(self) -> None:
        """Monotonically rising price should yield a CTS of 100 for every bar."""
        rising = pl.Series([float(i) for i in range(50)])
        result = chande_trend_score(rising, 10).drop_nulls()
        assert all(v == pytest.approx(100.0) for v in result.to_list())

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            chande_trend_score(CLOSE, 0)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


class TestMAEnvelope:
    """Moving Average Envelope."""

    def test_output_shape(self) -> None:
        result = ma_envelope(CLOSE, 10)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == _N

    def test_columns_present(self) -> None:
        result = ma_envelope(CLOSE, 10, pct=0.02)
        assert "mae_upper_10" in result.columns
        assert "mae_middle_10" in result.columns
        assert "mae_lower_10" in result.columns

    def test_leading_nulls(self) -> None:
        period = 10
        result = ma_envelope(CLOSE, period)
        mid = result[f"mae_middle_{period}"]
        assert mid[: period - 1].null_count() == period - 1
        assert mid[period - 1] is not None

    def test_upper_gt_middle_gt_lower(self) -> None:
        result = ma_envelope(CLOSE, 10, pct=0.02).drop_nulls()
        assert (result["mae_upper_10"] > result["mae_middle_10"]).all()
        assert (result["mae_middle_10"] > result["mae_lower_10"]).all()

    def test_invalid_pct_raises(self) -> None:
        with pytest.raises(ValueError):
            ma_envelope(CLOSE, 10, pct=0.0)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            ma_envelope(CLOSE, 0)


class TestLinregIntercept:
    """Rolling linear regression intercept."""

    def test_output_length(self) -> None:
        assert len(linreg_intercept(CLOSE, 14)) == _N

    def test_leading_nulls(self) -> None:
        period = 14
        result = linreg_intercept(CLOSE, period)
        assert result[: period - 1].null_count() == period - 1
        assert result[period - 1] is not None

    def test_first_valid_finite(self) -> None:
        val = linreg_intercept(CLOSE, 14)[13]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert linreg_intercept(CLOSE, 5).name == "linreg_intercept_5"

    def test_constant_series(self) -> None:
        """Intercept of a flat series equals that constant (slope = 0)."""
        flat = pl.Series([7.0] * 20)
        result = linreg_intercept(flat, 5).drop_nulls()
        assert all(abs(v - 7.0) < 1e-10 for v in result.to_list())

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            linreg_intercept(CLOSE, 1)


class TestStandardErrorBands:
    """Standard Error Bands."""

    def test_output_shape(self) -> None:
        result = standard_error_bands(CLOSE, 10)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == _N

    def test_columns_present(self) -> None:
        result = standard_error_bands(CLOSE, 10)
        assert "seb_upper_10" in result.columns
        assert "seb_middle_10" in result.columns
        assert "seb_lower_10" in result.columns

    def test_leading_nulls(self) -> None:
        period = 10
        result = standard_error_bands(CLOSE, period)
        mid = result[f"seb_middle_{period}"]
        assert mid[: period - 1].null_count() == period - 1
        assert mid[period - 1] is not None

    def test_upper_gt_lower(self) -> None:
        result = standard_error_bands(CLOSE, 10).drop_nulls()
        assert (result["seb_upper_10"] >= result["seb_lower_10"]).all()

    def test_constant_series_no_spread(self) -> None:
        """Flat series has no residuals → upper = middle = lower."""
        flat = pl.Series([5.0] * 20)
        result = standard_error_bands(flat, 5).drop_nulls()
        diff = (result["seb_upper_5"] - result["seb_lower_5"]).abs()
        assert diff.max() < 1e-10

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            standard_error_bands(CLOSE, 2)


class TestCOG:
    """Centre of Gravity oscillator."""

    def test_output_length(self) -> None:
        assert len(cog(CLOSE, 10)) == _N

    def test_leading_nulls(self) -> None:
        period = 10
        result = cog(CLOSE, period)
        assert result[: period - 1].null_count() == period - 1
        assert result[period - 1] is not None

    def test_first_valid_finite(self) -> None:
        val = cog(CLOSE, 10)[9]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert cog(CLOSE, 8).name == "cog_8"

    def test_values_negative(self) -> None:
        """COG is defined as negative; all valid values should be ≤ 0."""
        result = cog(CLOSE, 10).drop_nulls()
        assert (result <= 0.0).all()

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            cog(CLOSE, 0)


class TestRWI:
    """Random Walk Index."""

    def test_output_shape(self) -> None:
        result = rwi(OHLC, 14)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == _N

    def test_columns_present(self) -> None:
        result = rwi(OHLC, 14)
        assert "rwi_high_14" in result.columns
        assert "rwi_low_14" in result.columns

    def test_leading_nulls(self) -> None:
        period = 14
        result = rwi(OHLC, period)
        hi = result["rwi_high_14"]
        assert hi[:period].null_count() == period
        assert hi[period] is not None

    def test_first_valid_finite(self) -> None:
        result = rwi(OHLC, 14)
        assert math.isfinite(result["rwi_high_14"][14])
        assert math.isfinite(result["rwi_low_14"][14])

    def test_trending_market_exceeds_one(self) -> None:
        """In a strong monotone trend RWI should frequently exceed 1."""
        closes = [float(i) + 100 for i in range(60)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 0.5 for c in closes]
        df = pl.DataFrame({"high": highs, "low": lows, "close": closes})
        result = rwi(df, 10)["rwi_high_10"].drop_nulls()
        assert (result > 1.0).sum() > 0

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            rwi(OHLC, 0)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


class TestCoefficientOfVariation:
    """Coefficient of Variation."""

    def test_output_length(self) -> None:
        assert len(coefficient_of_variation(CLOSE, 20)) == _N

    def test_leading_nulls(self) -> None:
        period = 20
        result = coefficient_of_variation(CLOSE, period)
        assert result[: period - 1].null_count() == period - 1
        assert result[period - 1] is not None

    def test_first_valid_finite(self) -> None:
        val = coefficient_of_variation(CLOSE, 20)[19]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert coefficient_of_variation(CLOSE, 10).name == "cv_10"

    def test_constant_series_is_zero(self) -> None:
        """std of flat series = 0 → CV = 0."""
        flat = pl.Series([42.0] * 30)
        result = coefficient_of_variation(flat, 5).drop_nulls()
        assert all(abs(v) < 1e-10 for v in result.to_list())

    def test_nonnegative(self) -> None:
        result = coefficient_of_variation(CLOSE, 20).drop_nulls()
        assert (result >= 0.0).all()

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            coefficient_of_variation(CLOSE, 1)


class TestEfficiencyRatio:
    """Kaufman Efficiency Ratio."""

    def test_output_length(self) -> None:
        assert len(efficiency_ratio(CLOSE, 14)) == _N

    def test_leading_nulls(self) -> None:
        period = 14
        result = efficiency_ratio(CLOSE, period)
        assert result[:period].null_count() == period
        assert result[period] is not None

    def test_first_valid_finite(self) -> None:
        val = efficiency_ratio(CLOSE, 14)[14]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert efficiency_ratio(CLOSE, 10).name == "er_10"

    def test_range_0_to_1(self) -> None:
        result = efficiency_ratio(CLOSE, 14).drop_nulls()
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-10

    def test_straight_line_is_one(self) -> None:
        """Perfect straight-line price movement → ER = 1."""
        linear = pl.Series([float(i) for i in range(30)])
        result = efficiency_ratio(linear, 5).drop_nulls()
        assert all(abs(v - 1.0) < 1e-10 for v in result.to_list())

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            efficiency_ratio(CLOSE, 0)


class TestStandardError:
    """Rolling OLS residual standard error."""

    def test_output_length(self) -> None:
        assert len(standard_error(CLOSE, 14)) == _N

    def test_leading_nulls(self) -> None:
        period = 14
        result = standard_error(CLOSE, period)
        assert result[: period - 1].null_count() == period - 1
        assert result[period - 1] is not None

    def test_first_valid_finite(self) -> None:
        val = standard_error(CLOSE, 14)[13]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert standard_error(CLOSE, 7).name == "se_7"

    def test_nonnegative(self) -> None:
        result = standard_error(CLOSE, 10).drop_nulls()
        assert (result >= 0.0).all()

    def test_constant_series_is_zero(self) -> None:
        """Flat price has no residuals; SE = 0."""
        flat = pl.Series([5.0] * 20)
        result = standard_error(flat, 5).drop_nulls()
        assert all(abs(v) < 1e-10 for v in result.to_list())

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            standard_error(CLOSE, 2)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class TestVZO:
    """Volume Zone Oscillator."""

    def test_output_length(self) -> None:
        assert len(vzo(OHLCV, 14)) == _N

    def test_leading_nulls(self) -> None:
        period = 14
        result = vzo(OHLCV, period)
        assert result[: period - 1].null_count() == period - 1
        assert result[period - 1] is not None

    def test_first_valid_finite(self) -> None:
        val = vzo(OHLCV, 14)[13]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert vzo(OHLCV, 10).name == "vzo_10"

    def test_range_roughly_bounded(self) -> None:
        """VZO stays within [-100, 100] for normal data."""
        result = vzo(OHLCV, 14).drop_nulls()
        assert result.min() >= -100.0
        assert result.max() <= 100.0

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            vzo(OHLCV, 0)


class TestMFIBW:
    """Market Facilitation Index (Bill Williams)."""

    def test_output_length(self) -> None:
        assert len(mfi_bw(OHLCV)) == _N

    def test_no_leading_nulls(self) -> None:
        result = mfi_bw(OHLCV)
        assert result.null_count() == 0

    def test_nonnegative(self) -> None:
        result = mfi_bw(OHLCV)
        assert (result >= 0.0).all()

    def test_alias(self) -> None:
        assert mfi_bw(OHLCV).name == "mfi_bw"

    def test_zero_volume_yields_nan(self) -> None:
        df = pl.DataFrame(
            {"high": [5.0], "low": [3.0], "close": [4.0], "volume": [0.0], "open": [4.0]}
        )
        result = mfi_bw(df)
        assert math.isnan(result[0])

    def test_wide_range_high_mfi(self) -> None:
        """Wide range and low volume → high MFI."""
        df = pl.DataFrame(
            {"high": [200.0], "low": [100.0], "close": [150.0], "volume": [1.0], "open": [150.0]}
        )
        assert mfi_bw(df)[0] == pytest.approx(100.0)


class TestVolumeDelta:
    """Volume Delta."""

    def test_output_length(self) -> None:
        assert len(volume_delta(OHLCV)) == _N

    def test_no_leading_nulls(self) -> None:
        result = volume_delta(OHLCV)
        assert result.null_count() == 0

    def test_alias(self) -> None:
        assert volume_delta(OHLCV).name == "volume_delta"

    def test_close_at_high_positive(self) -> None:
        """Close at high → full positive delta (CLV = +1)."""
        df = pl.DataFrame(
            {"high": [10.0], "low": [5.0], "close": [10.0], "volume": [1000.0], "open": [7.5]}
        )
        val = volume_delta(df)[0]
        assert val == pytest.approx(1000.0)

    def test_close_at_low_negative(self) -> None:
        """Close at low → full negative delta (CLV = -1)."""
        df = pl.DataFrame(
            {"high": [10.0], "low": [5.0], "close": [5.0], "volume": [1000.0], "open": [7.5]}
        )
        val = volume_delta(df)[0]
        assert val == pytest.approx(-1000.0)

    def test_zero_range_yields_zero(self) -> None:
        """Doji bar (H = L = C) has no directional information → delta = 0."""
        df = pl.DataFrame(
            {"high": [7.0], "low": [7.0], "close": [7.0], "volume": [500.0], "open": [7.0]}
        )
        val = volume_delta(df)[0]
        assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


def _make_marubozu_bullish() -> pl.DataFrame:
    """Bullish Marubozu: open ≈ low, close ≈ high, tiny wicks."""
    return pl.DataFrame({"open": [10.0], "high": [20.02], "low": [9.98], "close": [20.0]})


def _make_marubozu_bearish() -> pl.DataFrame:
    """Bearish Marubozu: open ≈ high, close ≈ low, tiny wicks."""
    return pl.DataFrame({"open": [20.0], "high": [20.02], "low": [9.98], "close": [10.0]})


def _make_large_wick_candle() -> pl.DataFrame:
    """Candle with significant wicks — should not qualify as Marubozu."""
    return pl.DataFrame({"open": [10.0], "high": [25.0], "low": [5.0], "close": [20.0]})


def _make_doji_candle() -> pl.DataFrame:
    """Zero-range candle — should return False (excluded)."""
    return pl.DataFrame({"open": [10.0], "high": [10.0], "low": [10.0], "close": [10.0]})


class TestIsMarubozu:
    """Marubozu pattern tests (bullish and bearish)."""

    # --- bullish ---

    def test_bullish_detects_marubozu(self) -> None:
        df = _make_marubozu_bullish()
        assert is_marubozu_bullish(df)[0] is True

    def test_bullish_rejects_large_wick(self) -> None:
        df = _make_large_wick_candle()
        assert is_marubozu_bullish(df)[0] is False

    def test_bullish_rejects_doji(self) -> None:
        df = _make_doji_candle()
        assert is_marubozu_bullish(df)[0] is False

    def test_bullish_rejects_bearish_candle(self) -> None:
        df = _make_marubozu_bearish()
        assert is_marubozu_bullish(df)[0] is False

    def test_bullish_output_length(self) -> None:
        result = is_marubozu_bullish(OHLC)
        assert len(result) == _N

    def test_bullish_dtype(self) -> None:
        assert is_marubozu_bullish(OHLC).dtype == pl.Boolean

    def test_bullish_alias(self) -> None:
        assert is_marubozu_bullish(OHLC).name == "marubozu_bullish"

    def test_bullish_no_nulls(self) -> None:
        assert is_marubozu_bullish(OHLC).null_count() == 0

    # --- bearish ---

    def test_bearish_detects_marubozu(self) -> None:
        df = _make_marubozu_bearish()
        assert is_marubozu_bearish(df)[0] is True

    def test_bearish_rejects_large_wick(self) -> None:
        df = _make_large_wick_candle()
        assert is_marubozu_bearish(df)[0] is False

    def test_bearish_rejects_doji(self) -> None:
        df = _make_doji_candle()
        assert is_marubozu_bearish(df)[0] is False

    def test_bearish_rejects_bullish_candle(self) -> None:
        df = _make_marubozu_bullish()
        assert is_marubozu_bearish(df)[0] is False

    def test_bearish_output_length(self) -> None:
        result = is_marubozu_bearish(OHLC)
        assert len(result) == _N

    def test_bearish_dtype(self) -> None:
        assert is_marubozu_bearish(OHLC).dtype == pl.Boolean

    def test_bearish_alias(self) -> None:
        assert is_marubozu_bearish(OHLC).name == "marubozu_bearish"

    def test_bearish_no_nulls(self) -> None:
        assert is_marubozu_bearish(OHLC).null_count() == 0
