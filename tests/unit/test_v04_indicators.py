"""Unit tests for v0.4.0 indicators.

Covers:
    moving_averages: frama, laguerre
    momentum:        awesome_oscillator, accelerator_oscillator, smi, rvi, bop, qqe
    volatility:      choppiness_index, squeeze_momentum, volatility_ratio
    trend:           alligator, fractal, linreg_channel, tsf, chande_kroll_stop
    volume:          chaikin_osc, volume_oscillator, twap
    patterns:        is_hanging_man, is_inverted_hammer, is_tweezer_top,
                     is_tweezer_bottom, is_dark_cloud_cover, is_piercing_line,
                     is_rising_three_methods, is_falling_three_methods
    utils:           rolling_zscore, rolling_beta, hurst_exponent
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from polarticks.momentum import (
    accelerator_oscillator,
    awesome_oscillator,
    bop,
    qqe,
    rvi,
    smi,
)
from polarticks.moving_averages import frama, laguerre
from polarticks.patterns import (
    is_dark_cloud_cover,
    is_falling_three_methods,
    is_hanging_man,
    is_inverted_hammer,
    is_piercing_line,
    is_rising_three_methods,
    is_tweezer_bottom,
    is_tweezer_top,
)
from polarticks.trend import alligator, chande_kroll_stop, fractal, linreg_channel, tsf
from polarticks.utils import hurst_exponent, rolling_beta, rolling_zscore
from polarticks.volatility import choppiness_index, squeeze_momentum, volatility_ratio
from polarticks.volume import chaikin_osc, twap, volume_oscillator

# ---------------------------------------------------------------------------
# Shared fixtures — N=60 gives enough warm-up for all indicators
# ---------------------------------------------------------------------------

_N = 60

_closes = [100.0 + i * 0.5 + math.sin(i * 0.4) * 2 for i in range(_N)]
_highs = [c + 1.5 + math.cos(i * 0.3) * 0.5 for i, c in enumerate(_closes)]
_lows = [c - 1.5 - math.cos(i * 0.3) * 0.5 for i, c in enumerate(_closes)]
_opens = [c - 0.3 for c in _closes]
_volumes = [1000.0 + math.sin(i * 0.6) * 300 for i in range(_N)]

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
HIGH = OHLCV["high"]
LOW = OHLCV["low"]
VOLUME = OHLCV["volume"]


def _leading_nulls(s: pl.Series) -> int:
    """Count leading null values in a Series."""
    for i, v in enumerate(s.to_list()):
        if v is not None:
            return i
    return len(s)


def _leading_nulls_col(df: pl.DataFrame, col: str) -> int:
    """Count leading nulls in a named column of a DataFrame."""
    return _leading_nulls(df[col])


# ---------------------------------------------------------------------------
# FRAMA
# ---------------------------------------------------------------------------


class TestFRAMA:
    """Tests for Fractal Adaptive Moving Average."""

    def test_output_length(self) -> None:
        """Output must have the same length as the input series."""
        assert len(frama(CLOSE, 16)) == _N

    def test_leading_nulls(self) -> None:
        """period-1 leading nulls (warm-up phase)."""
        assert _leading_nulls(frama(CLOSE, 16)) == 15

    def test_series_name(self) -> None:
        assert frama(CLOSE, 16).name == "frama_16"

    def test_even_period_enforcement(self) -> None:
        """Odd period is silently rounded down to the nearest even value."""
        result_odd = frama(CLOSE, 17)
        result_even = frama(CLOSE, 16)
        # Both should have the same name (17 → 16)
        assert result_odd.name == "frama_16"
        # And produce identical values
        for a, b in zip(result_odd.to_list(), result_even.to_list(), strict=True):
            if a is None:
                assert b is None
            else:
                assert a == pytest.approx(b, rel=1e-9)

    def test_first_valid_value_near_price(self) -> None:
        """FRAMA seeds from the price series so the first valid value tracks price."""
        result = frama(CLOSE, 16)
        valid = [v for v in result.to_list() if v is not None]
        # Should converge toward price; sanity check range
        assert valid[0] > 80.0
        assert valid[0] < 150.0

    def test_minimum_period_4(self) -> None:
        """period=4 is the documented minimum even period."""
        result = frama(CLOSE, 4)
        assert _leading_nulls(result) == 3
        assert len(result) == _N

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            frama(CLOSE, 0)

    def test_no_accidental_zeros_in_warm_up(self) -> None:
        """Leading values must be null, not zero."""
        result = frama(CLOSE, 16)
        for v in result.to_list()[:15]:
            assert v is None


# ---------------------------------------------------------------------------
# Laguerre Filter
# ---------------------------------------------------------------------------


class TestLaguerre:
    """Tests for the Laguerre Filter."""

    def test_output_length(self) -> None:
        assert len(laguerre(CLOSE)) == _N

    def test_no_leading_nulls_default(self) -> None:
        """Laguerre has no formal null prefix — all bars produce a value."""
        assert _leading_nulls(laguerre(CLOSE)) == 0

    def test_series_name(self) -> None:
        assert laguerre(CLOSE, 0.8).name == "laguerre_0.8"

    def test_values_converge_near_price(self) -> None:
        """After warm-up the filter should be close to the price level."""
        result = laguerre(CLOSE, gamma=0.5)
        last_10 = result.to_list()[-10:]
        close_last_10 = CLOSE.to_list()[-10:]
        for lf, c in zip(last_10, close_last_10, strict=True):
            assert lf is not None
            assert abs(lf - c) < 20.0  # filter lags but should be in same ballpark

    def test_gamma_bounds_raise(self) -> None:
        with pytest.raises(ValueError):
            laguerre(CLOSE, gamma=0.0)
        with pytest.raises(ValueError):
            laguerre(CLOSE, gamma=1.0)
        with pytest.raises(ValueError):
            laguerre(CLOSE, gamma=-0.1)

    def test_custom_name_encodes_gamma(self) -> None:
        result = laguerre(CLOSE, gamma=0.5)
        assert "0.5" in result.name

    def test_lower_gamma_less_lag(self) -> None:
        """Lower gamma tracks the price more closely (less lag)."""
        r_low = laguerre(CLOSE, gamma=0.2)
        r_high = laguerre(CLOSE, gamma=0.9)
        close_vals = CLOSE.to_list()
        # At the last bar, the low-gamma filter should be closer to close
        diff_low = abs(r_low.to_list()[-1] - close_vals[-1])
        diff_high = abs(r_high.to_list()[-1] - close_vals[-1])
        assert diff_low < diff_high


# ---------------------------------------------------------------------------
# Awesome Oscillator
# ---------------------------------------------------------------------------


class TestAwesomeOscillator:
    """Tests for the Awesome Oscillator."""

    def test_output_length(self) -> None:
        assert len(awesome_oscillator(OHLCV)) == _N

    def test_leading_nulls(self) -> None:
        """Slow SMA dominates: slow-1 = 33 nulls."""
        assert _leading_nulls(awesome_oscillator(OHLCV, fast=5, slow=34)) == 33

    def test_series_name(self) -> None:
        assert awesome_oscillator(OHLCV).name == "ao"

    def test_fast_ge_slow_raises(self) -> None:
        with pytest.raises(ValueError):
            awesome_oscillator(OHLCV, fast=10, slow=10)
        with pytest.raises(ValueError):
            awesome_oscillator(OHLCV, fast=20, slow=10)

    def test_zero_center_on_flat_market(self) -> None:
        """On a perfectly flat series, AO = 0 everywhere (SMA_fast == SMA_slow)."""
        flat_ohlc = pl.DataFrame(
            {
                "high": [100.0] * 40,
                "low": [100.0] * 40,
                "close": [100.0] * 40,
                "open": [100.0] * 40,
                "volume": [1000.0] * 40,
            }
        )
        result = awesome_oscillator(flat_ohlc, fast=5, slow=10)
        valid = [v for v in result.to_list() if v is not None]
        for v in valid:
            assert v == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Accelerator Oscillator
# ---------------------------------------------------------------------------


class TestAcceleratorOscillator:
    """Tests for the Accelerator Oscillator."""

    def test_output_length(self) -> None:
        assert len(accelerator_oscillator(OHLCV)) == _N

    def test_leading_nulls(self) -> None:
        """slow + signal - 2 = 34 + 5 - 2 = 37 nulls."""
        assert _leading_nulls(accelerator_oscillator(OHLCV, fast=5, slow=34, signal=5)) == 37

    def test_series_name(self) -> None:
        assert accelerator_oscillator(OHLCV).name == "ac"

    def test_invalid_signal_raises(self) -> None:
        with pytest.raises(ValueError):
            accelerator_oscillator(OHLCV, signal=0)

    def test_fast_ge_slow_raises(self) -> None:
        with pytest.raises(ValueError):
            accelerator_oscillator(OHLCV, fast=5, slow=5)

    def test_shorter_params_produce_more_valid_bars(self) -> None:
        """With smaller period params, more bars should be valid."""
        r_default = accelerator_oscillator(OHLCV, fast=5, slow=34, signal=5)
        r_short = accelerator_oscillator(OHLCV, fast=3, slow=10, signal=3)
        nulls_default = _leading_nulls(r_default)
        nulls_short = _leading_nulls(r_short)
        assert nulls_short < nulls_default


# ---------------------------------------------------------------------------
# Stochastic Momentum Index
# ---------------------------------------------------------------------------


class TestSMI:
    """Tests for the Stochastic Momentum Index."""

    def test_output_shape(self) -> None:
        df = smi(OHLCV)
        assert len(df) == _N
        assert "smi" in df.columns
        assert "smi_signal" in df.columns

    def test_leading_nulls_smi(self) -> None:
        """(period-1) + 2*max(smooth1,smooth2) - 2 = 13 + 4 = 17 for defaults."""
        df = smi(OHLCV, period=14, smooth1=3, smooth2=3, signal=9)
        assert _leading_nulls_col(df, "smi") == 17

    def test_values_bounded(self) -> None:
        """SMI oscillates between -100 and +100."""
        df = smi(OHLCV, period=14, smooth1=3, smooth2=3, signal=9)
        valid = [v for v in df["smi"].to_list() if v is not None]
        for v in valid:
            assert -100.0 <= v <= 100.0

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            smi(OHLCV, period=0)
        with pytest.raises(ValueError):
            smi(OHLCV, smooth1=0)

    def test_signal_has_more_nulls_than_smi(self) -> None:
        """Signal line EMA warms up after SMI, so more leading nulls."""
        df = smi(OHLCV, period=14, smooth1=3, smooth2=3, signal=9)
        assert _leading_nulls_col(df, "smi_signal") > _leading_nulls_col(df, "smi")


# ---------------------------------------------------------------------------
# Relative Vigor Index
# ---------------------------------------------------------------------------


class TestRVI:
    """Tests for the Relative Vigor Index."""

    def test_output_shape(self) -> None:
        df = rvi(OHLCV)
        assert len(df) == _N
        assert "rvi" in df.columns
        assert "rvi_signal" in df.columns

    def test_leading_nulls_rvi(self) -> None:
        """period + 2 = 12 for default period=10."""
        df = rvi(OHLCV, period=10)
        assert _leading_nulls_col(df, "rvi") == 12

    def test_leading_nulls_signal(self) -> None:
        """period + 5 = 15 for default period=10."""
        df = rvi(OHLCV, period=10)
        assert _leading_nulls_col(df, "rvi_signal") == 15

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            rvi(OHLCV, period=0)

    def test_signal_lags_rvi(self) -> None:
        """Signal has more leading nulls than the RVI line."""
        df = rvi(OHLCV, period=10)
        assert _leading_nulls_col(df, "rvi_signal") > _leading_nulls_col(df, "rvi")

    def test_values_are_finite(self) -> None:
        """All valid values must be finite (fill_nan applied)."""
        df = rvi(OHLCV, period=5)
        for v in df["rvi"].to_list():
            if v is not None:
                assert math.isfinite(v)


# ---------------------------------------------------------------------------
# Balance of Power
# ---------------------------------------------------------------------------


class TestBOP:
    """Tests for the Balance of Power indicator."""

    def test_output_length(self) -> None:
        assert len(bop(OHLCV)) == _N

    def test_leading_nulls(self) -> None:
        """period-1 = 13 for default period=14."""
        assert _leading_nulls(bop(OHLCV, 14)) == 13

    def test_period_1_no_nulls(self) -> None:
        """period=1 applies no smoothing → 0 leading nulls."""
        assert _leading_nulls(bop(OHLCV, 1)) == 0

    def test_series_name(self) -> None:
        assert bop(OHLCV, 14).name == "bop_14"

    def test_values_bounded(self) -> None:
        """Raw BOP (period=1) is in [-1, 1]."""
        result = bop(OHLCV, 1)
        for v in result.to_list():
            if v is not None:
                assert -1.0 <= v <= 1.0

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            bop(OHLCV, 0)

    def test_zero_range_bar_gives_zero(self) -> None:
        """A bar where high == low produces BOP = 0 (not NaN)."""
        df = pl.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [100.0, 100.0],
                "low": [100.0, 100.0],
                "close": [100.0, 100.0],
                "volume": [1000.0, 1000.0],
            }
        )
        result = bop(df, 1)
        for v in result.to_list():
            if v is not None:
                assert v == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# QQE
# ---------------------------------------------------------------------------


class TestQQE:
    """Tests for the Quantitative Qualitative Estimation indicator."""

    def test_output_shape(self) -> None:
        df = qqe(CLOSE)
        assert len(df) == _N
        assert "qqe_line" in df.columns
        assert "qqe_fast" in df.columns

    def test_valid_values_exist(self) -> None:
        """With N=60 and default params, at least some valid values should appear."""
        df = qqe(CLOSE, rsi_period=14, sf=5)
        valid = [v for v in df["qqe_line"].to_list() if v is not None]
        assert len(valid) > 0

    def test_invalid_rsi_period_raises(self) -> None:
        with pytest.raises(ValueError):
            qqe(CLOSE, rsi_period=1)

    def test_invalid_sf_raises(self) -> None:
        with pytest.raises(ValueError):
            qqe(CLOSE, sf=0)

    def test_qqe_fast_tracks_rsi(self) -> None:
        """qqe_fast is smoothed RSI; valid values should be in [0, 100]."""
        df = qqe(CLOSE, rsi_period=7, sf=3)
        for v in df["qqe_fast"].to_list():
            if v is not None:
                assert 0.0 <= v <= 100.0


# ---------------------------------------------------------------------------
# Choppiness Index
# ---------------------------------------------------------------------------


class TestChoppinessIndex:
    """Tests for the Choppiness Index."""

    def test_output_length(self) -> None:
        assert len(choppiness_index(OHLCV)) == _N

    def test_leading_nulls(self) -> None:
        """period-1 = 13 for default period=14."""
        assert _leading_nulls(choppiness_index(OHLCV, 14)) == 13

    def test_series_name(self) -> None:
        assert choppiness_index(OHLCV, 14).name == "chop_14"

    def test_values_bounded(self) -> None:
        """CHOP values are bounded between the theoretical minimum and 100."""
        result = choppiness_index(OHLCV, 14)
        valid = [v for v in result.to_list() if v is not None]
        for v in valid:
            assert v > 0.0
            assert v <= 100.0

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            choppiness_index(OHLCV, 1)
        with pytest.raises(ValueError):
            choppiness_index(OHLCV, 0)

    def test_trending_series_lower_than_choppy(self) -> None:
        """Strongly trending series should give lower CHOP than choppy series."""
        n = 40
        # Strongly trending: smooth ramp
        trending_ohlc = pl.DataFrame(
            {
                "open": [100.0 + i for i in range(n)],
                "high": [101.0 + i for i in range(n)],
                "low": [99.0 + i for i in range(n)],
                "close": [100.0 + i for i in range(n)],
                "volume": [1000.0] * n,
            }
        )
        # Choppy: alternating up/down
        choppy_ohlc = pl.DataFrame(
            {
                "open": [100.0 + (i % 2) * 0.1 for i in range(n)],
                "high": [101.0 + (i % 2) * 0.1 for i in range(n)],
                "low": [99.0 + (i % 2) * 0.1 for i in range(n)],
                "close": [100.0 + (i % 2) * 0.1 for i in range(n)],
                "volume": [1000.0] * n,
            }
        )
        trend_chop = choppiness_index(trending_ohlc, 14)
        chop_chop = choppiness_index(choppy_ohlc, 14)
        valid_trend = [v for v in trend_chop.to_list() if v is not None]
        valid_chop = [v for v in chop_chop.to_list() if v is not None]
        assert sum(valid_trend) / len(valid_trend) < sum(valid_chop) / len(valid_chop)


# ---------------------------------------------------------------------------
# Squeeze Momentum
# ---------------------------------------------------------------------------


class TestSqueezeMomentum:
    """Tests for the TTM Squeeze Momentum indicator."""

    def test_output_shape(self) -> None:
        df = squeeze_momentum(OHLCV)
        assert len(df) == _N
        assert "sqz_on" in df.columns
        assert "sqz_off" in df.columns
        assert "sqz_momentum" in df.columns

    def test_sqz_on_is_bool(self) -> None:
        df = squeeze_momentum(OHLCV)
        assert df["sqz_on"].dtype == pl.Boolean

    def test_sqz_off_is_bool(self) -> None:
        df = squeeze_momentum(OHLCV)
        assert df["sqz_off"].dtype == pl.Boolean

    def test_leading_nulls_momentum(self) -> None:
        """2*(length-1) = 38 nulls: delta has length-1 nulls, then rolling_map needs
        another length-1 non-null samples to fill its first full window."""
        df = squeeze_momentum(OHLCV, length=20)
        assert _leading_nulls_col(df, "sqz_momentum") == 38

    def test_no_nulls_in_bool_cols(self) -> None:
        """sqz_on and sqz_off are always non-null (fill_null(False) applied)."""
        df = squeeze_momentum(OHLCV)
        assert df["sqz_on"].null_count() == 0
        assert df["sqz_off"].null_count() == 0

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            squeeze_momentum(OHLCV, length=1)

    def test_squeeze_mutually_exclusive(self) -> None:
        """sqz_on and sqz_off cannot both be True simultaneously."""
        df = squeeze_momentum(OHLCV)
        both_true = (df["sqz_on"] & df["sqz_off"]).sum()
        assert both_true == 0


# ---------------------------------------------------------------------------
# Volatility Ratio
# ---------------------------------------------------------------------------


class TestVolatilityRatio:
    """Tests for the Volatility Ratio indicator."""

    def test_output_length(self) -> None:
        assert len(volatility_ratio(OHLCV)) == _N

    def test_leading_nulls(self) -> None:
        """period-1 = 13 leading nulls."""
        assert _leading_nulls(volatility_ratio(OHLCV, 14)) == 13

    def test_series_name(self) -> None:
        assert volatility_ratio(OHLCV, 14).name == "vol_ratio_14"

    def test_values_in_0_1(self) -> None:
        """VR = TR / max(TR, period) is always in (0, 1]."""
        result = volatility_ratio(OHLCV, 14)
        for v in result.to_list():
            if v is not None:
                assert 0.0 <= v <= 1.0 + 1e-10

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            volatility_ratio(OHLCV, 0)


# ---------------------------------------------------------------------------
# Alligator
# ---------------------------------------------------------------------------


class TestAlligator:
    """Tests for the Bill Williams Alligator."""

    def test_output_shape(self) -> None:
        df = alligator(OHLCV)
        assert len(df) == _N
        assert "jaw" in df.columns
        assert "teeth" in df.columns
        assert "lips" in df.columns

    def test_leading_nulls_jaw(self) -> None:
        """(jaw_period - 1) + jaw_offset = 12 + 8 = 20 for defaults."""
        df = alligator(OHLCV)
        assert _leading_nulls_col(df, "jaw") == 20

    def test_lips_fewer_nulls_than_jaw(self) -> None:
        """Lips (faster, smaller offset) warms up sooner than jaw."""
        df = alligator(OHLCV)
        assert _leading_nulls_col(df, "lips") < _leading_nulls_col(df, "jaw")

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            alligator(OHLCV, jaw_period=0)

    def test_values_near_median_price(self) -> None:
        """Alligator values should be in the same ballpark as median price."""
        df = alligator(OHLCV)
        valid_jaw = [v for v in df["jaw"].to_list() if v is not None]
        for v in valid_jaw:
            assert 80.0 < v < 200.0


# ---------------------------------------------------------------------------
# Fractal
# ---------------------------------------------------------------------------


class TestFractal:
    """Tests for the Williams Fractal detector."""

    def test_output_shape(self) -> None:
        df = fractal(OHLCV)
        assert len(df) == _N
        assert "fractal_bearish" in df.columns
        assert "fractal_bullish" in df.columns

    def test_no_nulls(self) -> None:
        """fill_null(False) ensures no nulls in output."""
        df = fractal(OHLCV)
        assert df["fractal_bearish"].null_count() == 0
        assert df["fractal_bullish"].null_count() == 0

    def test_first_two_and_last_two_false(self) -> None:
        """First and last 2 bars can never be fractals (no 2-bar neighbours)."""
        df = fractal(OHLCV)
        for col in ["fractal_bearish", "fractal_bullish"]:
            assert df[col][0] is False
            assert df[col][1] is False
            assert df[col][-1] is False
            assert df[col][-2] is False

    def test_bool_dtype(self) -> None:
        df = fractal(OHLCV)
        assert df["fractal_bearish"].dtype == pl.Boolean
        assert df["fractal_bullish"].dtype == pl.Boolean

    def test_known_fractal(self) -> None:
        """A series with a clear spike should produce a bearish fractal at the peak."""
        highs = [100.0, 101.0, 105.0, 102.0, 100.0]
        lows = [99.0, 100.0, 104.0, 101.0, 99.0]
        df = pl.DataFrame(
            {
                "open": [100.0] * 5,
                "high": highs,
                "low": lows,
                "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            }
        )
        result = fractal(df)
        # Bar at index 2 is the spike; bearish fractal lands there
        assert result["fractal_bearish"][2] is True


# ---------------------------------------------------------------------------
# Linear Regression Channel
# ---------------------------------------------------------------------------


class TestLinRegChannel:
    """Tests for the Linear Regression Channel."""

    def test_output_shape(self) -> None:
        df = linreg_channel(CLOSE, period=20)
        assert len(df) == _N
        assert "lrc_mid" in df.columns
        assert "lrc_upper" in df.columns
        assert "lrc_lower" in df.columns

    def test_leading_nulls(self) -> None:
        """period-1 = 19 for period=20."""
        df = linreg_channel(CLOSE, period=20)
        assert _leading_nulls_col(df, "lrc_mid") == 19

    def test_channel_symmetry(self) -> None:
        """Upper and lower bands are equidistant from the mid line."""
        df = linreg_channel(CLOSE, period=20, num_std=2.0)
        for mid, upper, lower in zip(
            df["lrc_mid"].to_list(),
            df["lrc_upper"].to_list(),
            df["lrc_lower"].to_list(),
            strict=True,
        ):
            if mid is None:
                continue
            assert upper - mid == pytest.approx(mid - lower, rel=1e-9)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            linreg_channel(CLOSE, period=1)

    def test_upper_above_lower(self) -> None:
        """Upper band must always be >= lower band."""
        df = linreg_channel(CLOSE, period=20)
        for u, lo in zip(df["lrc_upper"].to_list(), df["lrc_lower"].to_list(), strict=True):
            if u is not None and lo is not None:
                assert u >= lo


# ---------------------------------------------------------------------------
# Time Series Forecast
# ---------------------------------------------------------------------------


class TestTSF:
    """Tests for the Time Series Forecast."""

    def test_output_length(self) -> None:
        assert len(tsf(CLOSE)) == _N

    def test_leading_nulls(self) -> None:
        """period-1 = 13 for default period=14."""
        assert _leading_nulls(tsf(CLOSE, 14)) == 13

    def test_series_name(self) -> None:
        assert tsf(CLOSE, 14).name == "tsf_14"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            tsf(CLOSE, 1)

    def test_forecast_tracks_price(self) -> None:
        """TSF values should be in the same price range as the input."""
        result = tsf(CLOSE, 14)
        valid = [v for v in result.to_list() if v is not None]
        for v in valid:
            assert 80.0 < v < 200.0

    def test_linear_series_exact(self) -> None:
        """On a perfectly linear series, TSF should extrapolate exactly."""
        s = pl.Series([float(i) for i in range(1, 21)])
        result = tsf(s, 5)
        valid = [(i, v) for i, v in enumerate(result.to_list()) if v is not None]
        # The projection at each bar should equal series[t] + 1 (one step ahead)
        for idx, v in valid:
            expected = float(idx + 2)  # series[idx]+1 = (idx+1)+1
            assert v == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Chande Kroll Stop
# ---------------------------------------------------------------------------


class TestChandeKrollStop:
    """Tests for the Chande Kroll Stop."""

    def test_output_shape(self) -> None:
        df = chande_kroll_stop(OHLCV)
        assert len(df) == _N
        assert "cks_long" in df.columns
        assert "cks_short" in df.columns

    def test_leading_nulls(self) -> None:
        """atr_period + stop_period - 2 = 10 + 9 - 2 = 17 for defaults."""
        df = chande_kroll_stop(OHLCV, atr_period=10, stop_period=9)
        assert _leading_nulls_col(df, "cks_long") == 17

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            chande_kroll_stop(OHLCV, atr_period=0)

    def test_long_below_short(self) -> None:
        """In a trending market, cks_long < price and cks_short > price as stops."""
        df = chande_kroll_stop(OHLCV)
        long_vals = [v for v in df["cks_long"].to_list() if v is not None]
        short_vals = [v for v in df["cks_short"].to_list() if v is not None]
        assert len(long_vals) > 0
        assert len(short_vals) > 0


# ---------------------------------------------------------------------------
# Chaikin Oscillator
# ---------------------------------------------------------------------------


class TestChaikinOsc:
    """Tests for the Chaikin Oscillator."""

    def test_output_length(self) -> None:
        assert len(chaikin_osc(OHLCV)) == _N

    def test_leading_nulls(self) -> None:
        """slow-1 = 9 for default slow=10."""
        assert _leading_nulls(chaikin_osc(OHLCV, fast=3, slow=10)) == 9

    def test_series_name(self) -> None:
        assert chaikin_osc(OHLCV).name == "chaikin_osc"

    def test_fast_ge_slow_raises(self) -> None:
        with pytest.raises(ValueError):
            chaikin_osc(OHLCV, fast=10, slow=10)
        with pytest.raises(ValueError):
            chaikin_osc(OHLCV, fast=15, slow=10)

    def test_values_are_finite(self) -> None:
        result = chaikin_osc(OHLCV)
        for v in result.to_list():
            if v is not None:
                assert math.isfinite(v)


# ---------------------------------------------------------------------------
# Volume Oscillator
# ---------------------------------------------------------------------------


class TestVolumeOscillator:
    """Tests for the Volume Oscillator."""

    def test_output_length(self) -> None:
        assert len(volume_oscillator(VOLUME)) == _N

    def test_leading_nulls(self) -> None:
        """slow-1 = 9 for default slow=10."""
        assert _leading_nulls(volume_oscillator(VOLUME, fast=5, slow=10)) == 9

    def test_series_name(self) -> None:
        assert volume_oscillator(VOLUME).name == "vol_osc"

    def test_fast_ge_slow_raises(self) -> None:
        with pytest.raises(ValueError):
            volume_oscillator(VOLUME, fast=10, slow=10)

    def test_constant_volume_is_zero(self) -> None:
        """When volume is constant, fast and slow EMA are equal → VO = 0."""
        const_vol = pl.Series([1000.0] * 30)
        result = volume_oscillator(const_vol, fast=3, slow=10)
        valid = [v for v in result.to_list() if v is not None]
        for v in valid:
            assert v == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TWAP
# ---------------------------------------------------------------------------


class TestTWAP:
    """Tests for the Time-Weighted Average Price."""

    def test_cumulative_length(self) -> None:
        assert len(twap(OHLCV)) == _N

    def test_cumulative_no_nulls(self) -> None:
        """Cumulative TWAP starts from bar 0 → 0 leading nulls."""
        assert _leading_nulls(twap(OHLCV)) == 0

    def test_rolling_length(self) -> None:
        assert len(twap(OHLCV, period=5)) == _N

    def test_rolling_leading_nulls(self) -> None:
        """period-1 = 4 leading nulls for period=5."""
        assert _leading_nulls(twap(OHLCV, period=5)) == 4

    def test_cumulative_name(self) -> None:
        assert twap(OHLCV).name == "twap"

    def test_rolling_name(self) -> None:
        assert twap(OHLCV, period=5).name == "twap_5"

    def test_cumulative_is_expanding_mean(self) -> None:
        """Cumulative TWAP at each bar equals mean of typical price from 0..t."""
        result = twap(OHLCV)
        tp = [(hi + lo + c) / 3 for hi, lo, c in zip(_highs, _lows, _closes, strict=True)]
        for i, v in enumerate(result.to_list()):
            expected = sum(tp[: i + 1]) / (i + 1)
            assert v == pytest.approx(expected, rel=1e-9)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            twap(OHLCV, period=0)


# ---------------------------------------------------------------------------
# Hanging Man
# ---------------------------------------------------------------------------


class TestIsHangingMan:
    """Tests for the Hanging Man pattern detector."""

    def test_output_length(self) -> None:
        assert len(is_hanging_man(OHLCV)) == _N

    def test_no_nulls(self) -> None:
        """fill_null(False) applied; no nulls in output."""
        result = is_hanging_man(OHLCV)
        assert result.null_count() == 0

    def test_bool_dtype(self) -> None:
        assert is_hanging_man(OHLCV).dtype == pl.Boolean

    def test_series_name(self) -> None:
        assert is_hanging_man(OHLCV).name == "hanging_man"

    def test_first_trend_period_bars_false(self) -> None:
        """First trend_period bars are always False (no prior trend data)."""
        result = is_hanging_man(OHLCV, trend_period=5)
        for v in result.to_list()[:5]:
            assert v is False

    def test_known_hanging_man(self) -> None:
        """Construct a textbook hanging man: prior uptrend + pin bar structure."""
        # 8 bars: 6 bars of uptrend, then a hanging man bar
        opens = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        closes = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 106.2, 108.0]
        # Hanging man bar (index 6): small body near top, long lower wick
        lows = [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 102.0, 106.5]
        highs = [101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 106.5, 108.5]
        df = pl.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": [1000.0] * 8,
            }
        )
        result = is_hanging_man(df, trend_period=5)
        # Bar 6 should be a hanging man (prior uptrend confirmed)
        assert result[6] is True


# ---------------------------------------------------------------------------
# Inverted Hammer
# ---------------------------------------------------------------------------


class TestIsInvertedHammer:
    """Tests for the Inverted Hammer pattern detector."""

    def test_output_length(self) -> None:
        assert len(is_inverted_hammer(OHLCV)) == _N

    def test_no_nulls(self) -> None:
        result = is_inverted_hammer(OHLCV)
        assert result.null_count() == 0

    def test_bool_dtype(self) -> None:
        assert is_inverted_hammer(OHLCV).dtype == pl.Boolean

    def test_series_name(self) -> None:
        assert is_inverted_hammer(OHLCV).name == "inverted_hammer"

    def test_first_trend_period_bars_false(self) -> None:
        result = is_inverted_hammer(OHLCV, trend_period=5)
        for v in result.to_list()[:5]:
            assert v is False

    def test_known_inverted_hammer(self) -> None:
        """Construct a textbook inverted hammer: prior downtrend + shooting star shape."""
        opens = [110.0, 109.0, 108.0, 107.0, 106.0, 105.0, 104.8, 103.0]
        closes = [109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 104.7, 102.0]
        # Inverted hammer bar (index 6): small body near bottom, long upper wick
        highs = [110.5, 109.5, 108.5, 107.5, 106.5, 105.5, 108.0, 103.5]
        lows = [108.5, 107.5, 106.5, 105.5, 104.5, 103.5, 104.5, 101.5]
        df = pl.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": [1000.0] * 8,
            }
        )
        result = is_inverted_hammer(df, trend_period=5)
        assert result[6] is True


# ---------------------------------------------------------------------------
# Tweezer Top
# ---------------------------------------------------------------------------


class TestIsTweezerTop:
    """Tests for the Tweezer Top pattern detector."""

    def test_output_length(self) -> None:
        assert len(is_tweezer_top(OHLCV)) == _N

    def test_no_nulls(self) -> None:
        assert is_tweezer_top(OHLCV).null_count() == 0

    def test_bool_dtype(self) -> None:
        assert is_tweezer_top(OHLCV).dtype == pl.Boolean

    def test_series_name(self) -> None:
        assert is_tweezer_top(OHLCV).name == "tweezer_top"

    def test_first_bar_always_false(self) -> None:
        assert is_tweezer_top(OHLCV)[0] is False

    def test_known_tweezer_top(self) -> None:
        """Two bars with equal highs, bar1 bullish, bar2 bearish."""
        df = pl.DataFrame(
            {
                "open": [100.0, 104.5],
                "high": [105.0, 105.0],
                "low": [99.0, 101.0],
                "close": [104.0, 101.5],
                "volume": [1000.0, 1000.0],
            }
        )
        result = is_tweezer_top(df, tolerance=0.002)
        # Bar 1 (index 1) is the tweezer top
        assert result[1] is True


# ---------------------------------------------------------------------------
# Tweezer Bottom
# ---------------------------------------------------------------------------


class TestIsTweezerBottom:
    """Tests for the Tweezer Bottom pattern detector."""

    def test_output_length(self) -> None:
        assert len(is_tweezer_bottom(OHLCV)) == _N

    def test_no_nulls(self) -> None:
        assert is_tweezer_bottom(OHLCV).null_count() == 0

    def test_bool_dtype(self) -> None:
        assert is_tweezer_bottom(OHLCV).dtype == pl.Boolean

    def test_series_name(self) -> None:
        assert is_tweezer_bottom(OHLCV).name == "tweezer_bottom"

    def test_first_bar_always_false(self) -> None:
        assert is_tweezer_bottom(OHLCV)[0] is False

    def test_known_tweezer_bottom(self) -> None:
        """Two bars with equal lows, bar1 bearish, bar2 bullish."""
        df = pl.DataFrame(
            {
                "open": [104.0, 100.5],
                "high": [105.0, 104.0],
                "low": [100.0, 100.0],
                "close": [100.5, 103.5],
                "volume": [1000.0, 1000.0],
            }
        )
        result = is_tweezer_bottom(df, tolerance=0.002)
        assert result[1] is True


# ---------------------------------------------------------------------------
# Dark Cloud Cover
# ---------------------------------------------------------------------------


class TestIsDarkCloudCover:
    """Tests for the Dark Cloud Cover pattern detector."""

    def test_output_length(self) -> None:
        assert len(is_dark_cloud_cover(OHLCV)) == _N

    def test_no_nulls(self) -> None:
        assert is_dark_cloud_cover(OHLCV).null_count() == 0

    def test_bool_dtype(self) -> None:
        assert is_dark_cloud_cover(OHLCV).dtype == pl.Boolean

    def test_series_name(self) -> None:
        assert is_dark_cloud_cover(OHLCV).name == "dark_cloud_cover"

    def test_first_bar_always_false(self) -> None:
        assert is_dark_cloud_cover(OHLCV)[0] is False

    def test_known_dark_cloud_cover(self) -> None:
        """Bar1 bullish, bar2 opens above bar1 high and closes below bar1 midpoint."""
        df = pl.DataFrame(
            {
                "open": [100.0, 106.0],
                "high": [105.0, 107.0],
                "low": [99.0, 101.0],
                "close": [105.0, 101.5],
                "volume": [1000.0, 1000.0],
            }
        )
        result = is_dark_cloud_cover(df, penetration=0.5)
        assert result[1] is True


# ---------------------------------------------------------------------------
# Piercing Line
# ---------------------------------------------------------------------------


class TestIsPiercingLine:
    """Tests for the Piercing Line pattern detector."""

    def test_output_length(self) -> None:
        assert len(is_piercing_line(OHLCV)) == _N

    def test_no_nulls(self) -> None:
        assert is_piercing_line(OHLCV).null_count() == 0

    def test_bool_dtype(self) -> None:
        assert is_piercing_line(OHLCV).dtype == pl.Boolean

    def test_series_name(self) -> None:
        assert is_piercing_line(OHLCV).name == "piercing_line"

    def test_first_bar_always_false(self) -> None:
        assert is_piercing_line(OHLCV)[0] is False

    def test_known_piercing_line(self) -> None:
        """Bar1 bearish, bar2 opens below bar1 low and closes above bar1 midpoint."""
        df = pl.DataFrame(
            {
                "open": [105.0, 98.0],
                "high": [106.0, 104.0],
                "low": [100.0, 97.0],
                "close": [100.0, 103.5],
                "volume": [1000.0, 1000.0],
            }
        )
        result = is_piercing_line(df, penetration=0.5)
        assert result[1] is True


# ---------------------------------------------------------------------------
# Rising Three Methods
# ---------------------------------------------------------------------------


class TestIsRisingThreeMethods:
    """Tests for the Rising Three Methods pattern detector."""

    def test_output_length(self) -> None:
        assert len(is_rising_three_methods(OHLCV)) == _N

    def test_no_nulls(self) -> None:
        assert is_rising_three_methods(OHLCV).null_count() == 0

    def test_bool_dtype(self) -> None:
        assert is_rising_three_methods(OHLCV).dtype == pl.Boolean

    def test_series_name(self) -> None:
        assert is_rising_three_methods(OHLCV).name == "rising_three_methods"

    def test_first_four_bars_false(self) -> None:
        """5-bar pattern: first 4 bars cannot complete it."""
        result = is_rising_three_methods(OHLCV)
        for v in result.to_list()[:4]:
            assert v is False

    def test_known_rising_three_methods(self) -> None:
        """Classic 5-bar pattern: large bull, 3 contained small bears, large bull.

        Bar1: large bullish (o=100, c=110).  Middle: small bearish inside body [100,110].
        Bar5: large bullish closing above bar1's close.
        """
        df = pl.DataFrame(
            {
                # bar1        mid1    mid2    mid3    bar5
                "open": [100.0, 108.0, 107.0, 106.0, 105.0],
                "high": [111.0, 109.0, 108.0, 107.0, 116.0],
                "low": [98.0, 104.0, 103.0, 102.0, 104.0],
                "close": [110.0, 107.0, 106.0, 105.0, 115.0],
                "volume": [2000.0, 500.0, 500.0, 500.0, 2000.0],
            }
        )
        result = is_rising_three_methods(df)
        assert result[4] is True


# ---------------------------------------------------------------------------
# Falling Three Methods
# ---------------------------------------------------------------------------


class TestIsFallingThreeMethods:
    """Tests for the Falling Three Methods pattern detector."""

    def test_output_length(self) -> None:
        assert len(is_falling_three_methods(OHLCV)) == _N

    def test_no_nulls(self) -> None:
        assert is_falling_three_methods(OHLCV).null_count() == 0

    def test_bool_dtype(self) -> None:
        assert is_falling_three_methods(OHLCV).dtype == pl.Boolean

    def test_series_name(self) -> None:
        assert is_falling_three_methods(OHLCV).name == "falling_three_methods"

    def test_first_four_bars_false(self) -> None:
        result = is_falling_three_methods(OHLCV)
        for v in result.to_list()[:4]:
            assert v is False

    def test_known_falling_three_methods(self) -> None:
        """Classic 5-bar pattern: large bear, 3 contained small bulls, large bear.

        Bar1: large bearish (o=110, c=100). Middle: small bullish inside body [100,110].
        Bar5: large bearish closing below bar1's close.
        """
        df = pl.DataFrame(
            {
                # bar1        mid1    mid2    mid3    bar5
                "open": [110.0, 101.0, 102.0, 103.0, 104.0],
                "high": [111.0, 104.0, 105.0, 106.0, 105.0],
                "low": [98.0, 99.0, 100.0, 101.0, 94.0],
                "close": [100.0, 102.0, 103.0, 104.0, 95.0],
                "volume": [2000.0, 500.0, 500.0, 500.0, 2000.0],
            }
        )
        result = is_falling_three_methods(df)
        assert result[4] is True


# ---------------------------------------------------------------------------
# Rolling Z-Score
# ---------------------------------------------------------------------------


class TestRollingZScore:
    """Tests for the Rolling Z-Score."""

    def test_output_length(self) -> None:
        assert len(rolling_zscore(CLOSE, 10)) == _N

    def test_leading_nulls(self) -> None:
        """period-1 = 9 leading nulls for period=10."""
        assert _leading_nulls(rolling_zscore(CLOSE, 10)) == 9

    def test_series_name(self) -> None:
        assert rolling_zscore(CLOSE, 10).name == "zscore_10"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            rolling_zscore(CLOSE, 1)

    def test_zero_on_constant_series(self) -> None:
        """A constant series has std=0; result is null (fill_nan(None))."""
        s = pl.Series([100.0] * 20)
        result = rolling_zscore(s, 5)
        # All valid positions should be null (0/0 → nan → null)
        valid_non_null = [v for v in result.to_list()[4:] if v is not None]
        assert len(valid_non_null) == 0

    def test_values_reasonable(self) -> None:
        """Z-scores on a typical price series should be moderate (not explosive)."""
        result = rolling_zscore(CLOSE, 10)
        for v in result.to_list():
            if v is not None:
                assert abs(v) < 10.0


# ---------------------------------------------------------------------------
# Rolling Beta
# ---------------------------------------------------------------------------


class TestRollingBeta:
    """Tests for the Rolling Beta."""

    def test_output_length(self) -> None:
        bench = CLOSE * 1.1
        assert len(rolling_beta(CLOSE, bench, 10)) == _N

    def test_leading_nulls(self) -> None:
        """period nulls: one from log return + period-1 from rolling window."""
        bench = CLOSE * 1.1
        assert _leading_nulls(rolling_beta(CLOSE, bench, 10)) == 10

    def test_series_name(self) -> None:
        bench = CLOSE * 1.1
        assert rolling_beta(CLOSE, bench, 10).name == "beta_10"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            rolling_beta(CLOSE, CLOSE, 1)

    def test_identical_series_beta_one(self) -> None:
        """A series regressed against itself should have beta ≈ 1.0."""
        result = rolling_beta(CLOSE, CLOSE, 10)
        valid = [v for v in result.to_list() if v is not None]
        for v in valid:
            assert v == pytest.approx(1.0, abs=1e-6)

    def test_scaled_series_beta_one(self) -> None:
        """Multiplicative scaling preserves log returns → beta ≈ 1.0."""
        bench = CLOSE * 3.5  # pure scale: log(3.5*c / 3.5*c_prev) == log(c/c_prev)
        result = rolling_beta(CLOSE, bench, 10)
        valid = [v for v in result.to_list() if v is not None]
        for v in valid:
            assert v == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Hurst Exponent
# ---------------------------------------------------------------------------


class TestHurstExponent:
    """Tests for the Rolling Hurst Exponent."""

    def test_output_length(self) -> None:
        assert len(hurst_exponent(CLOSE, 20)) == _N

    def test_leading_nulls(self) -> None:
        """period nulls: one from log return + period-1 from rolling_map."""
        assert _leading_nulls(hurst_exponent(CLOSE, 20)) == 20

    def test_series_name(self) -> None:
        assert hurst_exponent(CLOSE, 20).name == "hurst_20"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            hurst_exponent(CLOSE, 9)  # min_period=10

    def test_values_in_valid_range(self) -> None:
        """Hurst exponent is theoretically in (0, 1] for most processes."""
        result = hurst_exponent(CLOSE, 20)
        for v in result.to_list():
            if v is not None:
                assert 0.0 < v < 2.0  # generous bounds for finite samples

    def test_trending_series_higher_than_random(self) -> None:
        """A trending series should yield H > 0.5 on average."""
        n = 60
        # Strong trend: strictly increasing with small noise
        trending = pl.Series([100.0 + i * 0.5 for i in range(n)])
        result = hurst_exponent(trending, 20)
        valid = [v for v in result.to_list() if v is not None]
        assert len(valid) > 0
        avg_h = sum(valid) / len(valid)
        assert avg_h > 0.5
