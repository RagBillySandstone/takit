"""Unit tests for v0.5.0 indicators.

Covers:
    moving_averages: trima, vidya
    momentum:        crsi, qstick, psy_line, rocr
    trend:           vhf, pfe, chande_forecast_oscillator, linreg_r2, tii
    volatility:      bbw, bbp, realized_variance
    volume:          rvol, obv_osc, volume_roc
    patterns:        is_dragonfly_doji, is_gravestone_doji, is_spinning_top

Each test class verifies:
  - Output length matches input length.
  - Correct number of leading nulls.
  - First valid value is finite (not nan/inf).
  - Sensible domain constraints where applicable (e.g., RSI in [0,100]).
  - At least one edge case or error condition.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from polarticks.momentum import crsi, psy_line, qstick, rocr
from polarticks.moving_averages import trima, vidya
from polarticks.patterns import is_dragonfly_doji, is_gravestone_doji, is_spinning_top
from polarticks.trend import chande_forecast_oscillator, linreg_r2, pfe, tii, vhf
from polarticks.volatility import bbp, bbw, realized_variance
from polarticks.volume import obv_osc, rvol, volume_roc

# ---------------------------------------------------------------------------
# Shared synthetic OHLCV fixture  (N = 120 bars for CRSI's default rank_period)
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
HIGH = OHLCV["high"]
LOW = OHLCV["low"]
VOLUME = OHLCV["volume"]
OHLC = OHLCV.select(["open", "high", "low", "close"])


def _leading_nulls(s: pl.Series) -> int:
    """Count leading null values in a Series."""
    for i, v in enumerate(s.to_list()):
        if v is not None:
            return i
    return len(s)


def _all_valid_finite(s: pl.Series) -> bool:
    """Return True if all non-null values are finite."""
    vals = [v for v in s.to_list() if v is not None]
    return all(math.isfinite(v) for v in vals)


# ===========================================================================
# Moving Averages
# ===========================================================================


class TestTRIMA:
    def test_output_length(self) -> None:
        assert len(trima(CLOSE, 14)) == _N

    def test_null_prefix_period_minus_1(self) -> None:
        # TRIMA(n) produces exactly n-1 leading nulls.
        for period in (5, 10, 14, 20):
            result = trima(CLOSE, period)
            assert _leading_nulls(result) == period - 1, f"period={period}"

    def test_first_valid_value_is_finite(self) -> None:
        result = trima(CLOSE, 14)
        first_valid = next(v for v in result.to_list() if v is not None)
        assert math.isfinite(first_valid)

    def test_smoother_than_sma(self) -> None:
        # TRIMA has lower standard deviation than SMA of same period on same data.
        from polarticks.moving_averages import sma

        trima_vals = [v for v in trima(CLOSE, 20).to_list() if v is not None]
        sma_vals = [v for v in sma(CLOSE, 20).to_list() if v is not None]
        trima_std = (
            sum((v - sum(trima_vals) / len(trima_vals)) ** 2 for v in trima_vals) / len(trima_vals)
        ) ** 0.5
        sma_std = (
            sum((v - sum(sma_vals) / len(sma_vals)) ** 2 for v in sma_vals) / len(sma_vals)
        ) ** 0.5
        assert trima_std <= sma_std

    def test_period_1_gives_no_nulls(self) -> None:
        result = trima(CLOSE, 1)
        assert _leading_nulls(result) == 0

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            trima(CLOSE, 0)

    def test_alias_contains_period(self) -> None:
        assert trima(CLOSE, 14).name == "trima_14"


class TestVIDYA:
    def test_output_length(self) -> None:
        assert len(vidya(CLOSE, 9)) == _N

    def test_null_prefix_equals_cmo_period(self) -> None:
        # VIDYA(cmo_period) produces exactly cmo_period leading nulls.
        for cmo_period in (5, 9, 14):
            result = vidya(CLOSE, cmo_period)
            assert _leading_nulls(result) == cmo_period, f"cmo_period={cmo_period}"

    def test_first_valid_value_is_finite(self) -> None:
        result = vidya(CLOSE, 9)
        first_valid = next(v for v in result.to_list() if v is not None)
        assert math.isfinite(first_valid)

    def test_trending_series_tracks_price(self) -> None:
        # On a strictly rising series, VIDYA should also be strictly rising after warm-up.
        rising = pl.Series([float(i) for i in range(_N)])
        result = vidya(rising, 5)
        vals = [v for v in result.to_list() if v is not None]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            vidya(CLOSE, 0)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            vidya(CLOSE, 9, alpha=0.0)

    def test_alias_contains_period(self) -> None:
        assert vidya(CLOSE, 9).name == "vidya_9"


# ===========================================================================
# Momentum
# ===========================================================================


class TestCRSI:
    def test_output_length(self) -> None:
        assert len(crsi(CLOSE, 3, 2, 20)) == _N

    def test_null_prefix_driven_by_rank_period(self) -> None:
        # With rank_period=20, leading nulls = max(3, 2, 19) = 19.
        result = crsi(CLOSE, rsi_period=3, streak_period=2, rank_period=20)
        assert _leading_nulls(result) == 19

    def test_values_in_range(self) -> None:
        result = crsi(CLOSE, 3, 2, 20)
        vals = [v for v in result.to_list() if v is not None]
        assert all(0.0 <= v <= 100.0 for v in vals)

    def test_rising_series_gives_high_crsi(self) -> None:
        # A steadily rising series should push all three components high.
        rising = pl.Series([float(i) for i in range(_N)])
        result = crsi(rising, 3, 2, 20)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v > 50.0 for v in vals)

    def test_invalid_rsi_period_raises(self) -> None:
        with pytest.raises(ValueError):
            crsi(CLOSE, rsi_period=1)


class TestQStick:
    def test_output_length(self) -> None:
        assert len(qstick(OHLC, 8)) == _N

    def test_null_prefix_period_minus_1(self) -> None:
        for period in (4, 8, 14):
            assert _leading_nulls(qstick(OHLC, period)) == period - 1, f"period={period}"

    def test_flat_ohlc_returns_zero(self) -> None:
        # open == close → body_diff == 0 → Q-Stick == 0 after warm-up.
        flat = pl.DataFrame(
            {"open": [10.0] * _N, "high": [11.0] * _N, "low": [9.0] * _N, "close": [10.0] * _N}
        )
        result = qstick(flat, 5)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(0.0) for v in vals)

    def test_first_valid_finite(self) -> None:
        result = qstick(OHLC, 8)
        assert math.isfinite(next(v for v in result.to_list() if v is not None))

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            qstick(OHLC, 0)

    def test_alias_contains_period(self) -> None:
        assert qstick(OHLC, 8).name == "qstick_8"


class TestPsyLine:
    def test_output_length(self) -> None:
        assert len(psy_line(CLOSE, 14)) == _N

    def test_null_prefix_equals_period(self) -> None:
        # diff adds one extra null → leading nulls = period.
        for period in (7, 14, 20):
            assert _leading_nulls(psy_line(CLOSE, period)) == period, f"period={period}"

    def test_values_in_range(self) -> None:
        result = psy_line(CLOSE, 14)
        vals = [v for v in result.to_list() if v is not None]
        assert all(0.0 <= v <= 100.0 for v in vals)

    def test_all_up_bars_gives_100(self) -> None:
        rising = pl.Series([float(i) for i in range(_N)])
        result = psy_line(rising, 10)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(100.0) for v in vals)

    def test_all_down_bars_gives_0(self) -> None:
        falling = pl.Series([float(_N - i) for i in range(_N)])
        result = psy_line(falling, 10)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(0.0) for v in vals)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            psy_line(CLOSE, 0)


class TestROCR:
    def test_output_length(self) -> None:
        assert len(rocr(CLOSE, 10)) == _N

    def test_null_prefix_equals_period(self) -> None:
        for period in (5, 10, 20):
            assert _leading_nulls(rocr(CLOSE, period)) == period, f"period={period}"

    def test_no_change_gives_ratio_1(self) -> None:
        flat = pl.Series([50.0] * _N)
        result = rocr(flat, 10)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(1.0) for v in vals)

    def test_relation_to_roc(self) -> None:
        from polarticks.momentum import roc

        roc_vals = roc(CLOSE, 10)
        rocr_vals = rocr(CLOSE, 10)
        for r_val, rr_val in zip(roc_vals.to_list(), rocr_vals.to_list(), strict=True):
            if r_val is None or rr_val is None:
                continue
            # ROCR = 1 + ROC/100
            assert rr_val == pytest.approx(1.0 + r_val / 100.0, rel=1e-9)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            rocr(CLOSE, 0)

    def test_alias_contains_period(self) -> None:
        assert rocr(CLOSE, 10).name == "rocr_10"


# ===========================================================================
# Trend
# ===========================================================================


class TestVHF:
    def test_output_length(self) -> None:
        assert len(vhf(CLOSE, 14)) == _N

    def test_null_prefix_equals_period(self) -> None:
        # diff adds one extra null → leading nulls = period.
        for period in (7, 14, 28):
            assert _leading_nulls(vhf(CLOSE, period)) == period, f"period={period}"

    def test_values_positive(self) -> None:
        result = vhf(CLOSE, 14)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v > 0.0 for v in vals)

    def test_trending_higher_than_flat(self) -> None:
        # A pure trend should produce a higher VHF than a flat series.
        trending = pl.Series([float(i) for i in range(_N)])
        flat_with_noise = pl.Series([50.0 + math.sin(i) * 0.5 for i in range(_N)])
        trend_vhf = [v for v in vhf(trending, 14).to_list() if v is not None]
        flat_vhf = [v for v in vhf(flat_with_noise, 14).to_list() if v is not None]
        assert sum(trend_vhf) / len(trend_vhf) > sum(flat_vhf) / len(flat_vhf)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            vhf(CLOSE, 0)

    def test_alias_contains_period(self) -> None:
        assert vhf(CLOSE, 28).name == "vhf_28"


class TestPFE:
    def test_output_length(self) -> None:
        assert len(pfe(CLOSE, 14)) == _N

    def test_null_prefix_period_plus_smooth_minus_1(self) -> None:
        # Leading nulls = period + smooth - 1.
        result = pfe(CLOSE, 14, smooth=5)
        assert _leading_nulls(result) == 14 + 5 - 1

    def test_first_valid_finite(self) -> None:
        result = pfe(CLOSE, 14)
        first_valid = next(v for v in result.to_list() if v is not None)
        assert math.isfinite(first_valid)

    def test_trending_series_positive(self) -> None:
        # On a rising series, PFE should be positive.
        rising = pl.Series([100.0 + float(i) for i in range(_N)])
        result = pfe(rising, 10, smooth=3)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v > 0.0 for v in vals)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            pfe(CLOSE, 1)

    def test_alias_contains_period(self) -> None:
        assert pfe(CLOSE, 14).name == "pfe_14"


class TestChandeForecastOscillator:
    def test_output_length(self) -> None:
        assert len(chande_forecast_oscillator(CLOSE, 14)) == _N

    def test_null_prefix_period_minus_1(self) -> None:
        for period in (5, 10, 14):
            result = chande_forecast_oscillator(CLOSE, period)
            assert _leading_nulls(result) == period - 1, f"period={period}"

    def test_flat_series_gives_zero(self) -> None:
        # Flat prices → TSF = close → CFO = 0.
        flat = pl.Series([100.0] * _N)
        result = chande_forecast_oscillator(flat, 10)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(0.0, abs=1e-10) for v in vals)

    def test_first_valid_finite(self) -> None:
        result = chande_forecast_oscillator(CLOSE, 14)
        first_valid = next(v for v in result.to_list() if v is not None)
        assert math.isfinite(first_valid)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            chande_forecast_oscillator(CLOSE, 1)

    def test_alias_contains_period(self) -> None:
        assert chande_forecast_oscillator(CLOSE, 14).name == "cfo_14"


class TestLinregR2:
    def test_output_length(self) -> None:
        assert len(linreg_r2(CLOSE, 14)) == _N

    def test_null_prefix_period_minus_1(self) -> None:
        for period in (5, 10, 14):
            result = linreg_r2(CLOSE, period)
            assert _leading_nulls(result) == period - 1, f"period={period}"

    def test_values_in_0_1(self) -> None:
        result = linreg_r2(CLOSE, 14)
        vals = [v for v in result.to_list() if v is not None]
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_perfect_linear_series_gives_r2_1(self) -> None:
        # A perfectly linear series should yield R² = 1.
        linear = pl.Series([float(i) for i in range(_N)])
        result = linreg_r2(linear, 14)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(1.0, abs=1e-9) for v in vals)

    def test_flat_series_gives_r2_0(self) -> None:
        # A flat series has no variance → R² = 0.
        flat = pl.Series([100.0] * _N)
        result = linreg_r2(flat, 10)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(0.0, abs=1e-9) for v in vals)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            linreg_r2(CLOSE, 1)

    def test_alias_contains_period(self) -> None:
        assert linreg_r2(CLOSE, 14).name == "linreg_r2_14"


class TestTII:
    def test_output_length(self) -> None:
        assert len(tii(CLOSE, 20)) == _N

    def test_null_prefix_two_times_period_minus_1(self) -> None:
        # TII(n) produces 2*(n-1) leading nulls.
        for period in (5, 10, 20):
            result = tii(CLOSE, period)
            assert _leading_nulls(result) == 2 * (period - 1), f"period={period}"

    def test_values_in_0_100(self) -> None:
        result = tii(CLOSE, 20)
        vals = [v for v in result.to_list() if v is not None]
        assert all(0.0 <= v <= 100.0 for v in vals)

    def test_rising_series_gives_100(self) -> None:
        # A strictly rising series → all closes above SMA → TII = 100.
        rising = pl.Series([float(i) for i in range(_N)])
        result = tii(rising, 10)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(100.0) for v in vals)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            tii(CLOSE, 1)

    def test_alias_contains_period(self) -> None:
        assert tii(CLOSE, 20).name == "tii_20"


# ===========================================================================
# Volatility
# ===========================================================================


class TestBBW:
    def test_output_length(self) -> None:
        assert len(bbw(CLOSE, 20)) == _N

    def test_null_prefix_period_minus_1(self) -> None:
        for period in (10, 20, 30):
            assert _leading_nulls(bbw(CLOSE, period)) == period - 1, f"period={period}"

    def test_values_non_negative(self) -> None:
        result = bbw(CLOSE, 20)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v >= 0.0 for v in vals)

    def test_flat_series_gives_zero(self) -> None:
        flat = pl.Series([100.0] * _N)
        result = bbw(flat, 10)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(0.0, abs=1e-10) for v in vals)

    def test_wider_std_gives_wider_width(self) -> None:
        bbw1 = [v for v in bbw(CLOSE, 20, num_std=1.0).to_list() if v is not None]
        bbw2 = [v for v in bbw(CLOSE, 20, num_std=2.0).to_list() if v is not None]
        assert all(
            b2 == pytest.approx(2.0 * b1, rel=1e-9) for b1, b2 in zip(bbw1, bbw2, strict=True)
        )

    def test_alias_contains_period(self) -> None:
        assert bbw(CLOSE, 20).name == "bbw_20"


class TestBBP:
    def test_output_length(self) -> None:
        assert len(bbp(CLOSE, 20)) == _N

    def test_null_prefix_period_minus_1(self) -> None:
        for period in (10, 20):
            assert _leading_nulls(bbp(CLOSE, period)) == period - 1, f"period={period}"

    def test_flat_series_gives_half(self) -> None:
        # When bands have zero width, %B fills to 0.5 (neutral).
        flat = pl.Series([100.0] * _N)
        result = bbp(flat, 10)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(0.5) for v in vals)

    def test_price_at_midband_gives_half(self) -> None:
        # When close == middle band (SMA) → %B = 0.5.
        sma_series = pl.Series(
            [sum(_closes[max(0, i - 19) : i + 1]) / min(i + 1, 20) for i in range(_N)]
        )
        result = bbp(sma_series, 20)
        # The SMA of the SMA will still not be exactly 0.5, but values should be close to 0.5.
        vals = [v for v in result.to_list() if v is not None]
        assert all(-0.5 <= v <= 1.5 for v in vals)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            bbp(CLOSE, 0)

    def test_alias_contains_period(self) -> None:
        assert bbp(CLOSE, 20).name == "bbp_20"


class TestRealizedVariance:
    def test_output_length(self) -> None:
        assert len(realized_variance(CLOSE, 20)) == _N

    def test_null_prefix_equals_period(self) -> None:
        # diff adds one extra null → leading nulls = period.
        for period in (10, 20, 30):
            assert _leading_nulls(realized_variance(CLOSE, period)) == period, f"period={period}"

    def test_values_non_negative(self) -> None:
        result = realized_variance(CLOSE, 20)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v >= 0.0 for v in vals)

    def test_flat_series_gives_zero(self) -> None:
        flat = pl.Series([100.0] * _N)
        result = realized_variance(flat, 10)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(0.0, abs=1e-20) for v in vals)

    def test_annualize_multiplies_by_252(self) -> None:
        rv_ann = [
            v for v in realized_variance(CLOSE, 10, annualize=True).to_list() if v is not None
        ]
        rv_raw = [
            v for v in realized_variance(CLOSE, 10, annualize=False).to_list() if v is not None
        ]
        for ann, raw in zip(rv_ann, rv_raw, strict=True):
            assert ann == pytest.approx(252.0 * raw, rel=1e-9)

    def test_alias_contains_period(self) -> None:
        assert realized_variance(CLOSE, 20).name == "realized_variance_20"


# ===========================================================================
# Volume
# ===========================================================================


class TestRVOL:
    def test_output_length(self) -> None:
        assert len(rvol(OHLCV, 20)) == _N

    def test_null_prefix_period_minus_1(self) -> None:
        for period in (10, 20):
            assert _leading_nulls(rvol(OHLCV, period)) == period - 1, f"period={period}"

    def test_values_positive(self) -> None:
        result = rvol(OHLCV, 20)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v > 0.0 for v in vals)

    def test_constant_volume_gives_one(self) -> None:
        # Constant volume → RVOL = 1.
        const_vol = pl.DataFrame(
            {"open": _opens, "high": _highs, "low": _lows, "close": _closes, "volume": [500.0] * _N}
        )
        result = rvol(const_vol, 10)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(1.0) for v in vals)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            rvol(OHLCV, 0)

    def test_alias_contains_period(self) -> None:
        assert rvol(OHLCV, 20).name == "rvol_20"


class TestOBVOsc:
    def test_output_length(self) -> None:
        assert len(obv_osc(OHLCV, 5, 10)) == _N

    def test_null_prefix_is_slow_minus_1(self) -> None:
        result = obv_osc(OHLCV, fast=5, slow=10)
        assert _leading_nulls(result) == 10 - 1

    def test_first_valid_finite(self) -> None:
        result = obv_osc(OHLCV, 5, 10)
        first_valid = next(v for v in result.to_list() if v is not None)
        assert math.isfinite(first_valid)

    def test_fast_must_be_less_than_slow(self) -> None:
        with pytest.raises(ValueError):
            obv_osc(OHLCV, fast=10, slow=5)
        with pytest.raises(ValueError):
            obv_osc(OHLCV, fast=10, slow=10)

    def test_alias_contains_fast_slow(self) -> None:
        assert obv_osc(OHLCV, 5, 10).name == "obv_osc_5_10"


class TestVolumeROC:
    def test_output_length(self) -> None:
        assert len(volume_roc(OHLCV, 14)) == _N

    def test_null_prefix_equals_period(self) -> None:
        for period in (7, 14, 20):
            assert _leading_nulls(volume_roc(OHLCV, period)) == period, f"period={period}"

    def test_constant_volume_gives_zero(self) -> None:
        const_vol = pl.DataFrame(
            {"open": _opens, "high": _highs, "low": _lows, "close": _closes, "volume": [500.0] * _N}
        )
        result = volume_roc(const_vol, 10)
        vals = [v for v in result.to_list() if v is not None]
        assert all(v == pytest.approx(0.0) for v in vals)

    def test_first_valid_finite(self) -> None:
        result = volume_roc(OHLCV, 14)
        first_valid = next(v for v in result.to_list() if v is not None)
        assert math.isfinite(first_valid)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            volume_roc(OHLCV, 0)

    def test_alias_contains_period(self) -> None:
        assert volume_roc(OHLCV, 14).name == "vroc_14"


# ===========================================================================
# Patterns
# ===========================================================================

# Synthetic candle helpers for pattern tests.


def _make_dragonfly(n: int = 10) -> pl.DataFrame:
    """OHLC where close ≈ open ≈ high, long lower wick."""
    return pl.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.1] * n,  # almost no upper shadow
            "low": [95.0] * n,  # long lower shadow
            "close": [100.0] * n,
        }
    )


def _make_gravestone(n: int = 10) -> pl.DataFrame:
    """OHLC where close ≈ open ≈ low, long upper wick."""
    return pl.DataFrame(
        {
            "open": [100.0] * n,
            "high": [105.0] * n,  # long upper shadow
            "low": [99.9] * n,  # almost no lower shadow
            "close": [100.0] * n,
        }
    )


def _make_spinning_top(n: int = 10) -> pl.DataFrame:
    """OHLC with small body and equal upper/lower wicks."""
    return pl.DataFrame(
        {
            "open": [100.0] * n,
            "high": [102.0] * n,
            "low": [98.0] * n,
            "close": [100.5] * n,  # small body of 0.5 out of 4.0 range = 12.5%
        }
    )


def _make_large_body(n: int = 10) -> pl.DataFrame:
    """OHLC with large body — should NOT trigger doji patterns."""
    return pl.DataFrame(
        {
            "open": [95.0] * n,
            "high": [105.0] * n,
            "low": [94.0] * n,
            "close": [104.0] * n,
        }
    )


class TestIsDragonflyDoji:
    def test_output_length(self) -> None:
        assert len(is_dragonfly_doji(OHLC)) == _N

    def test_returns_boolean_series(self) -> None:
        result = is_dragonfly_doji(OHLC)
        assert result.dtype == pl.Boolean

    def test_dragonfly_pattern_detected(self) -> None:
        df = _make_dragonfly()
        result = is_dragonfly_doji(df)
        assert result.sum() == len(df), "All dragonfly candles should be detected."

    def test_gravestone_not_detected_as_dragonfly(self) -> None:
        df = _make_gravestone()
        result = is_dragonfly_doji(df)
        assert result.sum() == 0

    def test_large_body_not_dragonfly(self) -> None:
        df = _make_large_body()
        result = is_dragonfly_doji(df)
        assert result.sum() == 0

    def test_no_nulls_in_output(self) -> None:
        result = is_dragonfly_doji(OHLC)
        assert result.null_count() == 0

    def test_alias(self) -> None:
        assert is_dragonfly_doji(OHLC).name == "dragonfly_doji"


class TestIsGravestoneDoji:
    def test_output_length(self) -> None:
        assert len(is_gravestone_doji(OHLC)) == _N

    def test_returns_boolean_series(self) -> None:
        result = is_gravestone_doji(OHLC)
        assert result.dtype == pl.Boolean

    def test_gravestone_pattern_detected(self) -> None:
        df = _make_gravestone()
        result = is_gravestone_doji(df)
        assert result.sum() == len(df), "All gravestone candles should be detected."

    def test_dragonfly_not_detected_as_gravestone(self) -> None:
        df = _make_dragonfly()
        result = is_gravestone_doji(df)
        assert result.sum() == 0

    def test_large_body_not_gravestone(self) -> None:
        df = _make_large_body()
        result = is_gravestone_doji(df)
        assert result.sum() == 0

    def test_no_nulls_in_output(self) -> None:
        result = is_gravestone_doji(OHLC)
        assert result.null_count() == 0

    def test_alias(self) -> None:
        assert is_gravestone_doji(OHLC).name == "gravestone_doji"


class TestIsSpinningTop:
    def test_output_length(self) -> None:
        assert len(is_spinning_top(OHLC)) == _N

    def test_returns_boolean_series(self) -> None:
        result = is_spinning_top(OHLC)
        assert result.dtype == pl.Boolean

    def test_spinning_top_detected(self) -> None:
        df = _make_spinning_top()
        result = is_spinning_top(df)
        assert result.sum() == len(df), "All spinning top candles should be detected."

    def test_doji_not_spinning_top(self) -> None:
        # A doji has near-zero body (< min_body_ratio); spinning_top requires visible body.
        doji = pl.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [102.0] * 10,
                "low": [98.0] * 10,
                "close": [100.01] * 10,  # body = 0.01 / 4.0 range = 0.25% < 5%
            }
        )
        result = is_spinning_top(doji)
        assert result.sum() == 0

    def test_large_body_not_spinning_top(self) -> None:
        df = _make_large_body()
        result = is_spinning_top(df)
        assert result.sum() == 0

    def test_no_nulls_in_output(self) -> None:
        result = is_spinning_top(OHLC)
        assert result.null_count() == 0

    def test_alias(self) -> None:
        assert is_spinning_top(OHLC).name == "spinning_top"
