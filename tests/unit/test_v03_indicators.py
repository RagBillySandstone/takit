"""Unit tests for v0.3.0 indicators.

Covers:
    fisher_transform, elder_ray, force_index, nvi, pvi,
    parkinson, garman_klass, yang_zhang, williams_vix_fix,
    fibonacci_retracement, rolling_highest, rolling_lowest,
    rolling_std, percent_rank
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from polarticks.levels import fibonacci_retracement
from polarticks.momentum import fisher_transform
from polarticks.trend import elder_ray
from polarticks.utils import percent_rank, rolling_highest, rolling_lowest, rolling_std
from polarticks.volatility import garman_klass, parkinson, williams_vix_fix, yang_zhang
from polarticks.volume import force_index, nvi, pvi

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_N = 50

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


def _leading_nulls(s: pl.Series) -> int:
    """Count leading null values in a Series."""
    for i, v in enumerate(s.to_list()):
        if v is not None:
            return i
    return len(s)


# ---------------------------------------------------------------------------
# Rolling Highest
# ---------------------------------------------------------------------------


class TestRollingHighest:
    """Tests for rolling_highest."""

    def test_output_length_matches_input(self) -> None:
        assert len(rolling_highest(CLOSE, 5)) == _N

    def test_leading_nulls(self) -> None:
        assert _leading_nulls(rolling_highest(CLOSE, 5)) == 4

    def test_period_1_returns_series(self) -> None:
        """period=1 window is the bar itself, so result equals input."""
        result = rolling_highest(CLOSE, 1)
        for a, b in zip(result.to_list(), CLOSE.to_list(), strict=True):
            assert a == pytest.approx(b)

    def test_monotone_rising_series(self) -> None:
        """On a rising series, the highest in any window is the last element."""
        s = pl.Series([float(i) for i in range(1, 11)])
        result = rolling_highest(s, 3)
        # First valid bar is index 2 (value 3.0); index 9 should be 10.0.
        assert result[2] == pytest.approx(3.0)
        assert result[9] == pytest.approx(10.0)

    def test_alias_includes_period(self) -> None:
        assert rolling_highest(CLOSE, 7).name == "highest_7"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            rolling_highest(CLOSE, 0)


# ---------------------------------------------------------------------------
# Rolling Lowest
# ---------------------------------------------------------------------------


class TestRollingLowest:
    """Tests for rolling_lowest."""

    def test_output_length_matches_input(self) -> None:
        assert len(rolling_lowest(CLOSE, 5)) == _N

    def test_leading_nulls(self) -> None:
        assert _leading_nulls(rolling_lowest(CLOSE, 5)) == 4

    def test_period_1_returns_series(self) -> None:
        result = rolling_lowest(CLOSE, 1)
        for a, b in zip(result.to_list(), CLOSE.to_list(), strict=True):
            assert a == pytest.approx(b)

    def test_monotone_rising_series(self) -> None:
        """On a rising series, the lowest in any window is the first element."""
        s = pl.Series([float(i) for i in range(1, 11)])
        result = rolling_lowest(s, 3)
        # Window at index 2: [1,2,3] → lowest=1.0.
        assert result[2] == pytest.approx(1.0)
        # Window at index 9: [8,9,10] → lowest=8.0.
        assert result[9] == pytest.approx(8.0)

    def test_alias_includes_period(self) -> None:
        assert rolling_lowest(CLOSE, 7).name == "lowest_7"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            rolling_lowest(CLOSE, 0)

    def test_highest_ge_lowest(self) -> None:
        """rolling_highest must always be >= rolling_lowest."""
        h = rolling_highest(CLOSE, 5).to_list()
        lo = rolling_lowest(CLOSE, 5).to_list()
        for hi_val, lo_val in zip(h, lo, strict=True):
            if hi_val is not None:
                assert hi_val >= lo_val


# ---------------------------------------------------------------------------
# Rolling Std
# ---------------------------------------------------------------------------


class TestRollingStd:
    """Tests for rolling_std."""

    def test_output_length_matches_input(self) -> None:
        assert len(rolling_std(CLOSE, 5)) == _N

    def test_leading_nulls(self) -> None:
        assert _leading_nulls(rolling_std(CLOSE, 5)) == 4

    def test_constant_series_gives_zero(self) -> None:
        """A flat series has zero variance, so std = 0.0."""
        flat = pl.Series([5.0] * 20)
        result = rolling_std(flat, 5)
        for v in result.drop_nulls().to_list():
            assert v == pytest.approx(0.0)

    def test_known_two_bar_window(self) -> None:
        """[1.0, 3.0]: sample std = sqrt(((1-2)^2 + (3-2)^2) / 1) = sqrt(2)."""
        s = pl.Series([1.0, 3.0])
        result = rolling_std(s, 2)
        assert result[1] == pytest.approx(math.sqrt(2.0))

    def test_non_negative(self) -> None:
        for v in rolling_std(CLOSE, 5).drop_nulls().to_list():
            assert v >= 0.0

    def test_alias_includes_period(self) -> None:
        assert rolling_std(CLOSE, 5).name == "std_5"

    def test_period_less_than_2_raises(self) -> None:
        with pytest.raises(ValueError):
            rolling_std(CLOSE, 1)


# ---------------------------------------------------------------------------
# Percent Rank
# ---------------------------------------------------------------------------


class TestPercentRank:
    """Tests for percent_rank."""

    def test_output_length_matches_input(self) -> None:
        assert len(percent_rank(CLOSE, 5)) == _N

    def test_leading_nulls(self) -> None:
        assert _leading_nulls(percent_rank(CLOSE, 5)) == 4

    def test_monotone_rising_every_bar_ranks_100(self) -> None:
        """In a rising series, each bar is the highest in any window → rank=100."""
        rising = pl.Series([float(i) for i in range(1, 21)])
        result = percent_rank(rising, 5)
        for v in result.drop_nulls().to_list():
            assert v == pytest.approx(100.0)

    def test_monotone_falling_first_valid_bar_ranks_low(self) -> None:
        """In a falling series the current bar is the minimum → rank = 1/period * 100."""
        falling = pl.Series([float(20 - i) for i in range(20)])
        period = 5
        result = percent_rank(falling, period)
        # Only the current bar satisfies <= current (it is the minimum).
        expected = 1.0 / period * 100.0
        for v in result.drop_nulls().to_list():
            assert v == pytest.approx(expected)

    def test_values_between_0_and_100(self) -> None:
        for v in percent_rank(CLOSE, 10).drop_nulls().to_list():
            assert 0.0 <= v <= 100.0

    def test_alias_includes_period(self) -> None:
        assert percent_rank(CLOSE, 5).name == "prank_5"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            percent_rank(CLOSE, 0)


# ---------------------------------------------------------------------------
# Fisher Transform
# ---------------------------------------------------------------------------


class TestFisherTransform:
    """Tests for fisher_transform."""

    def test_output_columns(self) -> None:
        result = fisher_transform(OHLCV, 9)
        assert set(result.columns) == {"fisher", "fisher_signal"}

    def test_output_length_matches_input(self) -> None:
        result = fisher_transform(OHLCV, 9)
        assert len(result) == _N

    def test_fisher_leading_nulls(self) -> None:
        period = 9
        result = fisher_transform(OHLCV, period)
        assert _leading_nulls(result["fisher"]) == period - 1

    def test_signal_leads_fisher_by_one(self) -> None:
        """signal is fisher shifted by 1, so it has one more leading null."""
        period = 9
        result = fisher_transform(OHLCV, period)
        assert _leading_nulls(result["fisher_signal"]) == period

    def test_signal_equals_previous_fisher(self) -> None:
        """At each valid bar, fisher_signal[t] == fisher[t-1]."""
        result = fisher_transform(OHLCV, 5)
        fisher_list = result["fisher"].to_list()
        signal_list = result["fisher_signal"].to_list()
        # Compare from the first bar where signal is valid.
        for i in range(5, _N):
            assert signal_list[i] == pytest.approx(fisher_list[i - 1], rel=1e-9)

    def test_flat_hl_series_gives_zero_fisher(self) -> None:
        """A series where H=L collapses the range to 0 → value=0 → fisher=0."""
        flat = pl.DataFrame({"high": [100.0] * 15, "low": [100.0] * 15, "close": [100.0] * 15})
        result = fisher_transform(flat, 5)
        for v in result["fisher"].drop_nulls().to_list():
            assert v == pytest.approx(0.0)

    def test_values_are_finite(self) -> None:
        result = fisher_transform(OHLCV, 9)
        for v in result["fisher"].drop_nulls().to_list():
            assert math.isfinite(v)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            fisher_transform(OHLCV, 0)


# ---------------------------------------------------------------------------
# Elder Ray
# ---------------------------------------------------------------------------


class TestElderRay:
    """Tests for elder_ray."""

    def test_output_columns(self) -> None:
        result = elder_ray(OHLCV, 13)
        assert set(result.columns) == {"bull_power", "bear_power"}

    def test_output_length_matches_input(self) -> None:
        assert len(elder_ray(OHLCV, 13)) == _N

    def test_leading_nulls(self) -> None:
        period = 13
        result = elder_ray(OHLCV, period)
        assert _leading_nulls(result["bull_power"]) == period - 1
        assert _leading_nulls(result["bear_power"]) == period - 1

    def test_bull_power_positive_for_rising_highs(self) -> None:
        """On a steadily rising series with high above close, bull_power should be positive."""
        closes = pl.Series([float(i) + 100 for i in range(30)])
        df = pl.DataFrame(
            {
                "open": (closes - 0.1).to_list(),
                "high": (closes + 2.0).to_list(),
                "low": (closes - 2.0).to_list(),
                "close": closes.to_list(),
            }
        )
        result = elder_ray(df, period=5)
        # Bull power = high - EMA(close); high = close+2, EMA≈close for a rising series.
        for v in result["bull_power"].drop_nulls().to_list():
            assert v > 0.0

    def test_bear_power_negative_for_low_below_close(self) -> None:
        """low < EMA(close) → bear_power < 0."""
        closes = pl.Series([float(i) + 100 for i in range(30)])
        df = pl.DataFrame(
            {
                "open": (closes - 0.1).to_list(),
                "high": (closes + 2.0).to_list(),
                "low": (closes - 2.0).to_list(),
                "close": closes.to_list(),
            }
        )
        result = elder_ray(df, period=5)
        for v in result["bear_power"].drop_nulls().to_list():
            assert v < 0.0

    def test_bull_minus_bear_equals_hl_range(self) -> None:
        """bull_power - bear_power = high - low (the EMA cancels out)."""
        result = elder_ray(OHLCV, period=5)
        bull = result["bull_power"].to_list()
        bear = result["bear_power"].to_list()
        for b, br, h, lo in zip(bull, bear, HIGH.to_list(), LOW.to_list(), strict=True):
            if b is not None:
                assert (b - br) == pytest.approx(h - lo, rel=1e-9)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            elder_ray(OHLCV, 0)


# ---------------------------------------------------------------------------
# Force Index
# ---------------------------------------------------------------------------


class TestForceIndex:
    """Tests for force_index."""

    def test_output_length_matches_input(self) -> None:
        assert len(force_index(OHLCV, 13)) == _N

    def test_leading_nulls(self) -> None:
        """EMA of the raw force series → period - 1 leading nulls."""
        period = 13
        assert _leading_nulls(force_index(OHLCV, period)) == period - 1

    def test_flat_series_gives_zero(self) -> None:
        """No price change → raw force is 0 every bar → EMA = 0."""
        df = pl.DataFrame(
            {
                "open": [100.0] * 20,
                "high": [101.0] * 20,
                "low": [99.0] * 20,
                "close": [100.0] * 20,
                "volume": [1000.0] * 20,
            }
        )
        result = force_index(df, period=5)
        for v in result.drop_nulls().to_list():
            assert v == pytest.approx(0.0)

    def test_rising_prices_give_positive_force(self) -> None:
        """Steadily rising close with constant volume → force_index > 0."""
        closes = [100.0 + i for i in range(30)]
        df = pl.DataFrame(
            {
                "open": [c - 0.5 for c in closes],
                "high": [c + 1.0 for c in closes],
                "low": [c - 1.0 for c in closes],
                "close": closes,
                "volume": [1000.0] * 30,
            }
        )
        result = force_index(df, period=5)
        for v in result.drop_nulls().to_list():
            assert v > 0.0

    def test_alias_includes_period(self) -> None:
        assert force_index(OHLCV, 13).name == "force_index_13"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            force_index(OHLCV, 0)


# ---------------------------------------------------------------------------
# NVI
# ---------------------------------------------------------------------------


class TestNVI:
    """Tests for Negative Volume Index."""

    def test_output_length_matches_input(self) -> None:
        assert len(nvi(OHLCV)) == _N

    def test_no_leading_nulls(self) -> None:
        """NVI is cumulative from bar 0 — no warm-up."""
        assert _leading_nulls(nvi(OHLCV)) == 0

    def test_starts_at_1000(self) -> None:
        assert nvi(OHLCV)[0] == pytest.approx(1000.0)

    def test_unchanged_on_rising_volume(self) -> None:
        """NVI only changes when volume falls; rising-volume bars are flat."""
        # volume[1] > volume[0] → NVI[1] = NVI[0].
        df = pl.DataFrame(
            {
                "open": [10.0, 12.0, 11.0, 13.0],
                "high": [11.0, 13.0, 12.0, 14.0],
                "low": [9.0, 11.0, 10.0, 12.0],
                "close": [10.0, 12.0, 11.0, 13.0],
                "volume": [100.0, 200.0, 50.0, 150.0],
            }
        )
        result = nvi(df).to_list()
        # bar 0: 1000.0 (seed)
        assert result[0] == pytest.approx(1000.0)
        # bar 1: vol 200 > 100 → unchanged.
        assert result[1] == pytest.approx(1000.0)
        # bar 2: vol 50 < 200 → 1000 * (1 + (11-12)/12).
        assert result[2] == pytest.approx(1000.0 * (1.0 + (11.0 - 12.0) / 12.0))
        # bar 3: vol 150 > 50 → unchanged.
        assert result[3] == pytest.approx(result[2])

    def test_output_name(self) -> None:
        assert nvi(OHLCV).name == "nvi"

    def test_null_bar_carries_forward(self) -> None:
        """A bar with null OHLCV data must carry the prior NVI value forward."""
        df = pl.DataFrame(
            {
                "open": [10.0, 12.0, None, 13.0],
                "high": [11.0, 13.0, None, 14.0],
                "low": [9.0, 11.0, None, 12.0],
                "close": [10.0, 12.0, None, 13.0],
                "volume": [100.0, 200.0, None, 50.0],
            }
        )
        result = nvi(df).to_list()
        # bar 2 is null → carries bar 1's value; bar 3 has volume 50 < 200 → updates.
        assert result[2] == pytest.approx(result[1])

    def test_zero_prev_close_carries_forward(self) -> None:
        """A zero prev_close must not cause division by zero — value carries forward."""
        df = pl.DataFrame(
            {
                "open": [0.0, 1.0, 2.0],
                "high": [0.5, 1.5, 2.5],
                "low": [0.0, 0.5, 1.5],
                "close": [0.0, 1.0, 2.0],
                # volume falls at bar 1 so NVI would normally update, but prev_close=0
                "volume": [100.0, 50.0, 150.0],
            }
        )
        result = nvi(df).to_list()
        # bar 0 → 1000.0 (seed); bar 1: prev_close=0 → guard fires → carry forward.
        assert result[1] == pytest.approx(result[0])


# ---------------------------------------------------------------------------
# PVI
# ---------------------------------------------------------------------------


class TestPVI:
    """Tests for Positive Volume Index."""

    def test_output_length_matches_input(self) -> None:
        assert len(pvi(OHLCV)) == _N

    def test_no_leading_nulls(self) -> None:
        assert _leading_nulls(pvi(OHLCV)) == 0

    def test_starts_at_1000(self) -> None:
        assert pvi(OHLCV)[0] == pytest.approx(1000.0)

    def test_changes_on_rising_volume_only(self) -> None:
        """PVI changes when volume rises; falling-volume bars are flat."""
        df = pl.DataFrame(
            {
                "open": [10.0, 12.0, 11.0, 13.0],
                "high": [11.0, 13.0, 12.0, 14.0],
                "low": [9.0, 11.0, 10.0, 12.0],
                "close": [10.0, 12.0, 11.0, 13.0],
                "volume": [100.0, 200.0, 50.0, 150.0],
            }
        )
        result = pvi(df).to_list()
        # bar 0: 1000.0 (seed)
        assert result[0] == pytest.approx(1000.0)
        # bar 1: vol 200 > 100 → 1000 * (1 + (12-10)/10).
        assert result[1] == pytest.approx(1000.0 * (1.0 + (12.0 - 10.0) / 10.0))
        # bar 2: vol 50 < 200 → unchanged.
        assert result[2] == pytest.approx(result[1])
        # bar 3: vol 150 > 50 → pvi[2] * (1 + (13-11)/11).
        assert result[3] == pytest.approx(result[2] * (1.0 + (13.0 - 11.0) / 11.0))

    def test_output_name(self) -> None:
        assert pvi(OHLCV).name == "pvi"

    def test_null_bar_carries_forward(self) -> None:
        """A bar with null OHLCV data must carry the prior PVI value forward."""
        df = pl.DataFrame(
            {
                "open": [10.0, 12.0, None, 13.0],
                "high": [11.0, 13.0, None, 14.0],
                "low": [9.0, 11.0, None, 12.0],
                "close": [10.0, 12.0, None, 13.0],
                "volume": [100.0, 200.0, None, 50.0],
            }
        )
        result = pvi(df).to_list()
        # bar 2 is null → carries bar 1's value.
        assert result[2] == pytest.approx(result[1])

    def test_zero_prev_close_carries_forward(self) -> None:
        """A zero prev_close must not cause division by zero — value carries forward."""
        df = pl.DataFrame(
            {
                "open": [0.0, 1.0, 2.0],
                "high": [0.5, 1.5, 2.5],
                "low": [0.0, 0.5, 1.5],
                "close": [0.0, 1.0, 2.0],
                # volume rises at bar 1 so PVI would normally update, but prev_close=0
                "volume": [100.0, 200.0, 50.0],
            }
        )
        result = pvi(df).to_list()
        # bar 0 → 1000.0 (seed); bar 1: prev_close=0 → guard fires → carry forward.
        assert result[1] == pytest.approx(result[0])

    def test_nvi_pvi_diverge(self) -> None:
        """NVI and PVI cannot be numerically identical except by coincidence."""
        n_vals = nvi(OHLCV).to_list()
        p_vals = pvi(OHLCV).to_list()
        # They should differ on at least one bar (given the mixed volume pattern).
        assert n_vals != p_vals


# ---------------------------------------------------------------------------
# Parkinson Volatility
# ---------------------------------------------------------------------------


class TestParkinson:
    """Tests for the Parkinson volatility estimator."""

    def test_output_length_matches_input(self) -> None:
        assert len(parkinson(OHLCV, 10)) == _N

    def test_leading_nulls(self) -> None:
        period = 10
        assert _leading_nulls(parkinson(OHLCV, period)) == period - 1

    def test_non_negative(self) -> None:
        for v in parkinson(OHLCV, 10).drop_nulls().to_list():
            assert v >= 0.0

    def test_constant_hl_ratio_gives_zero(self) -> None:
        """If H = L on every bar, ln(H/L) = 0 → parkinson = 0."""
        df = pl.DataFrame(
            {
                "open": [100.0] * 20,
                "high": [100.0] * 20,
                "low": [100.0] * 20,
                "close": [100.0] * 20,
            }
        )
        result = parkinson(df, period=5, annualise=False)
        for v in result.drop_nulls().to_list():
            assert v == pytest.approx(0.0)

    def test_annualise_scales_by_sqrt_trading_days(self) -> None:
        """annualise=True must equal annualise=False × sqrt(trading_days)."""
        raw = parkinson(OHLCV, 10, annualise=False)
        ann = parkinson(OHLCV, 10, annualise=True, trading_days=252)
        for r, a in zip(raw.to_list(), ann.to_list(), strict=True):
            if r is not None:
                assert a == pytest.approx(r * math.sqrt(252.0), rel=1e-9)

    def test_alias_includes_period(self) -> None:
        assert parkinson(OHLCV, 10).name == "parkinson_10"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            parkinson(OHLCV, 1)


# ---------------------------------------------------------------------------
# Garman-Klass Volatility
# ---------------------------------------------------------------------------


class TestGarmanKlass:
    """Tests for the Garman-Klass volatility estimator."""

    def test_output_length_matches_input(self) -> None:
        assert len(garman_klass(OHLCV, 10)) == _N

    def test_leading_nulls(self) -> None:
        period = 10
        assert _leading_nulls(garman_klass(OHLCV, period)) == period - 1

    def test_non_negative(self) -> None:
        for v in garman_klass(OHLCV, 10).drop_nulls().to_list():
            assert v >= 0.0

    def test_no_drift_flat_series_gives_zero(self) -> None:
        """When H=L (and O=C), both log terms are zero → GK = 0."""
        df = pl.DataFrame(
            {
                "open": [100.0] * 20,
                "high": [100.0] * 20,
                "low": [100.0] * 20,
                "close": [100.0] * 20,
            }
        )
        result = garman_klass(df, period=5, annualise=False)
        for v in result.drop_nulls().to_list():
            assert v == pytest.approx(0.0)

    def test_annualise_scales_by_sqrt_trading_days(self) -> None:
        raw = garman_klass(OHLCV, 10, annualise=False)
        ann = garman_klass(OHLCV, 10, annualise=True, trading_days=252)
        for r, a in zip(raw.to_list(), ann.to_list(), strict=True):
            if r is not None:
                assert a == pytest.approx(r * math.sqrt(252.0), rel=1e-9)

    def test_alias_includes_period(self) -> None:
        assert garman_klass(OHLCV, 10).name == "garman_klass_10"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            garman_klass(OHLCV, 1)


# ---------------------------------------------------------------------------
# Yang-Zhang Volatility
# ---------------------------------------------------------------------------


class TestYangZhang:
    """Tests for the Yang-Zhang volatility estimator."""

    def test_output_length_matches_input(self) -> None:
        assert len(yang_zhang(OHLCV, 10)) == _N

    def test_leading_nulls(self) -> None:
        """Overnight shift(1) gives one extra null → period leading nulls total."""
        period = 10
        assert _leading_nulls(yang_zhang(OHLCV, period)) == period

    def test_non_negative(self) -> None:
        for v in yang_zhang(OHLCV, 10).drop_nulls().to_list():
            assert v >= 0.0

    def test_annualise_scales_by_sqrt_trading_days(self) -> None:
        raw = yang_zhang(OHLCV, 10, annualise=False)
        ann = yang_zhang(OHLCV, 10, annualise=True, trading_days=252)
        for r, a in zip(raw.to_list(), ann.to_list(), strict=True):
            if r is not None:
                assert a == pytest.approx(r * math.sqrt(252.0), rel=1e-9)

    def test_alias_includes_period(self) -> None:
        assert yang_zhang(OHLCV, 10).name == "yang_zhang_10"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            yang_zhang(OHLCV, 1)

    def test_larger_than_parkinson_with_overnight_gaps(self) -> None:
        """When large overnight gaps are present, YZ should exceed Parkinson's estimate."""
        # Alternate open: one bar gaps up 5%, next gaps down 5% — creating large overnight moves.
        closes = [100.0 + i * 0.1 for i in range(30)]
        opens = [c * 1.05 if i % 2 == 0 else c * 0.95 for i, c in enumerate(closes)]
        df = pl.DataFrame(
            {
                "open": opens,
                "high": [c + 2.0 for c in closes],
                "low": [c - 2.0 for c in closes],
                "close": closes,
            }
        )
        yz_vals = [v for v in yang_zhang(df, 10, annualise=False).to_list() if v is not None]
        pk_vals = [v for v in parkinson(df, 10, annualise=False).to_list() if v is not None]
        # At least some YZ values should exceed the corresponding Parkinson values.
        assert any(yz > pk for yz, pk in zip(yz_vals, pk_vals, strict=False))


# ---------------------------------------------------------------------------
# Williams VIX Fix
# ---------------------------------------------------------------------------


class TestWilliamsVixFix:
    """Tests for williams_vix_fix."""

    def test_output_length_matches_input(self) -> None:
        assert len(williams_vix_fix(OHLCV, 22)) == _N

    def test_leading_nulls(self) -> None:
        period = 22
        assert _leading_nulls(williams_vix_fix(OHLCV, period)) == period - 1

    def test_non_negative_for_realistic_data(self) -> None:
        """WVF = (max_close - low) / max_close ≥ 0 when close ≥ low (normal OHLC)."""
        for v in williams_vix_fix(OHLCV, 10).drop_nulls().to_list():
            assert v >= 0.0

    def test_flat_series_gives_zero(self) -> None:
        """If high = low = close everywhere, (max_close - low) = 0 → WVF = 0."""
        df = pl.DataFrame(
            {
                "open": [100.0] * 20,
                "high": [100.0] * 20,
                "low": [100.0] * 20,
                "close": [100.0] * 20,
            }
        )
        result = williams_vix_fix(df, period=5)
        for v in result.drop_nulls().to_list():
            assert v == pytest.approx(0.0)

    def test_spikes_when_low_far_from_rolling_high(self) -> None:
        """A sharp drop to a low far below the recent high should give a large WVF."""
        closes = [100.0] * 15 + [60.0]  # close collapses on bar 15
        lows = [99.0] * 15 + [50.0]  # low is even lower on the crash bar
        df = pl.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": lows,
                "close": closes,
            }
        )
        result = williams_vix_fix(df, period=10)
        last_valid = result.drop_nulls()[-1]
        # Highest close in the window is 100.0; low is 50.0 → WVF = 50%.
        assert last_valid == pytest.approx(50.0, rel=1e-6)

    def test_alias_includes_period(self) -> None:
        assert williams_vix_fix(OHLCV, 22).name == "wvf_22"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            williams_vix_fix(OHLCV, 0)


# ---------------------------------------------------------------------------
# Fibonacci Retracement
# ---------------------------------------------------------------------------


class TestFibonacciRetracement:
    """Tests for fibonacci_retracement."""

    def test_output_columns(self) -> None:
        result = fibonacci_retracement(HIGH, LOW)
        assert set(result.columns) == {
            "fib_0",
            "fib_236",
            "fib_382",
            "fib_500",
            "fib_618",
            "fib_786",
            "fib_100",
        }

    def test_output_length_matches_input(self) -> None:
        result = fibonacci_retracement(HIGH, LOW)
        assert len(result) == _N

    def test_fib_0_equals_high(self) -> None:
        result = fibonacci_retracement(HIGH, LOW)
        for h, f0 in zip(HIGH.to_list(), result["fib_0"].to_list(), strict=True):
            assert f0 == pytest.approx(h)

    def test_fib_100_equals_low(self) -> None:
        result = fibonacci_retracement(HIGH, LOW)
        for lo, f100 in zip(LOW.to_list(), result["fib_100"].to_list(), strict=True):
            assert f100 == pytest.approx(lo)

    def test_fib_500_is_midpoint(self) -> None:
        """fib_500 must equal (high + low) / 2."""
        result = fibonacci_retracement(HIGH, LOW)
        for h, lo, f500 in zip(
            HIGH.to_list(), LOW.to_list(), result["fib_500"].to_list(), strict=True
        ):
            assert f500 == pytest.approx((h + lo) / 2.0, rel=1e-9)

    def test_known_values_for_simple_range(self) -> None:
        """high=110, low=100 → range=10; verify each level analytically."""
        high = pl.Series([110.0])
        low = pl.Series([100.0])
        result = fibonacci_retracement(high, low)
        assert result["fib_0"][0] == pytest.approx(110.0)
        assert result["fib_236"][0] == pytest.approx(110.0 - 0.236 * 10.0)
        assert result["fib_382"][0] == pytest.approx(110.0 - 0.382 * 10.0)
        assert result["fib_500"][0] == pytest.approx(105.0)
        assert result["fib_618"][0] == pytest.approx(110.0 - 0.618 * 10.0)
        assert result["fib_786"][0] == pytest.approx(110.0 - 0.786 * 10.0)
        assert result["fib_100"][0] == pytest.approx(100.0)

    def test_levels_are_monotonically_decreasing(self) -> None:
        """fib_0 > fib_236 > fib_382 > fib_500 > fib_618 > fib_786 > fib_100."""
        high = pl.Series([120.0])
        low = pl.Series([100.0])
        r = fibonacci_retracement(high, low)
        levels = [
            r[col][0]
            for col in ["fib_0", "fib_236", "fib_382", "fib_500", "fib_618", "fib_786", "fib_100"]
        ]
        for i in range(len(levels) - 1):
            assert levels[i] > levels[i + 1]

    def test_no_nulls_with_scalar_inputs(self) -> None:
        """With non-null high/low, every output column is non-null."""
        result = fibonacci_retracement(HIGH, LOW)
        for col in result.columns:
            assert result[col].null_count() == 0
