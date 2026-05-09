"""Unit tests for takit.moving_averages."""

from __future__ import annotations

import polars as pl
import pytest

from takit.moving_averages import (
    dema,
    ema,
    hma,
    mcginley_dynamic,
    sma,
    tema,
    vwma,
    wilder_smooth,
    wma,
)

PRICES = pl.Series("close", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


class TestSMA:
    def test_warm_up_is_null(self) -> None:
        result = sma(PRICES, 3)
        assert result[0] is None
        assert result[1] is None

    def test_first_valid_value(self) -> None:
        result = sma(PRICES, 3)
        # First valid bar is index 2: mean(1, 2, 3) = 2.0
        assert result[2] == pytest.approx(2.0)

    def test_output_length_matches_input(self) -> None:
        assert len(sma(PRICES, 3)) == len(PRICES)

    def test_full_series(self) -> None:
        result = sma(PRICES, 3)
        expected = [None, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        for got, exp in zip(result.to_list(), expected, strict=True):
            if exp is None:
                assert got is None
            else:
                assert got == pytest.approx(exp)

    def test_period_1_returns_original(self) -> None:
        result = sma(PRICES, 1)
        assert result.to_list() == PRICES.to_list()

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            sma(PRICES, 0)


class TestEMA:
    def test_warm_up_is_null(self) -> None:
        result = ema(PRICES, 3)
        assert result[0] is None
        assert result[1] is None

    def test_output_length_matches_input(self) -> None:
        assert len(ema(PRICES, 5)) == len(PRICES)

    def test_first_valid_value_is_not_null(self) -> None:
        # Polars ewm_mean seeds from bar 0 (not from the SMA); the first valid
        # value is simply whatever the recursive formula produces at index period-1.
        result = ema(PRICES, 3)
        assert result[2] is not None

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            ema(PRICES, 0)


class TestWMA:
    def test_warm_up_is_null(self) -> None:
        result = wma(PRICES, 3)
        assert result[0] is None
        assert result[1] is None

    def test_first_valid_value(self) -> None:
        # weights = [1, 2, 3], values = [1, 2, 3], weight_sum = 6
        # WMA = (1*1 + 2*2 + 3*3) / 6 = 14 / 6 ≈ 2.333
        result = wma(PRICES, 3)
        assert result[2] == pytest.approx(14.0 / 6.0)

    def test_output_length_matches_input(self) -> None:
        assert len(wma(PRICES, 3)) == len(PRICES)


class TestWilderSmooth:
    def test_warm_up_is_null(self) -> None:
        result = wilder_smooth(PRICES, 3)
        assert result[0] is None
        assert result[1] is None

    def test_output_length_matches_input(self) -> None:
        assert len(wilder_smooth(PRICES, 3)) == len(PRICES)

    def test_alpha_is_inverse_period(self) -> None:
        # Wilder smooth with period=1 should equal alpha=1.0, which makes each
        # output equal to the input (ewm with α=1 gives the current value).
        result = wilder_smooth(PRICES, 1)
        for got, exp in zip(result.to_list(), PRICES.to_list(), strict=True):
            assert got == pytest.approx(exp)


class TestDEMA:
    def test_warm_up_longer_than_ema(self) -> None:
        # DEMA requires two EMA passes so more warm-up nulls than a single EMA.
        result_dema = dema(PRICES, 3)
        result_ema = ema(PRICES, 3)
        dema_nulls = sum(1 for v in result_dema.to_list() if v is None)
        ema_nulls = sum(1 for v in result_ema.to_list() if v is None)
        assert dema_nulls >= ema_nulls

    def test_output_length_matches_input(self) -> None:
        assert len(dema(PRICES, 3)) == len(PRICES)


class TestTEMA:
    def test_output_length_matches_input(self) -> None:
        assert len(tema(PRICES, 3)) == len(PRICES)

    def test_warm_up_longer_than_dema(self) -> None:
        result_tema = tema(PRICES, 3)
        result_dema = dema(PRICES, 3)
        tema_nulls = sum(1 for v in result_tema.to_list() if v is None)
        dema_nulls = sum(1 for v in result_dema.to_list() if v is None)
        assert tema_nulls >= dema_nulls


class TestHMA:
    def test_output_length_matches_input(self) -> None:
        assert len(hma(PRICES, 4)) == len(PRICES)

    def test_warm_up_is_null(self) -> None:
        # period=4 → half=2, sqrt=2; warm-up = (2-1) + (2-1) = 2 bars
        result = hma(PRICES, 4)
        assert result[0] is None

    def test_valid_values_are_not_null(self) -> None:
        result = hma(PRICES, 4)
        valid = [v for v in result.to_list() if v is not None]
        assert len(valid) > 0

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            hma(PRICES, 1)

    def test_alias_contains_period(self) -> None:
        assert hma(PRICES, 4).name == "hma_4"


class TestVWMA:
    def test_output_length_matches_input(self) -> None:
        volume = pl.Series("vol", [100.0] * 10)
        assert len(vwma(PRICES, volume, 3)) == len(PRICES)

    def test_warm_up_is_null(self) -> None:
        volume = pl.Series("vol", [100.0] * 10)
        result = vwma(PRICES, volume, 3)
        assert result[0] is None
        assert result[1] is None

    def test_equal_volume_matches_sma(self) -> None:
        # When all bars have equal volume, VWMA == SMA.
        volume = pl.Series("vol", [50.0] * 10)
        result_vwma = vwma(PRICES, volume, 3)
        result_sma = sma(PRICES, 3)
        for got, exp in zip(
            result_vwma.drop_nulls().to_list(), result_sma.drop_nulls().to_list(), strict=True
        ):
            assert got == pytest.approx(exp)

    def test_high_volume_bar_pulls_average(self) -> None:
        # Price: [1, 2, 100]; volume: [1, 1, 1000] → VWMA dominated by last bar.
        price = pl.Series([1.0, 2.0, 100.0])
        volume = pl.Series([1.0, 1.0, 1000.0])
        result = vwma(price, volume, 3)
        # VWMA = (1*1 + 2*1 + 100*1000) / (1 + 1 + 1000) ≈ 99.9
        assert result[2] == pytest.approx((1 + 2 + 100_000) / 1002, rel=1e-6)


class TestMcginleyDynamic:
    def test_output_length_matches_input(self) -> None:
        assert len(mcginley_dynamic(PRICES, 3)) == len(PRICES)

    def test_warm_up_is_null(self) -> None:
        result = mcginley_dynamic(PRICES, 3)
        assert result[0] is None
        assert result[1] is None

    def test_seed_value_equals_sma(self) -> None:
        # The first valid value (index period-1) is the SMA seed.
        result = mcginley_dynamic(PRICES, 3)
        expected_seed = (1.0 + 2.0 + 3.0) / 3.0
        assert result[2] == pytest.approx(expected_seed)

    def test_converges_on_flat_series(self) -> None:
        # On a perfectly flat series the McGinley Dynamic stays constant.
        flat = pl.Series([5.0] * 20)
        result = mcginley_dynamic(flat, 5)
        valid = [v for v in result.to_list() if v is not None]
        for value in valid:
            assert value == pytest.approx(5.0, rel=1e-9)

    def test_insufficient_non_null_seed_returns_all_null(self) -> None:
        # seed_window collects non-null values from raw_values[:period].
        # [None, None, 1.0][:3] → only 1 non-null, need 3 → early return all-null.
        sparse = pl.Series([None, None, 1.0, 2.0, 3.0], dtype=pl.Float64)
        result = mcginley_dynamic(sparse, 3)
        assert all(v is None for v in result.to_list())

    def test_null_mid_series_propagates_null(self) -> None:
        # A None value that appears after the warm-up should produce None at that index.
        series_with_gap = pl.Series([1.0, 2.0, 3.0, None, 5.0], dtype=pl.Float64)
        result = mcginley_dynamic(series_with_gap, 3)
        assert result[3] is None

    def test_zero_seed_propagates_null_on_all_subsequent_bars(self) -> None:
        # Seed SMA = 0.0 triggers the `md == 0.0` guard in the main loop,
        # setting output to None and skipping the md update for every later bar.
        series_zeros = pl.Series([0.0, 0.0, 0.0, 1.0, 2.0], dtype=pl.Float64)
        result = mcginley_dynamic(series_zeros, 3)
        # Seed is at index 2 (= 0.0); bars 3 and 4 must be None.
        assert result[2] == pytest.approx(0.0)
        assert result[3] is None
        assert result[4] is None
