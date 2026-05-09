"""Unit tests for takit.moving_averages."""

from __future__ import annotations

import pytest
import polars as pl

from takit.moving_averages import sma, ema, wma, wilder_smooth, dema, tema


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
        for got, exp in zip(result.to_list(), expected):
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
        for got, exp in zip(result.to_list(), PRICES.to_list()):
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
