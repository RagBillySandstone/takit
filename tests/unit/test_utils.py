"""Unit tests for takit.utils.

Covers crossover, crossunder, log_returns, and simple_returns.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from takit.utils import crossover, crossunder, log_returns, simple_returns


class TestCrossover:
    """Tests for the crossover (bullish cross) detector."""

    def test_basic_bullish_cross_detected(self) -> None:
        """fast transitions from below slow to above slow: True only on that bar."""
        fast = pl.Series([1.0, 1.0, 3.0])
        slow = pl.Series([2.0, 2.0, 2.0])
        result = crossover(fast, slow)
        assert result.to_list() == [False, False, True]

    def test_first_bar_fires_when_fast_starts_above(self) -> None:
        """The null sentinel (-1.0) means bar 0 counts as a cross when fast > slow."""
        fast = pl.Series([3.0, 4.0, 5.0])
        slow = pl.Series([1.0, 1.0, 1.0])
        result = crossover(fast, slow)
        assert result.to_list() == [True, False, False]

    def test_no_signal_when_fast_stays_below(self) -> None:
        fast = pl.Series([1.0, 1.0, 1.0])
        slow = pl.Series([2.0, 2.0, 2.0])
        result = crossover(fast, slow)
        assert result.to_list() == [False, False, False]

    def test_no_double_signal_when_fast_stays_above(self) -> None:
        """Once fast is above slow, only the transition bar fires — not every bar."""
        fast = pl.Series([1.0, 3.0, 4.0])
        slow = pl.Series([2.0, 2.0, 2.0])
        result = crossover(fast, slow)
        # Cross happens at index 1; index 2 must not re-fire.
        assert result.to_list() == [False, True, False]

    def test_atol_suppresses_noise_within_tolerance(self) -> None:
        """diff = 0.0001 is inside atol=0.001 — should not be counted as a cross."""
        fast = pl.Series([0.9, 1.0001])
        slow = pl.Series([1.0, 1.0])
        result = crossover(fast, slow, atol=0.001)
        assert result[1] is False

    def test_atol_allows_genuine_cross_above_tolerance(self) -> None:
        """diff = 0.1 exceeds atol=0.001 — should fire."""
        fast = pl.Series([0.9, 1.1])
        slow = pl.Series([1.0, 1.0])
        result = crossover(fast, slow, atol=0.001)
        assert result[1] is True

    def test_output_series_name(self) -> None:
        result = crossover(pl.Series([2.0]), pl.Series([1.0]))
        assert result.name == "crossover"

    def test_output_length_matches_input(self) -> None:
        fast = pl.Series([float(i) for i in range(10)])
        slow = pl.Series([5.0] * 10)
        assert len(crossover(fast, slow)) == 10

    def test_single_bar_returns_one_element(self) -> None:
        result = crossover(pl.Series([5.0]), pl.Series([3.0]))
        assert len(result) == 1
        assert result[0] is True


class TestCrossunder:
    """Tests for the crossunder (bearish cross) detector."""

    def test_basic_bearish_cross_detected(self) -> None:
        """fast transitions from above slow to below slow: True only on that bar."""
        fast = pl.Series([3.0, 3.0, 1.0])
        slow = pl.Series([2.0, 2.0, 2.0])
        result = crossunder(fast, slow)
        assert result.to_list() == [False, False, True]

    def test_first_bar_fires_when_fast_starts_below(self) -> None:
        """The null sentinel (-1.0) means bar 0 counts as a cross when fast < slow."""
        fast = pl.Series([1.0, 1.0, 1.0])
        slow = pl.Series([2.0, 2.0, 2.0])
        result = crossunder(fast, slow)
        assert result.to_list() == [True, False, False]

    def test_no_signal_when_fast_stays_above(self) -> None:
        fast = pl.Series([3.0, 3.0, 3.0])
        slow = pl.Series([2.0, 2.0, 2.0])
        result = crossunder(fast, slow)
        assert result.to_list() == [False, False, False]

    def test_no_double_signal_when_fast_stays_below(self) -> None:
        """Once fast is below slow, only the transition bar fires."""
        fast = pl.Series([3.0, 1.0, 0.5])
        slow = pl.Series([2.0, 2.0, 2.0])
        result = crossunder(fast, slow)
        assert result.to_list() == [False, True, False]

    def test_atol_suppresses_noise_within_tolerance(self) -> None:
        """diff = 0.0001 is inside atol=0.001 — should not fire."""
        fast = pl.Series([1.0, 0.9999])
        slow = pl.Series([1.0, 1.0])
        result = crossunder(fast, slow, atol=0.001)
        assert result[1] is False

    def test_atol_allows_genuine_cross_above_tolerance(self) -> None:
        fast = pl.Series([1.1, 0.9])
        slow = pl.Series([1.0, 1.0])
        result = crossunder(fast, slow, atol=0.001)
        assert result[1] is True

    def test_output_series_name(self) -> None:
        result = crossunder(pl.Series([1.0]), pl.Series([2.0]))
        assert result.name == "crossunder"

    def test_output_length_matches_input(self) -> None:
        fast = pl.Series([float(i) for i in range(10)])
        slow = pl.Series([5.0] * 10)
        assert len(crossunder(fast, slow)) == 10


class TestLogReturns:
    """Tests for log_returns: ln(price[t] / price[t-1])."""

    def test_first_value_is_null(self) -> None:
        result = log_returns(pl.Series([1.0, 2.0, 4.0]))
        assert result[0] is None

    def test_doubling_gives_ln_two(self) -> None:
        result = log_returns(pl.Series([10.0, 20.0]))
        assert result[1] == pytest.approx(math.log(2.0))

    def test_exponential_growth_gives_unit_returns(self) -> None:
        """Prices on an e^x curve should yield log-return of exactly 1.0 each bar."""
        prices = pl.Series([1.0, math.e, math.e**2, math.e**3])
        result = log_returns(prices)
        for idx in range(1, 4):
            assert result[idx] == pytest.approx(1.0, rel=1e-10)

    def test_flat_series_gives_zero_returns(self) -> None:
        result = log_returns(pl.Series([5.0] * 5))
        for val in result.drop_nulls().to_list():
            assert val == pytest.approx(0.0)

    def test_output_series_name(self) -> None:
        result = log_returns(pl.Series([1.0, 2.0]))
        assert result.name == "log_returns"

    def test_output_length_matches_input(self) -> None:
        result = log_returns(pl.Series([float(i + 1) for i in range(10)]))
        assert len(result) == 10


class TestSimpleReturns:
    """Tests for simple_returns: (price[t] - price[t-1]) / price[t-1]."""

    def test_first_value_is_null(self) -> None:
        result = simple_returns(pl.Series([1.0, 2.0, 4.0]))
        assert result[0] is None

    def test_doubling_gives_one_hundred_percent_return(self) -> None:
        # (2 - 1) / 1 = 1.0
        result = simple_returns(pl.Series([1.0, 2.0]))
        assert result[1] == pytest.approx(1.0)

    def test_halving_gives_minus_fifty_percent(self) -> None:
        result = simple_returns(pl.Series([100.0, 50.0]))
        assert result[1] == pytest.approx(-0.5)

    def test_flat_series_gives_zero_returns(self) -> None:
        result = simple_returns(pl.Series([7.0] * 5))
        for val in result.drop_nulls().to_list():
            assert val == pytest.approx(0.0)

    def test_multiple_bars_correct_values(self) -> None:
        # (2-1)/1=1.0, (4-2)/2=1.0
        result = simple_returns(pl.Series([1.0, 2.0, 4.0]))
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.0)

    def test_output_series_name(self) -> None:
        result = simple_returns(pl.Series([1.0, 2.0]))
        assert result.name == "simple_returns"

    def test_output_length_matches_input(self) -> None:
        result = simple_returns(pl.Series([float(i + 1) for i in range(10)]))
        assert len(result) == 10
