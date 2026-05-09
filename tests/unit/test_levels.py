"""Unit tests for takit.levels."""

from __future__ import annotations

import polars as pl
import pytest

from takit.levels import (
    pivot_points_camarilla,
    pivot_points_demark,
    pivot_points_fibonacci,
    pivot_points_floor,
    pivot_points_woodie,
)

# Single-bar prior session for scalar pivot calculation.
PREV_HIGH = pl.Series([12.0])
PREV_LOW = pl.Series([10.0])
PREV_CLOSE = pl.Series([11.0])


class TestPivotPointsFloor:
    def test_returns_correct_columns(self) -> None:
        result = pivot_points_floor(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert set(result.columns) == {"pp", "r1", "r2", "r3", "s1", "s2", "s3"}

    def test_pp_formula(self) -> None:
        result = pivot_points_floor(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        expected_pp = (12.0 + 10.0 + 11.0) / 3.0
        assert result["pp"][0] == pytest.approx(expected_pp)

    def test_resistance_levels_ascending(self) -> None:
        result = pivot_points_floor(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert result["r1"][0] < result["r2"][0] < result["r3"][0]

    def test_support_levels_descending(self) -> None:
        result = pivot_points_floor(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert result["s1"][0] > result["s2"][0] > result["s3"][0]

    def test_pp_between_s1_and_r1(self) -> None:
        result = pivot_points_floor(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert result["s1"][0] < result["pp"][0] < result["r1"][0]


class TestPivotPointsCalmarilla:
    def test_returns_correct_columns(self) -> None:
        result = pivot_points_camarilla(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        expected = {"cam_r1", "cam_r2", "cam_r3", "cam_r4", "cam_s1", "cam_s2", "cam_s3", "cam_s4"}
        assert set(result.columns) == expected

    def test_resistance_levels_ascending(self) -> None:
        result = pivot_points_camarilla(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert result["cam_r1"][0] < result["cam_r2"][0] < result["cam_r3"][0] < result["cam_r4"][0]

    def test_support_levels_descending(self) -> None:
        result = pivot_points_camarilla(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert result["cam_s1"][0] > result["cam_s2"][0] > result["cam_s3"][0] > result["cam_s4"][0]

    def test_symmetry_around_close(self) -> None:
        # R1 and S1 should be equidistant from prev_close.
        result = pivot_points_camarilla(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        r1_dist = result["cam_r1"][0] - PREV_CLOSE[0]
        s1_dist = PREV_CLOSE[0] - result["cam_s1"][0]
        assert r1_dist == pytest.approx(s1_dist)


PREV_OPEN = pl.Series([11.0])


class TestFibonacciPivots:
    def test_returns_seven_columns(self) -> None:
        result = pivot_points_fibonacci(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert set(result.columns) == {
            "fib_pp",
            "fib_r1",
            "fib_r2",
            "fib_r3",
            "fib_s1",
            "fib_s2",
            "fib_s3",
        }

    def test_pp_equals_floor_pp(self) -> None:
        # Fibonacci and floor pivots share the same pivot point formula.
        fib = pivot_points_fibonacci(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        floor = pivot_points_floor(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert fib["fib_pp"][0] == pytest.approx(floor["pp"][0])

    def test_r1_uses_382_ratio(self) -> None:
        result = pivot_points_fibonacci(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        hl_range = PREV_HIGH[0] - PREV_LOW[0]
        pp = (PREV_HIGH[0] + PREV_LOW[0] + PREV_CLOSE[0]) / 3.0
        assert result["fib_r1"][0] == pytest.approx(pp + 0.382 * hl_range)

    def test_resistances_increase_r1_r2_r3(self) -> None:
        result = pivot_points_fibonacci(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert result["fib_r1"][0] < result["fib_r2"][0] < result["fib_r3"][0]

    def test_supports_decrease_s1_s2_s3(self) -> None:
        result = pivot_points_fibonacci(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert result["fib_s1"][0] > result["fib_s2"][0] > result["fib_s3"][0]


class TestWoodiePivots:
    def test_returns_five_columns(self) -> None:
        result = pivot_points_woodie(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert set(result.columns) == {"wood_pp", "wood_r1", "wood_r2", "wood_s1", "wood_s2"}

    def test_pp_weights_close_double(self) -> None:
        result = pivot_points_woodie(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        expected_pp = (PREV_HIGH[0] + PREV_LOW[0] + 2.0 * PREV_CLOSE[0]) / 4.0
        assert result["wood_pp"][0] == pytest.approx(expected_pp)

    def test_pp_differs_from_floor_pp_when_close_off_midpoint(self) -> None:
        # When close ≠ (H+L)/2 the two formulas diverge.  Use close=11.8.
        close_off = pl.Series([11.8])
        wood = pivot_points_woodie(PREV_HIGH, PREV_LOW, close_off)
        floor = pivot_points_floor(PREV_HIGH, PREV_LOW, close_off)
        assert wood["wood_pp"][0] != pytest.approx(floor["pp"][0])

    def test_r1_above_pp_s1_below_pp(self) -> None:
        result = pivot_points_woodie(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert result["wood_r1"][0] > result["wood_pp"][0]
        assert result["wood_s1"][0] < result["wood_pp"][0]


class TestDeMarkPivots:
    def test_returns_three_columns(self) -> None:
        result = pivot_points_demark(PREV_OPEN, PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert set(result.columns) == {"dm_pp", "dm_r1", "dm_s1"}

    def test_bearish_session_uses_correct_x(self) -> None:
        # close (11) == open (11), neutral branch: X = H + L + 2C.
        result = pivot_points_demark(PREV_OPEN, PREV_HIGH, PREV_LOW, PREV_CLOSE)
        expected_x = PREV_HIGH[0] + PREV_LOW[0] + 2.0 * PREV_CLOSE[0]
        assert result["dm_pp"][0] == pytest.approx(expected_x / 4.0)

    def test_bullish_session_uses_double_high(self) -> None:
        # When close > open, X = 2H + L + C.
        bull_open = pl.Series([10.0])  # close (11) > open (10)
        result = pivot_points_demark(bull_open, PREV_HIGH, PREV_LOW, PREV_CLOSE)
        expected_x = 2.0 * PREV_HIGH[0] + PREV_LOW[0] + PREV_CLOSE[0]
        assert result["dm_pp"][0] == pytest.approx(expected_x / 4.0)

    def test_r1_above_s1(self) -> None:
        result = pivot_points_demark(PREV_OPEN, PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert result["dm_r1"][0] > result["dm_s1"][0]
