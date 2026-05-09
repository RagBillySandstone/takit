"""Unit tests for takit.levels."""

from __future__ import annotations

import pytest
import polars as pl

from takit.levels import pivot_points_floor, pivot_points_camarilla


# Single-bar prior session for scalar pivot calculation.
PREV_HIGH  = pl.Series([12.0])
PREV_LOW   = pl.Series([10.0])
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
        assert (
            result["cam_r1"][0]
            < result["cam_r2"][0]
            < result["cam_r3"][0]
            < result["cam_r4"][0]
        )

    def test_support_levels_descending(self) -> None:
        result = pivot_points_camarilla(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        assert (
            result["cam_s1"][0]
            > result["cam_s2"][0]
            > result["cam_s3"][0]
            > result["cam_s4"][0]
        )

    def test_symmetry_around_close(self) -> None:
        # R1 and S1 should be equidistant from prev_close.
        result = pivot_points_camarilla(PREV_HIGH, PREV_LOW, PREV_CLOSE)
        r1_dist = result["cam_r1"][0] - PREV_CLOSE[0]
        s1_dist = PREV_CLOSE[0] - result["cam_s1"][0]
        assert r1_dist == pytest.approx(s1_dist)
