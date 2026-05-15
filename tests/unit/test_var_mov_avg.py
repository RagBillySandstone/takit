"""Unit tests for polarticks.moving_averages.var_mov_avg."""

from __future__ import annotations

import polars as pl
import pytest

from polarticks.moving_averages import var_mov_avg

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# 60-bar synthetic close series: steady uptrend with mild chop.
CLOSE = pl.Series(
    "close",
    [
        100.0,
        101.0,
        100.5,
        102.0,
        101.5,
        103.0,
        102.5,
        104.0,
        103.5,
        105.0,
        104.5,
        106.0,
        105.5,
        107.0,
        106.5,
        108.0,
        107.5,
        109.0,
        108.5,
        110.0,
        109.5,
        111.0,
        110.5,
        112.0,
        111.5,
        113.0,
        112.5,
        114.0,
        113.5,
        115.0,
        114.5,
        116.0,
        115.5,
        117.0,
        116.5,
        118.0,
        117.5,
        119.0,
        118.5,
        120.0,
        119.5,
        121.0,
        120.5,
        122.0,
        121.5,
        123.0,
        122.5,
        124.0,
        123.5,
        125.0,
        124.5,
        126.0,
        125.5,
        127.0,
        126.5,
        128.0,
        127.5,
        129.0,
        128.5,
        130.0,
    ],
)


# ---------------------------------------------------------------------------
# TestVarMovAvg
# ---------------------------------------------------------------------------


class TestVarMovAvg:
    """Tests for the Variable Moving Average indicator."""

    def test_output_length_matches_input(self) -> None:
        """Output length must equal the input length."""
        result = var_mov_avg(CLOSE, period=10)
        assert len(result) == len(CLOSE)

    def test_null_prefix_count(self) -> None:
        """First period+1 values must be null; index period+1 is the first valid bar."""
        period = 10
        result = var_mov_avg(CLOSE, period=period)
        # Indices 0 through period (inclusive) are null — period+1 nulls total.
        for idx in range(period + 1):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[period + 1] is not None, "Expected first valid value at period+1"

    def test_no_accidental_zeros_in_null_prefix(self) -> None:
        """Null-prefix positions must not hold 0.0 sentinel values."""
        period = 10
        result = var_mov_avg(CLOSE, period=period)
        prefix = result.head(period + 1).to_list()
        assert not any(v == 0.0 for v in prefix)

    def test_alias_includes_period(self) -> None:
        """Output series name must embed the period parameter."""
        result = var_mov_avg(CLOSE, period=10)
        assert result.name == "var_mov_avg_10"

    def test_flat_series_stays_at_seed(self) -> None:
        """On a perfectly flat series VMA should equal the seed price at every bar."""
        flat = pl.Series("close", [50.0] * 30)
        result = var_mov_avg(flat, period=5)
        valid = [v for v in result.to_list() if v is not None]
        # ER is 0/epsilon ≈ 0 → SSC ≈ slow_sc → each step adds slow_sc*(50-50)=0.
        assert all(v == pytest.approx(50.0) for v in valid)

    def test_trending_series_tracks_upward(self) -> None:
        """On a steadily rising series VMA should be monotonically increasing."""
        rising = pl.Series("close", [float(i) for i in range(1, 61)])
        result = var_mov_avg(rising, period=5)
        valid = [v for v in result.to_list() if v is not None]
        for i in range(1, len(valid)):
            assert valid[i] >= valid[i - 1], f"VMA decreased at valid index {i}"

    def test_values_near_price_on_strong_trend(self) -> None:
        """VMA should not drift far from price on a strong trend.

        Uses nslow=1 (slow_sc=1.0) and nfast=2 (fast_sc≈0.667) so the SSC
        in a trending market is ~0.667 — large enough to keep lag small.
        """
        rising = pl.Series("close", [float(i) for i in range(1, 61)])
        result = var_mov_avg(rising, period=5, nfast=2, nslow=1)
        valid_vma = [v for v in result.to_list() if v is not None]
        valid_price = rising.to_list()[6:]  # first valid at period+1=6
        for vma_val, price_val in zip(valid_vma, valid_price, strict=True):
            assert abs(vma_val - price_val) < 3.0, (
                f"VMA {vma_val} drifted far from price {price_val}"
            )

    def test_period_1(self) -> None:
        """period=1 produces nulls only at index 0 and 1; first valid at index 2."""
        result = var_mov_avg(CLOSE, period=1)
        assert result[0] is None
        assert result[1] is None
        assert result[2] is not None

    def test_invalid_period_raises(self) -> None:
        """period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            var_mov_avg(CLOSE, period=0)

    def test_invalid_nfast_raises(self) -> None:
        """nfast < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            var_mov_avg(CLOSE, period=5, nfast=0)

    def test_invalid_nslow_raises(self) -> None:
        """nslow < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            var_mov_avg(CLOSE, period=5, nslow=0)

    def test_series_too_short_returns_all_null(self) -> None:
        """When period+1 >= len(series), all output values are null."""
        short = pl.Series("close", [1.0, 2.0, 3.0])
        result = var_mov_avg(short, period=10)
        assert all(v is None for v in result.to_list())

    def test_leading_null_seed_returns_all_null(self) -> None:
        """When price at the seed index (period) is null, output is all null."""
        # First period+1 values are null, so raw[period] is null.
        s = pl.Series([None] * 11 + [5.0] * 20, dtype=pl.Float64)
        result = var_mov_avg(s, period=10)
        assert all(v is None for v in result.to_list())

    def test_g_exponent_affects_output(self) -> None:
        """Different values of g should produce different VMA series."""
        result_g1 = var_mov_avg(CLOSE, period=10, g=1.0)
        result_g2 = var_mov_avg(CLOSE, period=10, g=2.0)
        valid_g1 = [v for v in result_g1.to_list() if v is not None]
        valid_g2 = [v for v in result_g2.to_list() if v is not None]
        # With g=2 the SSC^2 compresses small values more → different trajectory.
        assert valid_g1 != pytest.approx(valid_g2)

    def test_manual_calculation_first_valid_bar(self) -> None:
        """Verify the first valid output matches a hand-computed reference.

        Using price=[1,2,3,4,5,6], period=3, nfast=2, nslow=1, g=1:
            slow_sc = 2/(1+1) = 1.0
            fast_sc = 2/(2+1) = 2/3
            dsc     = 2/3 - 1 = -1/3
            Seed    : AMA0 = price[3] = 4.0
            At idx=4:
                noise = |4-3| + |3-2| + |2-1| = 3.0  (+1e-9)
                signal = |5 - 2| = 3.0
                ER  ≈ 1.0
                SSC = 1.0 * (-1/3) + 1.0 = 2/3
                VMA = 4.0 + (2/3)*(5-4) = 4.0 + 2/3 ≈ 4.6667
        """
        prices = pl.Series("close", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = var_mov_avg(prices, period=3, nfast=2, nslow=1, g=1.0)

        slow_sc = 2.0 / (1 + 1)
        fast_sc = 2.0 / (2 + 1)
        dsc = fast_sc - slow_sc

        ama0 = 4.0  # price[period=3]
        # noise[4] = rolling sum of |p[i]-p[i-1]| over indices 2..4 = 1+1+1 = 3
        noise_4 = 3.0 + 1e-9
        signal_4 = abs(5.0 - 2.0)
        er_4 = signal_4 / noise_4
        ssc_4 = er_4 * dsc + slow_sc
        expected_vma4 = ama0 + ssc_4**1.0 * (5.0 - ama0)

        assert result[4] == pytest.approx(expected_vma4, rel=1e-6)
