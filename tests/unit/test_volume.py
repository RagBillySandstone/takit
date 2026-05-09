"""Unit tests for takit.volume."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from takit.volume import obv, vwap, vwap_bands


def _make_ohlcv(n: int = 10) -> pl.DataFrame:
    """Build a minimal OHLC+volume DataFrame for testing."""
    return pl.DataFrame({
        "high":   [float(i + 1) for i in range(n)],
        "low":    [float(i) for i in range(n)],
        "close":  [float(i) + 0.5 for i in range(n)],
        "volume": [100] * n,
    })


class TestVWAP:
    def test_output_length_matches_input(self) -> None:
        df = _make_ohlcv(10)
        result = vwap(df)
        assert len(result) == 10

    def test_single_bar_equals_typical_price(self) -> None:
        df = pl.DataFrame({"high": [12.0], "low": [10.0], "close": [11.0], "volume": [100]})
        result = vwap(df)
        # typical_price = (12 + 10 + 11) / 3 = 11.0; VWAP of one bar = 11.0
        assert result[0] == pytest.approx(11.0)

    def test_equal_volume_equals_average_tp(self) -> None:
        # With equal volume each bar, VWAP = cumulative mean of typical price.
        df = pl.DataFrame({
            "high":   [11.0, 12.0],
            "low":    [ 9.0, 10.0],
            "close":  [10.0, 11.0],
            "volume": [100,  100],
        })
        result = vwap(df)
        tp0 = (11.0 + 9.0 + 10.0) / 3.0   # 10.0
        tp1 = (12.0 + 10.0 + 11.0) / 3.0  # 11.0
        assert result[0] == pytest.approx(tp0)
        assert result[1] == pytest.approx((tp0 + tp1) / 2.0)

    def test_session_reset_with_time_column(self) -> None:
        # Two sessions of 3 bars each, reset at hour 22.
        times = [
            datetime(2024, 1, 1, 22, 0, tzinfo=UTC),  # session 1 start
            datetime(2024, 1, 1, 23, 0, tzinfo=UTC),
            datetime(2024, 1, 2,  0, 0, tzinfo=UTC),
            datetime(2024, 1, 2, 22, 0, tzinfo=UTC),  # session 2 start
            datetime(2024, 1, 2, 23, 0, tzinfo=UTC),
            datetime(2024, 1, 3,  0, 0, tzinfo=UTC),
        ]
        df = pl.DataFrame({
            "time":   pl.Series(times).cast(pl.Datetime("us", "UTC")),
            "high":   [11.0] * 6,
            "low":    [ 9.0] * 6,
            "close":  [10.0] * 6,
            "volume": [100]  * 6,
        })
        result = vwap(df, session_start_hour=22)
        # All typical prices are 10.0, so VWAP is always 10.0.
        for v in result.to_list():
            assert v == pytest.approx(10.0)
        # After reset, bar 3 should restart accumulation — same value here,
        # but we verify no NaN leaks through from the prior session.
        assert not any(v != v for v in result.to_list())  # NaN check


class TestVWAPBands:
    def test_output_columns(self) -> None:
        df = pl.DataFrame(
            {"high": [11.0, 12.0], "low": [9.0, 10.0], "close": [10.0, 11.0], "volume": [100, 100]}
        )
        result = vwap_bands(df)
        assert set(result.columns) == {"vwap", "upper_1", "lower_1", "upper_2", "lower_2"}
        assert len(result) == 2

    def test_single_bar_zero_std(self) -> None:
        # With one bar, variance = 0, so all bands equal vwap.
        df = pl.DataFrame({"high": [12.0], "low": [10.0], "close": [11.0], "volume": [100]})
        result = vwap_bands(df)
        vwap_val = result["vwap"][0]
        assert result["upper_1"][0] == pytest.approx(vwap_val)
        assert result["lower_1"][0] == pytest.approx(vwap_val)

    def test_bands_symmetric_around_vwap(self) -> None:
        # upper and lower bands must be equidistant from VWAP.
        df = pl.DataFrame(
            {"high": [11.0, 12.0, 13.0], "low": [9.0, 10.0, 11.0], "close": [10.0, 11.0, 12.0], "volume": [100, 200, 150]}
        )
        result = vwap_bands(df)
        for i in range(len(result)):
            vwap_val = result["vwap"][i]
            assert result["upper_1"][i] - vwap_val == pytest.approx(vwap_val - result["lower_1"][i], abs=1e-10)
            assert result["upper_2"][i] - vwap_val == pytest.approx(vwap_val - result["lower_2"][i], abs=1e-10)

    def test_upper_2_wider_than_upper_1(self) -> None:
        # 2σ band must always be at least as wide as the 1σ band.
        df = pl.DataFrame(
            {"high": [11.0, 12.0, 13.0], "low": [9.0, 10.0, 11.0], "close": [10.0, 11.0, 12.0], "volume": [100, 200, 150]}
        )
        result = vwap_bands(df)
        for i in range(len(result)):
            assert result["upper_2"][i] >= result["upper_1"][i]
            assert result["lower_2"][i] <= result["lower_1"][i]


class TestOBV:
    def test_rising_closes_accumulate_volume(self) -> None:
        # Each bar closes higher: OBV increases by volume each bar.
        df = pl.DataFrame({"close": [10.0, 11.0, 12.0], "volume": [100, 200, 300]})
        result = obv(df)
        # Bar 0: 0 (no prior close). Bar 1: +200. Bar 2: +300.
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(200.0)
        assert result[2] == pytest.approx(500.0)

    def test_falling_closes_subtract_volume(self) -> None:
        df = pl.DataFrame({"close": [12.0, 11.0, 10.0], "volume": [100, 200, 300]})
        result = obv(df)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(-200.0)
        assert result[2] == pytest.approx(-500.0)

    def test_unchanged_close_does_not_change_obv(self) -> None:
        df = pl.DataFrame({"close": [10.0, 10.0, 10.0], "volume": [100, 200, 300]})
        result = obv(df)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)

    def test_output_length_matches_input(self) -> None:
        df = pl.DataFrame({"close": [float(i) for i in range(20)], "volume": [100] * 20})
        result = obv(df)
        assert len(result) == 20
        assert result.name == "obv"
