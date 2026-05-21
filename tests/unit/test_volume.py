"""Unit tests for polarticks.volume."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import polars as pl
import pytest

from polarticks.volume import obv, vwap, vwap_bands


def _make_ohlcv(n: int = 10) -> pl.DataFrame:
    """Build a minimal OHLC+volume DataFrame for testing."""
    return pl.DataFrame(
        {
            "high": [float(i + 1) for i in range(n)],
            "low": [float(i) for i in range(n)],
            "close": [float(i) + 0.5 for i in range(n)],
            "volume": [100] * n,
        }
    )


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
        df = pl.DataFrame(
            {
                "high": [11.0, 12.0],
                "low": [9.0, 10.0],
                "close": [10.0, 11.0],
                "volume": [100, 100],
            }
        )
        result = vwap(df)
        tp0 = (11.0 + 9.0 + 10.0) / 3.0  # 10.0
        tp1 = (12.0 + 10.0 + 11.0) / 3.0  # 11.0
        assert result[0] == pytest.approx(tp0)
        assert result[1] == pytest.approx((tp0 + tp1) / 2.0)

    def test_no_multi_reset_within_session_start_hour(self) -> None:
        """VWAP must not reset on every M1 bar that falls within session_start_hour.

        On M1/M5 data the session_start_hour contains many bars (e.g. 60 for M1).
        Only the *first* bar entering that hour should trigger a session reset.
        """
        times = [
            datetime(2024, 1, 1, 22, 0, tzinfo=UTC),  # session start
            datetime(2024, 1, 1, 22, 1, tzinfo=UTC),  # same hour — NOT a new session
            datetime(2024, 1, 1, 22, 2, tzinfo=UTC),
            datetime(2024, 1, 1, 22, 3, tzinfo=UTC),
        ]
        df = pl.DataFrame(
            {
                "time": pl.Series(times).cast(pl.Datetime("us", "UTC")),
                "high": [101.0, 103.0, 105.0, 107.0],
                "low": [99.0, 101.0, 103.0, 105.0],
                "close": [100.0, 102.0, 104.0, 106.0],
                "volume": [1000] * 4,
            }
        )
        result = vwap(df, session_start_hour=22)
        # All 4 bars are in the same session; equal volumes → VWAP = mean of TPs.
        tp_values = [100.0, 102.0, 104.0, 106.0]  # (h+l+c)/3 with h=c+1, l=c-1
        expected_final = sum(tp_values) / 4.0
        assert result[3] == pytest.approx(expected_final), (
            f"VWAP reset mid-session: got {result[3]:.4f}, expected {expected_final:.4f}"
        )
        # Accumulation check: each bar must differ from the previous.
        for i in range(1, 4):
            assert result[i] != pytest.approx(result[i - 1]), f"VWAP did not accumulate at bar {i}"

    def test_session_reset_with_time_column(self) -> None:
        # Two sessions of 3 bars each, reset at hour 22.
        times = [
            datetime(2024, 1, 1, 22, 0, tzinfo=UTC),  # session 1 start
            datetime(2024, 1, 1, 23, 0, tzinfo=UTC),
            datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 2, 22, 0, tzinfo=UTC),  # session 2 start
            datetime(2024, 1, 2, 23, 0, tzinfo=UTC),
            datetime(2024, 1, 3, 0, 0, tzinfo=UTC),
        ]
        df = pl.DataFrame(
            {
                "time": pl.Series(times).cast(pl.Datetime("us", "UTC")),
                "high": [11.0] * 6,
                "low": [9.0] * 6,
                "close": [10.0] * 6,
                "volume": [100] * 6,
            }
        )
        result = vwap(df, session_start_hour=22)
        # All typical prices are 10.0, so VWAP is always 10.0.
        for v in result.to_list():
            assert v == pytest.approx(10.0)
        # After reset, bar 3 should restart accumulation — same value here,
        # but we verify no NaN leaks through from the prior session.
        assert not any(math.isnan(v) for v in result.to_list())


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
            {
                "high": [11.0, 12.0, 13.0],
                "low": [9.0, 10.0, 11.0],
                "close": [10.0, 11.0, 12.0],
                "volume": [100, 200, 150],
            }
        )
        result = vwap_bands(df)
        for i in range(len(result)):
            vwap_val = result["vwap"][i]
            assert result["upper_1"][i] - vwap_val == pytest.approx(
                vwap_val - result["lower_1"][i], abs=1e-10
            )
            assert result["upper_2"][i] - vwap_val == pytest.approx(
                vwap_val - result["lower_2"][i], abs=1e-10
            )

    def test_upper_2_wider_than_upper_1(self) -> None:
        # 2σ band must always be at least as wide as the 1σ band.
        df = pl.DataFrame(
            {
                "high": [11.0, 12.0, 13.0],
                "low": [9.0, 10.0, 11.0],
                "close": [10.0, 11.0, 12.0],
                "volume": [100, 200, 150],
            }
        )
        result = vwap_bands(df)
        for i in range(len(result)):
            assert result["upper_2"][i] >= result["upper_1"][i]
            assert result["lower_2"][i] <= result["lower_1"][i]


class TestVWAPBandsWithTimeColumn:
    """Tests for the session-loop path of vwap_bands (time column present).

    Without a time column the function takes a vectorised shortcut (lines 130-153).
    These tests exercise the row-by-row accumulation path (lines 155-195) that
    is only reached when ohlc_vol contains a ``time`` column.
    """

    def _make_timed_df(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[int],
        session_start_hour: int = 22,
        offset_hours: list[int] | None = None,
    ) -> pl.DataFrame:
        """Build a DataFrame with a UTC Datetime ``time`` column for vwap_bands."""
        from datetime import UTC, datetime

        if offset_hours is None:
            # Default: first bar is session start, subsequent bars are +1 h each.
            offset_hours = [session_start_hour + i for i in range(len(highs))]

        times = [datetime(2024, 1, 1, h % 24, 0, tzinfo=UTC) for h in offset_hours]
        return pl.DataFrame(
            {
                "time": pl.Series(times).cast(pl.Datetime("us", "UTC")),
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )

    def test_output_columns_and_length_with_time(self) -> None:
        """Session-loop path returns the same five columns as the vectorised path."""
        df = self._make_timed_df(
            [11.0, 12.0, 11.0],
            [9.0, 10.0, 9.0],
            [10.0, 11.0, 10.0],
            [100, 100, 100],
        )
        result = vwap_bands(df, session_start_hour=22)
        assert set(result.columns) == {"vwap", "upper_1", "lower_1", "upper_2", "lower_2"}
        assert len(result) == 3

    def test_no_multi_reset_within_session_start_hour(self) -> None:
        """vwap_bands must not reset on every M1 bar in session_start_hour."""
        df = self._make_timed_df(
            highs=[101.0, 103.0, 105.0, 107.0],
            lows=[99.0, 101.0, 103.0, 105.0],
            closes=[100.0, 102.0, 104.0, 106.0],
            volumes=[1000, 1000, 1000, 1000],
            session_start_hour=22,
            # All 4 bars fall within the 22:xx hour (M1 granularity).
            offset_hours=[22, 22, 22, 22],
        )
        result = vwap_bands(df, session_start_hour=22)
        # All bars are one session; VWAP at bar 3 must be the 4-bar average.
        tp_values = [100.0, 102.0, 104.0, 106.0]
        expected_final = sum(tp_values) / 4.0
        assert result["vwap"][3] == pytest.approx(expected_final), (
            f"vwap_bands reset mid-session: got {result['vwap'][3]:.4f}, "
            f"expected {expected_final:.4f}"
        )

    def test_session_reset_restarts_vwap_to_typical_price(self) -> None:
        """On the first bar of a new session VWAP equals that bar's typical price."""
        df = self._make_timed_df(
            highs=[12.0, 13.0, 20.0],
            lows=[10.0, 11.0, 18.0],
            closes=[11.0, 12.0, 19.0],
            volumes=[100, 100, 100],
            session_start_hour=22,
            # bar 0: session start (hour 22), bar 1: hour 23, bar 2: new session (hour 22)
            offset_hours=[22, 23, 22 + 24],
        )
        result = vwap_bands(df, session_start_hour=22)
        # After reset at bar 2: only that bar is in the new session.
        expected_tp = (20.0 + 18.0 + 19.0) / 3.0
        assert result["vwap"][2] == pytest.approx(expected_tp)
        # Single bar in session → σ = 0 → all bands equal VWAP.
        assert result["upper_1"][2] == pytest.approx(expected_tp)
        assert result["lower_1"][2] == pytest.approx(expected_tp)

    def test_bands_symmetric_with_time_column(self) -> None:
        """Upper and lower bands are equidistant from VWAP on the session-loop path."""
        df = self._make_timed_df(
            [11.0, 12.0, 13.0],
            [9.0, 10.0, 11.0],
            [10.0, 11.0, 12.0],
            [100, 200, 150],
        )
        result = vwap_bands(df, session_start_hour=22)
        for i in range(len(result)):
            vwap_val = result["vwap"][i]
            assert result["upper_1"][i] - vwap_val == pytest.approx(
                vwap_val - result["lower_1"][i], abs=1e-10
            )
            assert result["upper_2"][i] - vwap_val == pytest.approx(
                vwap_val - result["lower_2"][i], abs=1e-10
            )

    def test_zero_volume_bar_produces_nan(self) -> None:
        """A session-start bar with zero cumulative volume fills NaN (not a crash)."""
        df = self._make_timed_df(
            highs=[11.0, 12.0],
            lows=[9.0, 10.0],
            closes=[10.0, 11.0],
            volumes=[0, 100],  # first bar has no volume → cum_vol = 0
        )
        result = vwap_bands(df, session_start_hour=22)
        # Bar 0 has zero volume: VWAP and bands are NaN, not an error.
        assert math.isnan(result["vwap"][0])
        assert math.isnan(result["upper_1"][0])
        # Bar 1 accumulates volume normally.
        assert not math.isnan(result["vwap"][1])

    def test_multi_session_no_nan_leakage(self) -> None:
        """VWAP values after a session reset must not carry NaN from the prior session."""
        df = self._make_timed_df(
            highs=[11.0] * 4,
            lows=[9.0] * 4,
            closes=[10.0] * 4,
            volumes=[100] * 4,
            session_start_hour=22,
            offset_hours=[22, 23, 22 + 24, 23 + 24],
        )
        result = vwap_bands(df, session_start_hour=22)
        for val in result["vwap"].to_list():
            assert not math.isnan(val)


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
