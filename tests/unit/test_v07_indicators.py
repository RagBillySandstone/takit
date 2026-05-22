"""Unit tests for v0.7.0 indicators.

Covers:
    moving_averages: mama, dominant_cycle_period
    momentum:        dss, vwrsi
    volume:          session_range
    patterns:        heiken_ashi

Each test class verifies:
  - Output length / shape matches input length.
  - Correct number of leading nulls (or zero nulls where documented).
  - First valid value is finite (not nan/inf).
  - Sensible domain constraints where applicable.
  - At least one edge case or error condition.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from polarticks.momentum import dss, vwrsi
from polarticks.moving_averages import dominant_cycle_period, mama
from polarticks.patterns import heiken_ashi
from polarticks.volume import session_range

# ---------------------------------------------------------------------------
# Shared synthetic OHLCV fixture (N = 120 bars)
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
OHLC = OHLCV.select(["open", "high", "low", "close"])

# ---------------------------------------------------------------------------
# MAMA / FAMA
# ---------------------------------------------------------------------------


class TestMAMA:
    """MESA Adaptive Moving Average / Following Adaptive Moving Average."""

    def test_output_shape(self) -> None:
        result = mama(CLOSE)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == _N

    def test_columns_present(self) -> None:
        result = mama(CLOSE)
        assert "mama" in result.columns
        assert "fama" in result.columns

    def test_leading_nulls(self) -> None:
        result = mama(CLOSE)
        # First 6 bars are null (HT kernel warm-up)
        assert result["mama"][:6].null_count() == 6
        assert result["fama"][:6].null_count() == 6

    def test_first_valid_finite(self) -> None:
        result = mama(CLOSE)
        assert math.isfinite(result["mama"][6])
        assert math.isfinite(result["fama"][6])

    def test_mama_tracks_close(self) -> None:
        """MAMA values should be close to the price series (no extreme outliers)."""
        result = mama(CLOSE)
        valid_mama = result["mama"].drop_nulls()
        close_valid = CLOSE.tail(len(valid_mama))
        # MAMA should stay within 2× the price range of the close series
        close_range = CLOSE.max() - CLOSE.min()
        assert (valid_mama - close_valid).abs().max() < 2.0 * close_range

    def test_fama_smoother_than_mama(self) -> None:
        """FAMA should exhibit less bar-to-bar variation than MAMA."""
        result = mama(CLOSE).drop_nulls()
        mama_std = result["mama"].std()
        fama_std = result["fama"].std()
        # FAMA is a half-speed follower → lower or equal variance
        assert fama_std <= mama_std + 1e-6

    def test_default_limits(self) -> None:
        result = mama(CLOSE, fast_limit=0.5, slow_limit=0.05)
        assert result["mama"].drop_nulls().len() > 0

    def test_invalid_limits_raises(self) -> None:
        with pytest.raises(ValueError):
            mama(CLOSE, fast_limit=0.05, slow_limit=0.5)

    def test_equal_limits_raises(self) -> None:
        with pytest.raises(ValueError):
            mama(CLOSE, fast_limit=0.3, slow_limit=0.3)

    def test_custom_limits(self) -> None:
        result = mama(CLOSE, fast_limit=0.8, slow_limit=0.01)
        assert result["mama"].drop_nulls().len() > 0


# ---------------------------------------------------------------------------
# Dominant Cycle Period
# ---------------------------------------------------------------------------


class TestDominantCyclePeriod:
    """Hilbert Transform Dominant Cycle Period."""

    def test_output_length(self) -> None:
        result = dominant_cycle_period(CLOSE)
        assert len(result) == _N

    def test_leading_nulls(self) -> None:
        result = dominant_cycle_period(CLOSE)
        # First 6 bars are null
        assert result[:6].null_count() == 6
        assert result[6] is not None

    def test_first_valid_finite(self) -> None:
        val = dominant_cycle_period(CLOSE)[6]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert dominant_cycle_period(CLOSE).name == "dominant_cycle"

    def test_range_bounded(self) -> None:
        """After convergence, cycle period should stay within [6, 50]."""
        result = dominant_cycle_period(CLOSE).drop_nulls()
        assert result.min() >= 6.0
        assert result.max() <= 50.0

    def test_dtype_float64(self) -> None:
        assert dominant_cycle_period(CLOSE).dtype == pl.Float64


# ---------------------------------------------------------------------------
# Double-smoothed Stochastic (DSS)
# ---------------------------------------------------------------------------


class TestDSS:
    """Bressert Double-smoothed Stochastic."""

    def test_output_shape(self) -> None:
        result = dss(OHLC)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == _N

    def test_columns_present(self) -> None:
        result = dss(OHLC)
        assert "dss" in result.columns
        assert "dss_signal" in result.columns

    def test_leading_nulls(self) -> None:
        period, smooth = 5, 5
        result = dss(OHLC, period=period, smooth=smooth)
        # dss: 2*(period-1) + (smooth-1) nulls
        # dss_signal: 2*(period-1) + 2*(smooth-1) nulls
        expected_dss = 2 * (period - 1) + (smooth - 1)
        expected_sig = 2 * (period - 1) + 2 * (smooth - 1)
        assert result["dss"][:expected_dss].null_count() == expected_dss
        assert result["dss"][expected_dss] is not None
        assert result["dss_signal"][:expected_sig].null_count() == expected_sig
        assert result["dss_signal"][expected_sig] is not None

    def test_first_valid_finite(self) -> None:
        result = dss(OHLC, period=5, smooth=5)
        idx = 2 * (5 - 1) + (5 - 1)
        assert math.isfinite(result["dss"][idx])

    def test_range_0_to_100(self) -> None:
        result = dss(OHLC).drop_nulls()
        assert result["dss"].min() >= 0.0
        assert result["dss"].max() <= 100.0
        assert result["dss_signal"].min() >= 0.0
        assert result["dss_signal"].max() <= 100.0

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            dss(OHLC, period=0)

    def test_invalid_smooth_raises(self) -> None:
        with pytest.raises(ValueError):
            dss(OHLC, smooth=0)

    def test_constant_market_midpoint(self) -> None:
        """Flat market (no H/L range) → DSS should be 50 (neutral fill)."""
        flat = pl.DataFrame(
            {
                "high": [10.0] * 30,
                "low": [10.0] * 30,
                "close": [10.0] * 30,
            }
        )
        result = dss(flat, period=5, smooth=3).drop_nulls()
        assert all(abs(v - 50.0) < 1e-10 for v in result["dss"].to_list())


# ---------------------------------------------------------------------------
# Volume-weighted RSI
# ---------------------------------------------------------------------------


class TestVWRSI:
    """Volume-weighted RSI."""

    def test_output_length(self) -> None:
        assert len(vwrsi(OHLCV)) == _N

    def test_leading_nulls(self) -> None:
        period = 14
        result = vwrsi(OHLCV, period=period)
        assert result[:period].null_count() == period
        assert result[period] is not None

    def test_first_valid_finite(self) -> None:
        val = vwrsi(OHLCV)[14]
        assert math.isfinite(val)

    def test_alias(self) -> None:
        assert vwrsi(OHLCV, period=14).name == "vwrsi_14"

    def test_range_0_to_100(self) -> None:
        result = vwrsi(OHLCV).drop_nulls()
        assert result.min() >= 0.0
        assert result.max() <= 100.0

    def test_all_up_volume_gives_100(self) -> None:
        """When every bar is up, VW-RSI should converge to 100."""
        closes = [100.0 + i for i in range(30)]
        vols = [1000.0] * 30
        df = pl.DataFrame({"close": closes, "volume": vols})
        result = vwrsi(df, period=5).drop_nulls()
        assert all(v == pytest.approx(100.0, abs=1e-6) for v in result.to_list())

    def test_all_down_volume_gives_0(self) -> None:
        """When every bar is down, VW-RSI should converge to 0."""
        closes = [100.0 - i for i in range(30)]
        vols = [1000.0] * 30
        df = pl.DataFrame({"close": closes, "volume": vols})
        result = vwrsi(df, period=5).drop_nulls()
        assert all(v == pytest.approx(0.0, abs=1e-6) for v in result.to_list())

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            vwrsi(OHLCV, period=1)

    def test_dtype_float64(self) -> None:
        assert vwrsi(OHLCV).dtype == pl.Float64


# ---------------------------------------------------------------------------
# Heiken-Ashi
# ---------------------------------------------------------------------------


class TestHeikenAshi:
    """Heiken-Ashi transformation."""

    def test_output_shape(self) -> None:
        result = heiken_ashi(OHLC)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == _N

    def test_columns_present(self) -> None:
        result = heiken_ashi(OHLC)
        assert "ha_open" in result.columns
        assert "ha_high" in result.columns
        assert "ha_low" in result.columns
        assert "ha_close" in result.columns

    def test_no_leading_nulls(self) -> None:
        """All bars including bar 0 should have a value."""
        result = heiken_ashi(OHLC)
        assert result["ha_open"].null_count() == 0
        assert result["ha_close"].null_count() == 0

    def test_ha_close_formula(self) -> None:
        """ha_close[t] = (open + high + low + close) / 4."""
        result = heiken_ashi(OHLC)
        expected = (OHLC["open"] + OHLC["high"] + OHLC["low"] + OHLC["close"]) / 4.0
        for i in range(_N):
            assert result["ha_close"][i] == pytest.approx(expected[i])

    def test_ha_open_bar0_seed(self) -> None:
        """ha_open[0] = (open[0] + close[0]) / 2."""
        result = heiken_ashi(OHLC)
        expected = (OHLC["open"][0] + OHLC["close"][0]) / 2.0
        assert result["ha_open"][0] == pytest.approx(expected)

    def test_ha_high_ge_ha_open_and_ha_close(self) -> None:
        """ha_high must be >= both ha_open and ha_close."""
        result = heiken_ashi(OHLC)
        assert (result["ha_high"] >= result["ha_open"]).all()
        assert (result["ha_high"] >= result["ha_close"]).all()

    def test_ha_low_le_ha_open_and_ha_close(self) -> None:
        """ha_low must be <= both ha_open and ha_close."""
        result = heiken_ashi(OHLC)
        assert (result["ha_low"] <= result["ha_open"]).all()
        assert (result["ha_low"] <= result["ha_close"]).all()

    def test_ha_high_ge_actual_high(self) -> None:
        """ha_high is always >= actual high because we take max(high, ha_open, ha_close)."""
        result = heiken_ashi(OHLC)
        assert (result["ha_high"] >= OHLC["high"]).all()

    def test_ha_low_le_actual_low(self) -> None:
        """ha_low is always <= actual low because we take min(low, ha_open, ha_close)."""
        result = heiken_ashi(OHLC)
        assert (result["ha_low"] <= OHLC["low"]).all()

    def test_single_bar(self) -> None:
        """Single-bar input should produce a valid row with no prior HA state."""
        single = pl.DataFrame(
            {"open": [10.0], "high": [12.0], "low": [8.0], "close": [11.0]}
        )
        result = heiken_ashi(single)
        assert len(result) == 1
        assert result["ha_open"][0] == pytest.approx(10.5)  # (10+11)/2
        assert result["ha_close"][0] == pytest.approx(10.25)  # (10+12+8+11)/4


# ---------------------------------------------------------------------------
# Session Range
# ---------------------------------------------------------------------------


def _make_timed_ohlcv(n_hours: int = 48) -> pl.DataFrame:
    """Build an hourly OHLCV frame spanning n_hours starting at 2024-01-15 00:00 UTC."""
    base = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    times = [base + timedelta(hours=i) for i in range(n_hours)]
    closes = [100.0 + i * 0.1 for i in range(n_hours)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    opens = [c - 0.2 for c in closes]
    vols = [1000.0] * n_hours
    return pl.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
            "time": pl.Series(times).dt.convert_time_zone("UTC"),
        }
    )


class TestSessionRange:
    """Session Range (Asian / London / NY)."""

    def test_output_shape(self) -> None:
        df = _make_timed_ohlcv()
        result = session_range(df)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 48

    def test_columns_present(self) -> None:
        df = _make_timed_ohlcv()
        result = session_range(df)
        assert "asian_high" in result.columns
        assert "asian_low" in result.columns
        assert "london_high" in result.columns
        assert "london_low" in result.columns
        assert "ny_high" in result.columns
        assert "ny_low" in result.columns

    def test_missing_time_column_raises(self) -> None:
        with pytest.raises(ValueError, match="time"):
            session_range(OHLC)

    def test_asian_session_non_null_in_window(self) -> None:
        """Bars at UTC hours 0-8 should have non-null Asian H/L."""
        df = _make_timed_ohlcv()
        result = session_range(df, asian_start=0, asian_end=9)
        hours = df["time"].dt.hour().to_list()
        for i, hr in enumerate(hours):
            in_asian = 0 <= hr < 9
            if in_asian:
                assert result["asian_high"][i] is not None
                assert result["asian_low"][i] is not None

    def test_outside_session_is_null(self) -> None:
        """Bars outside all sessions should have null for each session's columns."""
        df = _make_timed_ohlcv()
        result = session_range(
            df,
            asian_start=0,
            asian_end=9,
            london_start=8,
            london_end=17,
            ny_start=13,
            ny_end=22,
        )
        hours = df["time"].dt.hour().to_list()
        for i, hr in enumerate(hours):
            if not (0 <= hr < 9):
                assert result["asian_high"][i] is None
            if not (8 <= hr < 17):
                assert result["london_high"][i] is None
            if not (13 <= hr < 22):
                assert result["ny_high"][i] is None

    def test_session_high_is_monotone_increasing(self) -> None:
        """Within a session, the running high can only increase."""
        df = _make_timed_ohlcv(24)
        result = session_range(df, asian_start=0, asian_end=9)
        asian_h = result["asian_high"].to_list()
        prev = None
        for val in asian_h:
            if val is not None:
                if prev is not None:
                    assert val >= prev
                prev = val
            else:
                # Session reset: next non-null can be lower
                prev = None

    def test_session_low_is_monotone_decreasing(self) -> None:
        """Within a session, the running low can only decrease or stay the same."""
        df = _make_timed_ohlcv(24)
        result = session_range(df, asian_start=0, asian_end=9)
        asian_l = result["asian_low"].to_list()
        prev = None
        for val in asian_l:
            if val is not None:
                if prev is not None:
                    assert val <= prev
                prev = val
            else:
                prev = None

    def test_session_resets_each_day(self) -> None:
        """Asian session on day 2 should start fresh (lower high than day 1 end)."""
        df = _make_timed_ohlcv(30)  # spans into hour 6 of day 2
        result = session_range(df, asian_start=0, asian_end=9)
        hours = df["time"].dt.hour().to_list()
        # Find last bar of day-1 Asian (hour 8) and first bar of day-2 Asian (hour 24→0)
        first_day2_asian_idx = None
        for i, hr in enumerate(hours):
            if i >= 24 and 0 <= hr < 9:
                first_day2_asian_idx = i
                break
        if first_day2_asian_idx is not None:
            # Day 2 Asian high at bar 0 equals that bar's high exactly (fresh reset).
            bar_high = df["high"][first_day2_asian_idx]
            assert result["asian_high"][first_day2_asian_idx] == pytest.approx(bar_high)

    def test_overnight_session_window(self) -> None:
        """Overnight session (start > end) should mark bars that span midnight."""
        df = _make_timed_ohlcv(48)
        # Asian FX session: 22:00-07:00 (overnight)
        result = session_range(
            df,
            asian_start=22,
            asian_end=7,
            london_start=8,
            london_end=17,
            ny_start=13,
            ny_end=22,
        )
        hours = df["time"].dt.hour().to_list()
        for i, hr in enumerate(hours):
            in_overnight = hr >= 22 or hr < 7
            if in_overnight:
                assert result["asian_high"][i] is not None
            else:
                assert result["asian_high"][i] is None
