"""Unit tests for takit.patterns."""

from __future__ import annotations

import polars as pl

from takit.patterns import (
    is_bullish_engulfing,
    is_bearish_engulfing,
    is_pin_bar_bullish,
    is_pin_bar_bearish,
    is_inside_bar,
    is_doji,
)


def _ohlc(open_: list[float], high: list[float], low: list[float], close: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})


class TestBullishEngulfing:
    def test_detects_pattern(self) -> None:
        # Bar 0: bearish (open 5, close 4). Bar 1: bullish, engulfs (open 3.5, close 5.5).
        df = _ohlc([5.0, 3.5], [5.5, 6.0], [3.5, 3.0], [4.0, 5.5])
        result = is_bullish_engulfing(df)
        assert result[0] is False or result[0] == False  # first bar always False
        assert result[1] == True

    def test_first_bar_always_false(self) -> None:
        df = _ohlc([5.0], [6.0], [4.0], [4.5])
        result = is_bullish_engulfing(df)
        assert result[0] == False

    def test_no_pattern_when_same_direction(self) -> None:
        # Both bars bullish — no engulfing.
        df = _ohlc([4.0, 3.5], [5.0, 5.5], [3.5, 3.0], [4.5, 5.0])
        result = is_bullish_engulfing(df)
        assert result[1] == False


class TestBearishEngulfing:
    def test_detects_pattern(self) -> None:
        # Bar 0: bullish. Bar 1: bearish, engulfs.
        df = _ohlc([4.0, 5.5], [5.0, 6.0], [3.5, 3.0], [4.5, 3.5])
        result = is_bearish_engulfing(df)
        assert result[1] == True

    def test_first_bar_always_false(self) -> None:
        df = _ohlc([4.0], [5.0], [3.5], [4.5])
        result = is_bearish_engulfing(df)
        assert result[0] == False


class TestPinBarBullish:
    def test_detects_hammer(self) -> None:
        # Classic hammer: open and close near top, long lower wick.
        # Range = 2.0. Body = 0.1 (5%). Lower wick = 1.7 (85%).
        df = _ohlc([1.8], [2.0], [0.0], [1.9])
        result = is_pin_bar_bullish(df, wick_ratio=0.6, body_ratio=0.25)
        assert result[0] == True

    def test_rejects_large_body(self) -> None:
        # Body too large to be a pin bar.
        df = _ohlc([1.0], [2.0], [0.0], [1.9])
        result = is_pin_bar_bullish(df, wick_ratio=0.6, body_ratio=0.25)
        assert result[0] == False


class TestPinBarBearish:
    def test_detects_shooting_star(self) -> None:
        # Shooting star: open and close near bottom, long upper wick.
        # Range = 2.0. Body = 0.1. Upper wick = 1.7.
        df = _ohlc([0.1], [2.0], [0.0], [0.2])
        result = is_pin_bar_bearish(df, wick_ratio=0.6, body_ratio=0.25)
        assert result[0] == True


class TestInsideBar:
    def test_detects_inside_bar(self) -> None:
        # Bar 1 high < bar 0 high AND bar 1 low > bar 0 low.
        df = _ohlc([10.0, 10.2], [12.0, 11.5], [8.0, 8.5], [11.0, 10.5])
        result = is_inside_bar(df)
        assert result[0] == False
        assert result[1] == True

    def test_rejects_when_breaks_high(self) -> None:
        df = _ohlc([10.0, 10.2], [12.0, 12.5], [8.0, 8.5], [11.0, 11.5])
        result = is_inside_bar(df)
        assert result[1] == False


class TestDoji:
    def test_detects_doji(self) -> None:
        # Body = 0.05, range = 2.0 → body/range = 0.025 < 0.1.
        df = _ohlc([10.0], [11.0], [9.0], [10.05])
        result = is_doji(df, threshold=0.1)
        assert result[0] == True

    def test_rejects_large_body(self) -> None:
        # Body = 1.0, range = 2.0 → body/range = 0.5 > 0.1.
        df = _ohlc([10.0], [11.0], [9.0], [11.0])
        result = is_doji(df, threshold=0.1)
        assert result[0] == False
