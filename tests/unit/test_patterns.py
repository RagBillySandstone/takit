"""Unit tests for takit.patterns."""

from __future__ import annotations

import polars as pl

from takit.patterns import (
    is_bearish_engulfing,
    is_bearish_harami,
    is_bullish_engulfing,
    is_bullish_harami,
    is_doji,
    is_evening_star,
    is_inside_bar,
    is_morning_star,
    is_pin_bar_bearish,
    is_pin_bar_bullish,
    is_three_black_crows,
    is_three_white_soldiers,
)


def _ohlc(
    open_: list[float], high: list[float], low: list[float], close: list[float]
) -> pl.DataFrame:
    return pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})


class TestBullishEngulfing:
    def test_detects_pattern(self) -> None:
        # Bar 0: bearish (open 5, close 4). Bar 1: bullish, engulfs (open 3.5, close 5.5).
        df = _ohlc([5.0, 3.5], [5.5, 6.0], [3.5, 3.0], [4.0, 5.5])
        result = is_bullish_engulfing(df)
        assert result[0] is False or not result[0]  # first bar always False
        assert result[1]

    def test_first_bar_always_false(self) -> None:
        df = _ohlc([5.0], [6.0], [4.0], [4.5])
        result = is_bullish_engulfing(df)
        assert not result[0]

    def test_no_pattern_when_same_direction(self) -> None:
        # Both bars bullish — no engulfing.
        df = _ohlc([4.0, 3.5], [5.0, 5.5], [3.5, 3.0], [4.5, 5.0])
        result = is_bullish_engulfing(df)
        assert not result[1]


class TestBearishEngulfing:
    def test_detects_pattern(self) -> None:
        # Bar 0: bullish. Bar 1: bearish, engulfs.
        df = _ohlc([4.0, 5.5], [5.0, 6.0], [3.5, 3.0], [4.5, 3.5])
        result = is_bearish_engulfing(df)
        assert result[1]

    def test_first_bar_always_false(self) -> None:
        df = _ohlc([4.0], [5.0], [3.5], [4.5])
        result = is_bearish_engulfing(df)
        assert not result[0]


class TestPinBarBullish:
    def test_detects_hammer(self) -> None:
        # Classic hammer: open and close near top, long lower wick.
        # Range = 2.0. Body = 0.1 (5%). Lower wick = 1.7 (85%).
        df = _ohlc([1.8], [2.0], [0.0], [1.9])
        result = is_pin_bar_bullish(df, wick_ratio=0.6, body_ratio=0.25)
        assert result[0]

    def test_rejects_large_body(self) -> None:
        # Body too large to be a pin bar.
        df = _ohlc([1.0], [2.0], [0.0], [1.9])
        result = is_pin_bar_bullish(df, wick_ratio=0.6, body_ratio=0.25)
        assert not result[0]


class TestPinBarBearish:
    def test_detects_shooting_star(self) -> None:
        # Shooting star: open and close near bottom, long upper wick.
        # Range = 2.0. Body = 0.1. Upper wick = 1.7.
        df = _ohlc([0.1], [2.0], [0.0], [0.2])
        result = is_pin_bar_bearish(df, wick_ratio=0.6, body_ratio=0.25)
        assert result[0]


class TestInsideBar:
    def test_detects_inside_bar(self) -> None:
        # Bar 1 high < bar 0 high AND bar 1 low > bar 0 low.
        df = _ohlc([10.0, 10.2], [12.0, 11.5], [8.0, 8.5], [11.0, 10.5])
        result = is_inside_bar(df)
        assert not result[0]
        assert result[1]

    def test_rejects_when_breaks_high(self) -> None:
        df = _ohlc([10.0, 10.2], [12.0, 12.5], [8.0, 8.5], [11.0, 11.5])
        result = is_inside_bar(df)
        assert not result[1]


class TestDoji:
    def test_detects_doji(self) -> None:
        # Body = 0.05, range = 2.0 → body/range = 0.025 < 0.1.
        df = _ohlc([10.0], [11.0], [9.0], [10.05])
        result = is_doji(df, threshold=0.1)
        assert result[0]

    def test_rejects_large_body(self) -> None:
        # Body = 1.0, range = 2.0 → body/range = 0.5 > 0.1.
        df = _ohlc([10.0], [11.0], [9.0], [11.0])
        result = is_doji(df, threshold=0.1)
        assert not result[0]

    def test_zero_range_bar_is_doji(self) -> None:
        # high == low == open == close: the purest possible doji.
        df = _ohlc([10.0], [10.0], [10.0], [10.0])
        result = is_doji(df, threshold=0.1)
        assert result[0]


class TestThreeWhiteSoldiers:
    def _soldiers(self) -> pl.DataFrame:
        # Three bullish candles, each opening within the prior body and closing higher.
        # Bar 0: open=10, close=10.8 (body=0.8, range=1.0 → 80%)
        # Bar 1: open=10.4, close=11.6 (body=1.2, range=1.5 → 80%)
        # Bar 2: open=11.2, close=12.5 (body=1.3, range=1.6 → ~81%)
        return _ohlc(
            [10.0, 10.4, 11.2],
            [11.0, 11.8, 12.8],
            [10.0, 10.3, 11.2],
            [10.8, 11.6, 12.5],
        )

    def test_detects_pattern(self) -> None:
        df = self._soldiers()
        result = is_three_white_soldiers(df, body_ratio=0.5)
        assert not result[0]
        assert not result[1]
        assert result[2]

    def test_first_two_bars_always_false(self) -> None:
        df = self._soldiers()
        result = is_three_white_soldiers(df)
        assert not result[0]
        assert not result[1]

    def test_rejects_if_one_bar_bearish(self) -> None:
        # Middle bar is bearish — pattern fails.
        df = _ohlc(
            [10.0, 11.0, 10.5],
            [11.0, 11.5, 12.0],
            [9.5, 10.5, 10.3],
            [10.8, 10.6, 11.8],
        )
        result = is_three_white_soldiers(df, body_ratio=0.1)
        assert not result[2]


class TestThreeBlackCrows:
    def _crows(self) -> pl.DataFrame:
        # Three bearish candles, each opening within the prior body and closing lower.
        return _ohlc(
            [12.5, 12.1, 11.5],
            [12.8, 12.3, 11.7],
            [11.2, 10.8, 10.2],
            [11.7, 11.0, 10.4],
        )

    def test_detects_pattern(self) -> None:
        df = self._crows()
        result = is_three_black_crows(df, body_ratio=0.5)
        assert not result[0]
        assert not result[1]
        assert result[2]

    def test_rejects_if_one_bar_bullish(self) -> None:
        # Middle bar is bullish — pattern fails.
        df = _ohlc(
            [12.5, 11.0, 11.5],
            [12.8, 11.5, 11.7],
            [11.2, 10.8, 10.2],
            [11.7, 11.3, 10.4],
        )
        result = is_three_black_crows(df, body_ratio=0.1)
        assert not result[2]


class TestMorningStar:
    def test_detects_pattern(self) -> None:
        # Bar 0: large bearish (open=12, close=10, range=2.5, body=2.0 → 80%)
        # Bar 1: small star (open=9.9, close=10.1, range=0.8, body=0.2 → 25% — but we want <= 15%)
        # Use a very small body for the star.
        # Bar 0: open=12, high=12.2, low=9.8, close=10  → body=2.0, range=2.4 → 83%
        # Bar 1: open=9.9, high=10.2, low=9.7, close=10.0 → body=0.1, range=0.5 → 20% > 15%
        # Let's use body=0.05, range=0.5 → 10% ≤ 15%
        # Bar 2: open=10.2, high=11.5, low=10.1, close=11.2 → body=1.0, range=1.4 → 71%
        #   midpoint of bar 0 body = (12+10)/2 = 11.0; close=11.2 > 11.0 ✓
        df = _ohlc(
            [12.0, 9.95, 10.2],
            [12.2, 10.2, 11.5],
            [9.8, 9.7, 10.1],
            [10.0, 10.0, 11.2],
        )
        result = is_morning_star(df, body_ratio=0.3, star_body_ratio=0.15)
        assert not result[0]
        assert not result[1]
        assert result[2]

    def test_first_two_bars_always_false(self) -> None:
        df = _ohlc([12.0, 9.95, 10.2], [12.2, 10.2, 11.5], [9.8, 9.7, 10.1], [10.0, 10.0, 11.2])
        result = is_morning_star(df)
        assert not result[0]
        assert not result[1]

    def test_rejects_if_bar3_does_not_close_into_bar1(self) -> None:
        # Bar 3 closes at 10.4, midpoint is 11.0 — does not penetrate.
        df = _ohlc(
            [12.0, 9.95, 10.2],
            [12.2, 10.2, 10.8],
            [9.8, 9.7, 10.1],
            [10.0, 10.0, 10.4],
        )
        result = is_morning_star(df, body_ratio=0.1, star_body_ratio=0.15)
        assert not result[2]


class TestEveningStar:
    def test_detects_pattern(self) -> None:
        # Bar 0: large bullish (open=10, close=12)
        # Bar 1: small star
        # Bar 2: bearish, closes below midpoint of bar 0 body (midpoint=11.0)
        df = _ohlc(
            [10.0, 12.05, 11.8],
            [12.2, 12.3, 12.0],
            [9.8, 11.7, 10.5],
            [12.0, 12.1, 10.6],
        )
        result = is_evening_star(df, body_ratio=0.3, star_body_ratio=0.15)
        assert result[2]

    def test_first_two_bars_always_false(self) -> None:
        df = _ohlc([10.0, 12.05, 11.8], [12.2, 12.3, 12.0], [9.8, 11.7, 10.5], [12.0, 12.1, 10.6])
        result = is_evening_star(df)
        assert not result[0]
        assert not result[1]


class TestBullishHarami:
    def test_detects_pattern(self) -> None:
        # Bar 0: large bearish open=12, close=10 (body=2, inside bounds 10..12)
        # Bar 1: small bullish open=10.5, close=11.0 (body contained in 10..12)
        df = _ohlc([12.0, 10.5], [12.5, 11.2], [9.8, 10.3], [10.0, 11.0])
        result = is_bullish_harami(df)
        assert not result[0]
        assert result[1]

    def test_rejects_body_outside_prior(self) -> None:
        # Bar 1 close (12.5) exceeds prior open (12.0) — not contained.
        df = _ohlc([12.0, 10.5], [12.5, 13.0], [9.8, 10.3], [10.0, 12.5])
        result = is_bullish_harami(df)
        assert not result[1]

    def test_rejects_if_prior_not_bearish(self) -> None:
        # Prior bar is bullish — no harami.
        df = _ohlc([10.0, 10.5], [12.5, 11.2], [9.8, 10.3], [12.0, 11.0])
        result = is_bullish_harami(df)
        assert not result[1]


class TestBearishHarami:
    def test_detects_pattern(self) -> None:
        # Bar 0: large bullish open=10, close=12
        # Bar 1: small bearish open=11.5, close=10.8 (body contained in 10..12)
        df = _ohlc([10.0, 11.5], [12.5, 11.8], [9.8, 10.6], [12.0, 10.8])
        result = is_bearish_harami(df)
        assert not result[0]
        assert result[1]

    def test_rejects_body_outside_prior(self) -> None:
        # Bar 1 open (12.5) exceeds prior close (12.0) — not contained.
        df = _ohlc([10.0, 12.5], [12.5, 13.0], [9.8, 11.0], [12.0, 11.5])
        result = is_bearish_harami(df)
        assert not result[1]
