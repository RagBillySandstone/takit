"""Unit tests for is_abandoned_baby_bullish and is_abandoned_baby_bearish."""

from __future__ import annotations

import polars as pl

from polarticks.patterns import is_abandoned_baby_bearish, is_abandoned_baby_bullish

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlc(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> pl.DataFrame:
    """Build a minimal OHLC DataFrame from lists."""
    return pl.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


# ---------------------------------------------------------------------------
# Bullish Abandoned Baby
# ---------------------------------------------------------------------------


class TestAbandonedBabyBullish:
    """Tests for the bullish abandoned baby pattern."""

    def test_canonical_pattern_detected(self) -> None:
        """A textbook bullish abandoned baby must be detected on Bar 3.

        Conditions at index 3 (viewing shift(2)=index 1 and shift(1)=index 2):
          Bar 1 (idx 1): large bearish (close=105 < open=110, ratio≈0.71)
          Bar 2 (idx 2): doji (body=0.1 / range=1.0 = 0.10) with high=101.5 < Bar 1 low=104
          Bar 3 (idx 3): large bullish (close=107 > open=103, ratio≈0.67) with low=102 > Bar 2 high=101.5
        """
        df = _make_ohlc(
            opens=[100.0, 110.0, 101.0, 103.0],
            highs=[100.5, 111.0, 101.5, 108.0],
            lows=[99.5, 104.0, 100.5, 102.0],
            closes=[100.2, 105.0, 101.1, 107.0],
        )
        result = is_abandoned_baby_bullish(df)
        assert result[3] is True

    def test_first_two_bars_always_false(self) -> None:
        """The first two bars can never complete the 3-bar pattern."""
        df = _make_ohlc(
            opens=[100.0, 110.0, 101.0, 103.0],
            highs=[100.5, 111.0, 101.5, 108.0],
            lows=[99.5, 104.0, 100.5, 102.0],
            closes=[100.2, 105.0, 101.1, 107.0],
        )
        result = is_abandoned_baby_bullish(df)
        assert result[0] is False
        assert result[1] is False

    def test_output_length_matches_input(self) -> None:
        """Output Series must have the same length as the input DataFrame."""
        df = _make_ohlc([100.0] * 10, [101.0] * 10, [99.0] * 10, [100.0] * 10)
        assert len(is_abandoned_baby_bullish(df)) == 10

    def test_alias_is_correct(self) -> None:
        """Output Series name must be 'abandoned_baby_bullish'."""
        df = _make_ohlc([100.0] * 5, [101.0] * 5, [99.0] * 5, [100.0] * 5)
        assert is_abandoned_baby_bullish(df).name == "abandoned_baby_bullish"

    def test_no_gap_not_detected(self) -> None:
        """Without a strict gap between Bar 1 and Bar 2 the pattern must not fire.

        Make Bar 2's high (104.5) >= Bar 1's low (104.0) so gap_down_from_bar1 is False.
        """
        df = _make_ohlc(
            opens=[100.0, 110.0, 104.0, 103.0],
            highs=[100.5, 111.0, 104.5, 108.0],  # Bar 2 high=104.5 >= Bar 1 low=104.0 → no gap
            lows=[99.5, 104.0, 103.5, 102.0],
            closes=[100.2, 105.0, 104.1, 107.0],
        )
        result = is_abandoned_baby_bullish(df)
        assert result[3] is False

    def test_dtype_is_bool(self) -> None:
        """Output dtype must be Boolean."""
        df = _make_ohlc([100.0] * 5, [101.0] * 5, [99.0] * 5, [100.0] * 5)
        assert is_abandoned_baby_bullish(df).dtype == pl.Boolean

    def test_returns_false_not_null_on_early_bars(self) -> None:
        """fill_null(False) must be applied — no null values in the output."""
        df = _make_ohlc([100.0] * 10, [101.0] * 10, [99.0] * 10, [100.0] * 10)
        result = is_abandoned_baby_bullish(df)
        assert result.null_count() == 0


# ---------------------------------------------------------------------------
# Bearish Abandoned Baby
# ---------------------------------------------------------------------------


class TestAbandonedBabyBearish:
    """Tests for the bearish abandoned baby pattern."""

    def test_canonical_pattern_detected(self) -> None:
        """A textbook bearish abandoned baby must be detected on Bar 3.

        Conditions at index 3 (viewing shift(2)=index 1 and shift(1)=index 2):
          Bar 1 (idx 1): large bullish (close=109 > open=100, ratio=0.9)
          Bar 2 (idx 2): doji (body=0.1 / range=1.0 = 0.10) with low=111.5 > Bar 1 high=110
          Bar 3 (idx 3): large bearish (close=107 < open=109, ratio≈0.57) with high=109.5 < Bar 2 low=111.5
        """
        df = _make_ohlc(
            opens=[100.0, 100.0, 112.0, 109.0],
            highs=[100.5, 110.0, 112.5, 109.5],
            lows=[99.5, 100.0, 111.5, 106.0],
            closes=[100.2, 109.0, 112.1, 107.0],
        )
        result = is_abandoned_baby_bearish(df)
        assert result[3] is True

    def test_first_two_bars_always_false(self) -> None:
        """The first two bars can never complete the 3-bar pattern."""
        df = _make_ohlc(
            opens=[100.0, 100.0, 112.0, 109.0],
            highs=[100.5, 110.0, 112.5, 109.5],
            lows=[99.5, 100.0, 111.5, 106.0],
            closes=[100.2, 109.0, 112.1, 107.0],
        )
        result = is_abandoned_baby_bearish(df)
        assert result[0] is False
        assert result[1] is False

    def test_output_length_matches_input(self) -> None:
        """Output Series must have the same length as the input DataFrame."""
        df = _make_ohlc([100.0] * 10, [101.0] * 10, [99.0] * 10, [100.0] * 10)
        assert len(is_abandoned_baby_bearish(df)) == 10

    def test_alias_is_correct(self) -> None:
        """Output Series name must be 'abandoned_baby_bearish'."""
        df = _make_ohlc([100.0] * 5, [101.0] * 5, [99.0] * 5, [100.0] * 5)
        assert is_abandoned_baby_bearish(df).name == "abandoned_baby_bearish"

    def test_no_gap_not_detected(self) -> None:
        """Without a strict gap between Bar 1 and Bar 2 the pattern must not fire.

        Make Bar 2's low (110.0) <= Bar 1's high (110.0) so gap_up_from_bar1 is False.
        """
        df = _make_ohlc(
            opens=[100.0, 100.0, 110.0, 109.0],
            highs=[100.5, 110.0, 110.5, 109.5],
            lows=[99.5, 100.0, 110.0, 106.0],  # Bar 2 low=110.0 not > Bar 1 high=110.0
            closes=[100.2, 109.0, 110.2, 107.0],
        )
        result = is_abandoned_baby_bearish(df)
        assert result[3] is False

    def test_dtype_is_bool(self) -> None:
        """Output dtype must be Boolean."""
        df = _make_ohlc([100.0] * 5, [101.0] * 5, [99.0] * 5, [100.0] * 5)
        assert is_abandoned_baby_bearish(df).dtype == pl.Boolean

    def test_returns_false_not_null_on_early_bars(self) -> None:
        """fill_null(False) must be applied — no null values in the output."""
        df = _make_ohlc([100.0] * 10, [101.0] * 10, [99.0] * 10, [100.0] * 10)
        result = is_abandoned_baby_bearish(df)
        assert result.null_count() == 0
