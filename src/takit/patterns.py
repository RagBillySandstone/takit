"""
Candlestick pattern detection.

All functions accept a ``pl.DataFrame`` with columns ``open``, ``high``,
``low``, ``close`` and return a Boolean ``pl.Series`` — ``True`` on bars
where the pattern is present.

Patterns are defined by common structural rules.  The ``body_pct``
parameter (default 0.3) controls the minimum body-to-range ratio required
to confirm a candle as "real" (non-doji).  Adjust to taste.

Functions
---------
is_bullish_engulfing        Bearish candle followed by a larger bullish candle
is_bearish_engulfing        Bullish candle followed by a larger bearish candle
is_pin_bar_bullish          Hammer: small body near top, long lower wick
is_pin_bar_bearish          Shooting star: small body near bottom, long upper wick
is_inside_bar               Current bar's range is fully contained by the prior bar
is_doji                     Open ≈ Close (indecision candle)
is_three_white_soldiers     Three consecutive substantial bullish candles advancing higher
is_three_black_crows        Three consecutive substantial bearish candles declining lower
is_morning_star             Three-candle bullish reversal with a small star in the middle
is_evening_star             Three-candle bearish reversal with a small star in the middle
is_bullish_harami           Small bullish body contained within a prior large bearish body
is_bearish_harami           Small bearish body contained within a prior large bullish body
"""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _body(ohlc: pl.DataFrame) -> pl.Series:
    """Absolute candle body size: |close - open|."""
    return (ohlc["close"] - ohlc["open"]).abs()


def _range(ohlc: pl.DataFrame) -> pl.Series:
    """Full candle range: high - low."""
    return ohlc["high"] - ohlc["low"]


def _is_bullish(ohlc: pl.DataFrame) -> pl.Series:
    """True where close > open (bullish/green candle)."""
    return ohlc["close"] > ohlc["open"]


def _is_bearish(ohlc: pl.DataFrame) -> pl.Series:
    """True where close < open (bearish/red candle)."""
    return ohlc["close"] < ohlc["open"]


# ---------------------------------------------------------------------------
# Engulfing patterns
# ---------------------------------------------------------------------------


def is_bullish_engulfing(ohlc: pl.DataFrame) -> pl.Series:
    """Detect bullish engulfing candles.

    A bullish engulfing pattern requires:
        1. The prior bar is bearish (close < open).
        2. The current bar is bullish (close > open).
        3. The current bar's body completely engulfs the prior bar's body:
           current open ≤ prior close AND current close ≥ prior open.

    This pattern signals a potential bullish reversal after a downtrend.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.

    Returns:
        Boolean Series, ``True`` on bars where the pattern is present.
        The first bar is always ``False`` (no prior bar to compare).
    """
    prev_open = ohlc["open"].shift(1)
    prev_close = ohlc["close"].shift(1)

    prior_bearish = _is_bearish(pl.DataFrame({"open": prev_open, "close": prev_close}))
    curr_bullish = _is_bullish(ohlc)

    # Current body engulfs prior body.
    engulfs = (ohlc["open"] <= prev_close) & (ohlc["close"] >= prev_open)

    return (prior_bearish & curr_bullish & engulfs).fill_null(False).alias("bullish_engulfing")


def is_bearish_engulfing(ohlc: pl.DataFrame) -> pl.Series:
    """Detect bearish engulfing candles.

    A bearish engulfing pattern requires:
        1. The prior bar is bullish (close > open).
        2. The current bar is bearish (close < open).
        3. The current bar's body completely engulfs the prior bar's body:
           current open ≥ prior close AND current close ≤ prior open.

    This pattern signals a potential bearish reversal after an uptrend.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.

    Returns:
        Boolean Series, ``True`` on bars where the pattern is present.
    """
    prev_open = ohlc["open"].shift(1)
    prev_close = ohlc["close"].shift(1)

    prior_bullish = _is_bullish(pl.DataFrame({"open": prev_open, "close": prev_close}))
    curr_bearish = _is_bearish(ohlc)

    engulfs = (ohlc["open"] >= prev_close) & (ohlc["close"] <= prev_open)

    return (prior_bullish & curr_bearish & engulfs).fill_null(False).alias("bearish_engulfing")


# ---------------------------------------------------------------------------
# Pin bars
# ---------------------------------------------------------------------------


def is_pin_bar_bullish(
    ohlc: pl.DataFrame,
    wick_ratio: float = 0.6,
    body_ratio: float = 0.25,
) -> pl.Series:
    """Detect bullish pin bars (hammers).

    A bullish pin bar (hammer) has:
        - A long lower wick (≥ ``wick_ratio`` × total range).
        - A small body (≤ ``body_ratio`` × total range).
        - The body is in the upper portion of the bar.

    These form after a decline and signal a potential reversal as sellers
    were unable to hold price down.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        wick_ratio: Minimum lower-wick-to-range ratio (default 0.6).
        body_ratio: Maximum body-to-range ratio (default 0.25).

    Returns:
        Boolean Series, ``True`` on bullish pin bar bars.
    """
    candle_range = _range(ohlc)
    body = _body(ohlc)

    # Lower wick = min(open, close) - low; min_horizontal avoids a Python loop.
    # pl.select() materialises the Expr to a Series so arithmetic is possible.
    lower_wick = pl.select(pl.min_horizontal(ohlc["open"], ohlc["close"])).to_series() - ohlc["low"]

    # Avoid division by zero on doji-like bars with zero range.
    safe_range = candle_range.replace(0.0, float("nan"))

    long_lower_wick = lower_wick / safe_range >= wick_ratio
    small_body = body / safe_range <= body_ratio

    return (long_lower_wick & small_body).fill_null(False).alias("pin_bar_bullish")


def is_pin_bar_bearish(
    ohlc: pl.DataFrame,
    wick_ratio: float = 0.6,
    body_ratio: float = 0.25,
) -> pl.Series:
    """Detect bearish pin bars (shooting stars / inverted hammers).

    A bearish pin bar has:
        - A long upper wick (≥ ``wick_ratio`` × total range).
        - A small body (≤ ``body_ratio`` × total range).
        - The body is in the lower portion of the bar.

    These form after a rally and signal potential reversal as buyers were
    unable to hold price up.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        wick_ratio: Minimum upper-wick-to-range ratio (default 0.6).
        body_ratio: Maximum body-to-range ratio (default 0.25).

    Returns:
        Boolean Series, ``True`` on bearish pin bar bars.
    """
    candle_range = _range(ohlc)
    body = _body(ohlc)

    # Upper wick = high - max(open, close); max_horizontal avoids a Python loop.
    upper_wick = (
        ohlc["high"] - pl.select(pl.max_horizontal(ohlc["open"], ohlc["close"])).to_series()
    )

    safe_range = candle_range.replace(0.0, float("nan"))

    long_upper_wick = upper_wick / safe_range >= wick_ratio
    small_body = body / safe_range <= body_ratio

    return (long_upper_wick & small_body).fill_null(False).alias("pin_bar_bearish")


# ---------------------------------------------------------------------------
# Inside bar
# ---------------------------------------------------------------------------


def is_inside_bar(ohlc: pl.DataFrame) -> pl.Series:
    """Detect inside bars (price consolidation within prior bar's range).

    An inside bar's high is strictly lower than the prior bar's high, and
    its low is strictly higher than the prior bar's low.  This signals
    market indecision and is often traded as a breakout setup.

    Args:
        ohlc: DataFrame with columns ``high`` and ``low``.

    Returns:
        Boolean Series, ``True`` on inside bar bars.
        The first bar is always ``False``.
    """
    prev_high = ohlc["high"].shift(1)
    prev_low = ohlc["low"].shift(1)

    inside = (ohlc["high"] < prev_high) & (ohlc["low"] > prev_low)
    return inside.fill_null(False).alias("inside_bar")


# ---------------------------------------------------------------------------
# Doji
# ---------------------------------------------------------------------------


def is_doji(ohlc: pl.DataFrame, threshold: float = 0.1) -> pl.Series:
    """Detect doji candles (open ≈ close, indicating indecision).

    A doji is identified when the body size is less than *threshold* times
    the total range.  The precise threshold is subjective; 0.1 (10%) is a
    common default.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        threshold: Maximum body/range ratio for a bar to qualify as a doji
                   (default 0.1).

    Returns:
        Boolean Series, ``True`` on doji bars.
    """
    candle_range = _range(ohlc)
    body = _body(ohlc)

    safe_range = candle_range.replace(0.0, float("nan"))
    return (body / safe_range <= threshold).fill_null(True).alias("doji")


# ---------------------------------------------------------------------------
# Three white soldiers / three black crows
# ---------------------------------------------------------------------------


def is_three_white_soldiers(
    ohlc: pl.DataFrame,
    body_ratio: float = 0.5,
) -> pl.Series:
    """Detect three white soldiers (consecutive bullish advance).

    Three white soldiers is a bullish continuation / reversal pattern
    consisting of three consecutive bullish candles where:

        1. Each candle has a substantial body (``body_ratio`` × full range).
        2. Each candle opens within the body of the prior candle.
        3. Each candle closes higher than the prior candle's close.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        body_ratio: Minimum body-to-range ratio for each candle (default 0.5).

    Returns:
        Boolean Series, ``True`` on the third (confirming) bar.
        The first two bars are always ``False``.
    """
    open_ = ohlc["open"]
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    # Shifted values for the two prior bars.
    open_1 = open_.shift(1)
    close_1 = close.shift(1)
    open_2 = open_.shift(2)
    close_2 = close.shift(2)
    high_1 = high.shift(1)
    low_1 = low.shift(1)
    high_2 = high.shift(2)
    low_2 = low.shift(2)

    # All three candles must be bullish (close > open).
    bull_0 = close > open_
    bull_1 = close_1 > open_1
    bull_2 = close_2 > open_2

    # Each close must be strictly higher than the prior close.
    higher_0 = close > close_1
    higher_1 = close_1 > close_2

    # Each open must fall within the prior candle's real body:
    # prior_open <= current_open <= prior_close (since all are bullish).
    open_in_body_0 = (open_ >= open_1) & (open_ <= close_1)
    open_in_body_1 = (open_1 >= open_2) & (open_1 <= close_2)

    # Each candle must have a substantial body — avoids counting doji runs.
    body_0 = (close - open_).abs()
    safe_range_0 = (high - low).replace(0.0, float("nan"))
    big_0 = body_0 / safe_range_0 >= body_ratio

    body_1 = (close_1 - open_1).abs()
    safe_range_1 = (high_1 - low_1).replace(0.0, float("nan"))
    big_1 = body_1 / safe_range_1 >= body_ratio

    body_2 = (close_2 - open_2).abs()
    safe_range_2 = (high_2 - low_2).replace(0.0, float("nan"))
    big_2 = body_2 / safe_range_2 >= body_ratio

    result = (
        bull_0
        & bull_1
        & bull_2
        & higher_0
        & higher_1
        & open_in_body_0
        & open_in_body_1
        & big_0
        & big_1
        & big_2
    )
    return result.fill_null(False).alias("three_white_soldiers")


def is_three_black_crows(
    ohlc: pl.DataFrame,
    body_ratio: float = 0.5,
) -> pl.Series:
    """Detect three black crows (consecutive bearish decline).

    Three black crows is the bearish mirror of three white soldiers:
    three consecutive bearish candles where each opens within the prior
    candle's body and closes lower.

        1. Each candle has a substantial body (``body_ratio`` × full range).
        2. Each candle opens within the body of the prior candle.
        3. Each candle closes lower than the prior candle's close.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        body_ratio: Minimum body-to-range ratio for each candle (default 0.5).

    Returns:
        Boolean Series, ``True`` on the third (confirming) bar.
        The first two bars are always ``False``.
    """
    open_ = ohlc["open"]
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    open_1 = open_.shift(1)
    close_1 = close.shift(1)
    open_2 = open_.shift(2)
    close_2 = close.shift(2)
    high_1 = high.shift(1)
    low_1 = low.shift(1)
    high_2 = high.shift(2)
    low_2 = low.shift(2)

    # All three candles must be bearish (open > close).
    bear_0 = open_ > close
    bear_1 = open_1 > close_1
    bear_2 = open_2 > close_2

    # Each close must be strictly lower than the prior close.
    lower_0 = close < close_1
    lower_1 = close_1 < close_2

    # Each open must fall within the prior bearish candle's real body:
    # prior_close <= current_open <= prior_open (bearish: open > close).
    open_in_body_0 = (open_ >= close_1) & (open_ <= open_1)
    open_in_body_1 = (open_1 >= close_2) & (open_1 <= open_2)

    body_0 = (open_ - close).abs()
    safe_range_0 = (high - low).replace(0.0, float("nan"))
    big_0 = body_0 / safe_range_0 >= body_ratio

    body_1 = (open_1 - close_1).abs()
    safe_range_1 = (high_1 - low_1).replace(0.0, float("nan"))
    big_1 = body_1 / safe_range_1 >= body_ratio

    body_2 = (open_2 - close_2).abs()
    safe_range_2 = (high_2 - low_2).replace(0.0, float("nan"))
    big_2 = body_2 / safe_range_2 >= body_ratio

    result = (
        bear_0
        & bear_1
        & bear_2
        & lower_0
        & lower_1
        & open_in_body_0
        & open_in_body_1
        & big_0
        & big_1
        & big_2
    )
    return result.fill_null(False).alias("three_black_crows")


# ---------------------------------------------------------------------------
# Morning star / evening star
# ---------------------------------------------------------------------------


def is_morning_star(
    ohlc: pl.DataFrame,
    body_ratio: float = 0.3,
    star_body_ratio: float = 0.15,
) -> pl.Series:
    """Detect morning star (three-candle bullish reversal).

    The morning star is a three-candle pattern signalling a potential
    bottom reversal:

        Bar 1 (i-2):  Large bearish candle — body ≥ ``body_ratio`` × range.
        Bar 2 (i-1):  Small-bodied "star" — body ≤ ``star_body_ratio`` × range.
                      Indicates indecision at the bottom.
        Bar 3 (i):    Bullish candle that closes above the midpoint of Bar 1's
                      body, demonstrating strong buying recovery.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        body_ratio: Minimum body-to-range ratio for Bar 1 and Bar 3 (default 0.3).
        star_body_ratio: Maximum body-to-range ratio for the star candle (default 0.15).

    Returns:
        Boolean Series, ``True`` on Bar 3 (the confirming candle).
        The first two bars are always ``False``.
    """
    open_ = ohlc["open"]
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    # Bar 1: two periods ago (the large bearish candle).
    open_2 = open_.shift(2)
    high_2 = high.shift(2)
    low_2 = low.shift(2)
    close_2 = close.shift(2)

    # Bar 2: one period ago (the star).
    open_1 = open_.shift(1)
    high_1 = high.shift(1)
    low_1 = low.shift(1)
    close_1 = close.shift(1)

    # Bar 1 must be a large bearish candle.
    range_2 = high_2 - low_2
    body_2 = (open_2 - close_2).abs()
    safe_range_2 = range_2.replace(0.0, float("nan"))
    bar1_large_bearish = (close_2 < open_2) & (body_2 / safe_range_2 >= body_ratio)

    # Bar 2 (star) must have a small body.
    range_1 = high_1 - low_1
    body_1 = (close_1 - open_1).abs()
    safe_range_1 = range_1.replace(0.0, float("nan"))
    bar2_small = body_1 / safe_range_1 <= star_body_ratio

    # Bar 3 must be bullish and close above the midpoint of Bar 1's body.
    range_0 = high - low
    body_0 = (close - open_).abs()
    safe_range_0 = range_0.replace(0.0, float("nan"))
    bar3_bullish = (close > open_) & (body_0 / safe_range_0 >= body_ratio)

    # Midpoint of Bar 1's bearish body: average of Bar 1's open and close.
    bar1_midpoint = (open_2 + close_2) / 2.0
    bar3_closes_into_bar1 = close > bar1_midpoint

    result = bar1_large_bearish & bar2_small & bar3_bullish & bar3_closes_into_bar1
    return result.fill_null(False).alias("morning_star")


def is_evening_star(
    ohlc: pl.DataFrame,
    body_ratio: float = 0.3,
    star_body_ratio: float = 0.15,
) -> pl.Series:
    """Detect evening star (three-candle bearish reversal).

    The evening star is the bearish mirror of the morning star — a three-candle
    pattern signalling a potential top reversal:

        Bar 1 (i-2):  Large bullish candle — body ≥ ``body_ratio`` × range.
        Bar 2 (i-1):  Small-bodied "star" — body ≤ ``star_body_ratio`` × range.
                      Indicates indecision at the top.
        Bar 3 (i):    Bearish candle that closes below the midpoint of Bar 1's
                      body, demonstrating strong selling pressure.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        body_ratio: Minimum body-to-range ratio for Bar 1 and Bar 3 (default 0.3).
        star_body_ratio: Maximum body-to-range ratio for the star candle (default 0.15).

    Returns:
        Boolean Series, ``True`` on Bar 3 (the confirming candle).
        The first two bars are always ``False``.
    """
    open_ = ohlc["open"]
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    # Bar 1: two periods ago (the large bullish candle).
    open_2 = open_.shift(2)
    high_2 = high.shift(2)
    low_2 = low.shift(2)
    close_2 = close.shift(2)

    # Bar 2: one period ago (the star).
    open_1 = open_.shift(1)
    high_1 = high.shift(1)
    low_1 = low.shift(1)
    close_1 = close.shift(1)

    # Bar 1 must be a large bullish candle.
    range_2 = high_2 - low_2
    body_2 = (close_2 - open_2).abs()
    safe_range_2 = range_2.replace(0.0, float("nan"))
    bar1_large_bullish = (close_2 > open_2) & (body_2 / safe_range_2 >= body_ratio)

    # Bar 2 (star) must have a small body.
    range_1 = high_1 - low_1
    body_1 = (close_1 - open_1).abs()
    safe_range_1 = range_1.replace(0.0, float("nan"))
    bar2_small = body_1 / safe_range_1 <= star_body_ratio

    # Bar 3 must be bearish and close below the midpoint of Bar 1's body.
    range_0 = high - low
    body_0 = (close - open_).abs()
    safe_range_0 = range_0.replace(0.0, float("nan"))
    bar3_bearish = (open_ > close) & (body_0 / safe_range_0 >= body_ratio)

    # Midpoint of Bar 1's bullish body.
    bar1_midpoint = (open_2 + close_2) / 2.0
    bar3_closes_into_bar1 = close < bar1_midpoint

    result = bar1_large_bullish & bar2_small & bar3_bearish & bar3_closes_into_bar1
    return result.fill_null(False).alias("evening_star")


# ---------------------------------------------------------------------------
# Harami
# ---------------------------------------------------------------------------


def is_bullish_harami(ohlc: pl.DataFrame) -> pl.Series:
    """Detect bullish harami (inside candle after a large bearish bar).

    A bullish harami ("pregnant" in Japanese) consists of:

        Bar 1 (i-1):  Large bearish candle.
        Bar 2 (i):    Small bullish candle whose real body is entirely
                      contained within Bar 1's real body.

    The pattern suggests that the prior bearish momentum is stalling and
    a potential reversal may follow.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.

    Returns:
        Boolean Series, ``True`` on Bar 2 (the inside candle).
        The first bar is always ``False``.
    """
    prev_open = ohlc["open"].shift(1)
    prev_close = ohlc["close"].shift(1)

    # Prior bar must be bearish: open[i-1] > close[i-1].
    prior_bearish = _is_bearish(pl.DataFrame({"open": prev_open, "close": prev_close}))

    # Current bar must be bullish.
    curr_bullish = _is_bullish(ohlc)

    # Current body must be contained within the prior bearish body.
    # Prior bearish body spans from close[i-1] (low) to open[i-1] (high).
    # Current bullish body spans from open[i] (low) to close[i] (high).
    body_contained = (ohlc["open"] >= prev_close) & (ohlc["close"] <= prev_open)

    return (prior_bearish & curr_bullish & body_contained).fill_null(False).alias("bullish_harami")


def is_bearish_harami(ohlc: pl.DataFrame) -> pl.Series:
    """Detect bearish harami (inside candle after a large bullish bar).

    A bearish harami consists of:

        Bar 1 (i-1):  Large bullish candle.
        Bar 2 (i):    Small bearish candle whose real body is entirely
                      contained within Bar 1's real body.

    The pattern suggests that the prior bullish momentum is stalling and
    a potential reversal may follow.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.

    Returns:
        Boolean Series, ``True`` on Bar 2 (the inside candle).
        The first bar is always ``False``.
    """
    prev_open = ohlc["open"].shift(1)
    prev_close = ohlc["close"].shift(1)

    # Prior bar must be bullish: close[i-1] > open[i-1].
    prior_bullish = _is_bullish(pl.DataFrame({"open": prev_open, "close": prev_close}))

    # Current bar must be bearish.
    curr_bearish = _is_bearish(ohlc)

    # Current body must be contained within the prior bullish body.
    # Prior bullish body spans from open[i-1] (low) to close[i-1] (high).
    # Current bearish body spans from close[i] (low) to open[i] (high).
    body_contained = (ohlc["close"] >= prev_open) & (ohlc["open"] <= prev_close)

    return (prior_bullish & curr_bearish & body_contained).fill_null(False).alias("bearish_harami")
