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
is_bullish_engulfing            Bearish candle followed by a larger bullish candle
is_bearish_engulfing            Bullish candle followed by a larger bearish candle
is_pin_bar_bullish              Hammer: small body near top, long lower wick
is_pin_bar_bearish              Shooting star: small body near bottom, long upper wick
is_inside_bar                   Current bar's range is fully contained by the prior bar
is_doji                         Open ≈ Close (indecision candle)
is_three_white_soldiers         Three consecutive substantial bullish candles advancing higher
is_three_black_crows            Three consecutive substantial bearish candles declining lower
is_morning_star                 Three-candle bullish reversal with a small star in the middle
is_evening_star                 Three-candle bearish reversal with a small star in the middle
is_bullish_harami               Small bullish body contained within a prior large bearish body
is_bearish_harami               Small bearish body contained within a prior large bullish body
is_abandoned_baby_bullish       Gap-down doji between a bearish and a bullish bar (bottom reversal)
is_abandoned_baby_bearish       Gap-up doji between a bullish and a bearish bar (top reversal)
is_hanging_man                  Hammer structure appearing after an uptrend (bearish reversal signal)
is_inverted_hammer              Shooting-star structure after a downtrend (bullish reversal signal)
is_tweezer_top                  Two-bar equal-high bearish reversal pattern
is_tweezer_bottom               Two-bar equal-low bullish reversal pattern
is_dark_cloud_cover             Bearish two-bar reversal: bearish bar opens above prior high, closes into prior body
is_piercing_line                Bullish two-bar reversal: bullish bar opens below prior low, closes into prior body
is_rising_three_methods         Five-bar bullish continuation pattern
is_falling_three_methods        Five-bar bearish continuation pattern
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


def _body_range_ratio(
    open_: pl.Series,
    close: pl.Series,
    high: pl.Series,
    low: pl.Series,
) -> pl.Series:
    """Body-to-range ratio for each bar; ``NaN`` on zero-range bars.

    Args:
        open_: Open price series (possibly shifted).
        close: Close price series (possibly shifted).
        high: High price series (possibly shifted).
        low: Low price series (possibly shifted).

    Returns:
        Series of body-to-range ratios; ``NaN`` where ``high == low``.
    """
    body = (close - open_).abs()
    safe_range = (high - low).replace(0.0, float("nan"))
    return body / safe_range


def _big_body(
    open_: pl.Series,
    close: pl.Series,
    high: pl.Series,
    low: pl.Series,
    body_ratio: float,
) -> pl.Series:
    """True where body/range >= body_ratio (substantial candle body).

    Args:
        open_: Open price series (possibly shifted).
        close: Close price series (possibly shifted).
        high: High price series (possibly shifted).
        low: Low price series (possibly shifted).
        body_ratio: Minimum body-to-range ratio.

    Returns:
        Boolean Series; ``False`` on zero-range bars.
    """
    return _body_range_ratio(open_, close, high, low) >= body_ratio


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

    prior_bearish = prev_close < prev_open
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

    prior_bullish = prev_close > prev_open
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
    # fill_nan on the float ratio before the comparison: a zero-range bar
    # produces NaN which we map to 0.0 so it satisfies any positive threshold
    # (a zero-range bar is the purest doji).
    return ((body / safe_range).fill_nan(0.0) <= threshold).fill_null(True).alias("doji")


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
    big_0 = _big_body(open_, close, high, low, body_ratio)
    big_1 = _big_body(open_1, close_1, high_1, low_1, body_ratio)
    big_2 = _big_body(open_2, close_2, high_2, low_2, body_ratio)

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

    # Each candle must have a substantial body — avoids counting doji runs.
    big_0 = _big_body(open_, close, high, low, body_ratio)
    big_1 = _big_body(open_1, close_1, high_1, low_1, body_ratio)
    big_2 = _big_body(open_2, close_2, high_2, low_2, body_ratio)

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

    # Bar 1 must be a large bearish candle with a substantial body.
    bar1_large_bearish = (close_2 < open_2) & (
        _body_range_ratio(open_2, close_2, high_2, low_2) >= body_ratio
    )

    # Bar 2 (star) must have a small body indicating indecision.
    bar2_small = _body_range_ratio(open_1, close_1, high_1, low_1) <= star_body_ratio

    # Bar 3 must be bullish with a substantial body closing above Bar 1's midpoint.
    bar3_bullish = (close > open_) & (_body_range_ratio(open_, close, high, low) >= body_ratio)

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

    # Bar 1 must be a large bullish candle with a substantial body.
    bar1_large_bullish = (close_2 > open_2) & (
        _body_range_ratio(open_2, close_2, high_2, low_2) >= body_ratio
    )

    # Bar 2 (star) must have a small body indicating indecision.
    bar2_small = _body_range_ratio(open_1, close_1, high_1, low_1) <= star_body_ratio

    # Bar 3 must be bearish with a substantial body closing below Bar 1's midpoint.
    bar3_bearish = (open_ > close) & (_body_range_ratio(open_, close, high, low) >= body_ratio)

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
    prior_bearish = prev_close < prev_open

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
    prior_bullish = prev_close > prev_open

    # Current bar must be bearish.
    curr_bearish = _is_bearish(ohlc)

    # Current body must be contained within the prior bullish body.
    # Prior bullish body spans from open[i-1] (low) to close[i-1] (high).
    # Current bearish body spans from close[i] (low) to open[i] (high).
    body_contained = (ohlc["close"] >= prev_open) & (ohlc["open"] <= prev_close)

    return (prior_bullish & curr_bearish & body_contained).fill_null(False).alias("bearish_harami")


# ---------------------------------------------------------------------------
# Abandoned Baby
# ---------------------------------------------------------------------------


def is_abandoned_baby_bullish(
    ohlc: pl.DataFrame,
    body_ratio: float = 0.3,
    doji_ratio: float = 0.1,
) -> pl.Series:
    """Detect bullish abandoned baby — gap-reversal three-candle bottom pattern.

    The bullish abandoned baby signals a potential bottom reversal when a doji
    "abandons" both neighbouring candles via strict price gaps:

        Bar 1 (i-2): Large bearish candle (body ≥ ``body_ratio`` × range).
        Bar 2 (i-1): Doji (body ≤ ``doji_ratio`` × range) with a full gap
                     *below* Bar 1 — Bar 2's high is strictly below Bar 1's low.
        Bar 3 (i):   Large bullish candle with a full gap *above* Bar 2 —
                     Bar 3's low is strictly above Bar 2's high.

    Both gaps must be strict (no price overlap) for the pattern to qualify.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        body_ratio: Minimum body-to-range ratio for Bars 1 and 3 (default 0.3).
        doji_ratio: Maximum body-to-range ratio for the doji Bar 2 (default 0.1).

    Returns:
        Boolean Series, ``True`` on Bar 3 (the confirming candle).
        The first two bars are always ``False``.
    """
    open_ = ohlc["open"]
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    open_2, high_2, low_2, close_2 = open_.shift(2), high.shift(2), low.shift(2), close.shift(2)
    open_1, high_1, low_1, close_1 = open_.shift(1), high.shift(1), low.shift(1), close.shift(1)

    # Bar 1: large bearish candle.
    bar1_bearish = (close_2 < open_2) & (
        _body_range_ratio(open_2, close_2, high_2, low_2) >= body_ratio
    )

    # Bar 2: doji with a strict gap below Bar 1 (no overlap between the two bars).
    bar2_doji = _body_range_ratio(open_1, close_1, high_1, low_1) <= doji_ratio
    gap_down_from_bar1 = high_1 < low_2  # Bar 2's high is strictly below Bar 1's low

    # Bar 3: large bullish candle with a strict gap above Bar 2.
    bar3_bullish = (close > open_) & (_body_range_ratio(open_, close, high, low) >= body_ratio)
    gap_up_from_bar2 = low > high_1  # Bar 3's low is strictly above Bar 2's high

    result = bar1_bearish & bar2_doji & gap_down_from_bar1 & bar3_bullish & gap_up_from_bar2
    return result.fill_null(False).alias("abandoned_baby_bullish")


def is_abandoned_baby_bearish(
    ohlc: pl.DataFrame,
    body_ratio: float = 0.3,
    doji_ratio: float = 0.1,
) -> pl.Series:
    """Detect bearish abandoned baby — gap-reversal three-candle top pattern.

    The bearish abandoned baby signals a potential top reversal when a doji
    "abandons" both neighbouring candles via strict price gaps:

        Bar 1 (i-2): Large bullish candle (body ≥ ``body_ratio`` × range).
        Bar 2 (i-1): Doji (body ≤ ``doji_ratio`` × range) with a full gap
                     *above* Bar 1 — Bar 2's low is strictly above Bar 1's high.
        Bar 3 (i):   Large bearish candle with a full gap *below* Bar 2 —
                     Bar 3's high is strictly below Bar 2's low.

    Both gaps must be strict (no price overlap) for the pattern to qualify.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        body_ratio: Minimum body-to-range ratio for Bars 1 and 3 (default 0.3).
        doji_ratio: Maximum body-to-range ratio for the doji Bar 2 (default 0.1).

    Returns:
        Boolean Series, ``True`` on Bar 3 (the confirming candle).
        The first two bars are always ``False``.
    """
    open_ = ohlc["open"]
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    open_2, high_2, low_2, close_2 = open_.shift(2), high.shift(2), low.shift(2), close.shift(2)
    open_1, high_1, low_1, close_1 = open_.shift(1), high.shift(1), low.shift(1), close.shift(1)

    # Bar 1: large bullish candle.
    bar1_bullish = (close_2 > open_2) & (
        _body_range_ratio(open_2, close_2, high_2, low_2) >= body_ratio
    )

    # Bar 2: doji with a strict gap above Bar 1.
    bar2_doji = _body_range_ratio(open_1, close_1, high_1, low_1) <= doji_ratio
    gap_up_from_bar1 = low_1 > high_2  # Bar 2's low is strictly above Bar 1's high

    # Bar 3: large bearish candle with a strict gap below Bar 2.
    bar3_bearish = (close < open_) & (_body_range_ratio(open_, close, high, low) >= body_ratio)
    gap_down_from_bar2 = high < low_1  # Bar 3's high is strictly below Bar 2's low

    result = bar1_bullish & bar2_doji & gap_up_from_bar1 & bar3_bearish & gap_down_from_bar2
    return result.fill_null(False).alias("abandoned_baby_bearish")


# ---------------------------------------------------------------------------
# Hanging Man / Inverted Hammer (trend-contextual pin bars)
# ---------------------------------------------------------------------------


def is_hanging_man(
    ohlc: pl.DataFrame,
    wick_ratio: float = 0.6,
    body_ratio: float = 0.25,
    trend_period: int = 5,
) -> pl.Series:
    """Detect hanging man candles — hammer structure after an uptrend.

    A hanging man has the same structural shape as a bullish pin bar
    (long lower wick, small body near the top of the range) but appears
    after a rising market, signalling a potential bearish reversal as
    sellers pushed price down before buyers recovered.

    Structure:
        - Long lower wick: ≥ ``wick_ratio`` × total range.
        - Small body: ≤ ``body_ratio`` × total range.
        - Prior uptrend: close[t] > close[t − trend_period].

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        wick_ratio: Minimum lower-wick-to-range ratio (default 0.6).
        body_ratio: Maximum body-to-range ratio (default 0.25).
        trend_period: Look-back bars to confirm a prior uptrend (default 5).

    Returns:
        Boolean Series, ``True`` on hanging man bars.
        The first ``trend_period`` bars are always ``False``.
    """
    candle_range = _range(ohlc)
    body = _body(ohlc)
    safe_range = candle_range.replace(0.0, float("nan"))

    lower_wick = pl.select(pl.min_horizontal(ohlc["open"], ohlc["close"])).to_series() - ohlc["low"]

    # Structural conditions — identical to pin_bar_bullish.
    pin_shape = (lower_wick / safe_range >= wick_ratio) & (body / safe_range <= body_ratio)

    # Trend context: current close must be above the close trend_period bars ago.
    prior_uptrend = ohlc["close"] > ohlc["close"].shift(trend_period)

    return (pin_shape & prior_uptrend).fill_null(False).alias("hanging_man")


def is_inverted_hammer(
    ohlc: pl.DataFrame,
    wick_ratio: float = 0.6,
    body_ratio: float = 0.25,
    trend_period: int = 5,
) -> pl.Series:
    """Detect inverted hammer candles — shooting-star structure after a downtrend.

    An inverted hammer has the same structural shape as a bearish pin bar
    (long upper wick, small body near the bottom of the range) but appears
    after a declining market, signalling a potential bullish reversal as
    buyers briefly pushed price up before sellers took control for that bar.

    Structure:
        - Long upper wick: ≥ ``wick_ratio`` × total range.
        - Small body: ≤ ``body_ratio`` × total range.
        - Prior downtrend: close[t] < close[t − trend_period].

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        wick_ratio: Minimum upper-wick-to-range ratio (default 0.6).
        body_ratio: Maximum body-to-range ratio (default 0.25).
        trend_period: Look-back bars to confirm a prior downtrend (default 5).

    Returns:
        Boolean Series, ``True`` on inverted hammer bars.
        The first ``trend_period`` bars are always ``False``.
    """
    candle_range = _range(ohlc)
    body = _body(ohlc)
    safe_range = candle_range.replace(0.0, float("nan"))

    upper_wick = (
        ohlc["high"] - pl.select(pl.max_horizontal(ohlc["open"], ohlc["close"])).to_series()
    )

    # Structural conditions — identical to pin_bar_bearish.
    pin_shape = (upper_wick / safe_range >= wick_ratio) & (body / safe_range <= body_ratio)

    # Trend context: current close must be below the close trend_period bars ago.
    prior_downtrend = ohlc["close"] < ohlc["close"].shift(trend_period)

    return (pin_shape & prior_downtrend).fill_null(False).alias("inverted_hammer")


# ---------------------------------------------------------------------------
# Tweezer Top / Tweezer Bottom
# ---------------------------------------------------------------------------


def is_tweezer_top(
    ohlc: pl.DataFrame,
    tolerance: float = 0.001,
    body_ratio: float = 0.3,
) -> pl.Series:
    """Detect tweezer top — two consecutive bars with equal highs (bearish reversal).

    The tweezer top signals a failed breakout: the first bar is bullish and
    the second is bearish, both with approximately equal highs.  The shared
    high acts as resistance, and the rejection of the level by the second
    bearish bar suggests the uptrend is stalling.

    Conditions:
        Bar 1 (i−1): Bullish candle with substantial body.
        Bar 2 (i):   Bearish candle with substantial body.
        |high[i] − high[i−1]| / high[i−1] ≤ tolerance  (equal highs).

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        tolerance: Maximum relative difference between the two highs (default 0.001).
        body_ratio: Minimum body-to-range ratio for both candles (default 0.3).

    Returns:
        Boolean Series, ``True`` on Bar 2.  The first bar is always ``False``.
    """
    prev_open = ohlc["open"].shift(1)
    prev_high = ohlc["high"].shift(1)
    prev_close = ohlc["close"].shift(1)
    prev_low = ohlc["low"].shift(1)

    # Bar 1 must be a substantial bullish candle.
    bar1_bullish = (prev_close > prev_open) & (
        _body_range_ratio(prev_open, prev_close, prev_high, prev_low) >= body_ratio
    )

    # Bar 2 must be a substantial bearish candle.
    curr_bearish = (ohlc["close"] < ohlc["open"]) & (
        _body_range_ratio(ohlc["open"], ohlc["close"], ohlc["high"], ohlc["low"]) >= body_ratio
    )

    # Highs are approximately equal (relative tolerance guards against price-scale issues).
    equal_highs = (
        (ohlc["high"] - prev_high).abs() / prev_high.abs().replace(0.0, float("nan"))
    ) <= tolerance

    return (bar1_bullish & curr_bearish & equal_highs).fill_null(False).alias("tweezer_top")


def is_tweezer_bottom(
    ohlc: pl.DataFrame,
    tolerance: float = 0.001,
    body_ratio: float = 0.3,
) -> pl.Series:
    """Detect tweezer bottom — two consecutive bars with equal lows (bullish reversal).

    The tweezer bottom signals a double test of support: the first bar is
    bearish and the second is bullish, both with approximately equal lows.
    The shared low acts as support, and the bullish recovery on the second
    bar suggests the downtrend is losing momentum.

    Conditions:
        Bar 1 (i−1): Bearish candle with substantial body.
        Bar 2 (i):   Bullish candle with substantial body.
        |low[i] − low[i−1]| / |low[i−1]| ≤ tolerance  (equal lows).

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        tolerance: Maximum relative difference between the two lows (default 0.001).
        body_ratio: Minimum body-to-range ratio for both candles (default 0.3).

    Returns:
        Boolean Series, ``True`` on Bar 2.  The first bar is always ``False``.
    """
    prev_open = ohlc["open"].shift(1)
    prev_high = ohlc["high"].shift(1)
    prev_close = ohlc["close"].shift(1)
    prev_low = ohlc["low"].shift(1)

    # Bar 1 must be a substantial bearish candle.
    bar1_bearish = (prev_close < prev_open) & (
        _body_range_ratio(prev_open, prev_close, prev_high, prev_low) >= body_ratio
    )

    # Bar 2 must be a substantial bullish candle.
    curr_bullish = (ohlc["close"] > ohlc["open"]) & (
        _body_range_ratio(ohlc["open"], ohlc["close"], ohlc["high"], ohlc["low"]) >= body_ratio
    )

    # Lows are approximately equal.
    equal_lows = (
        (ohlc["low"] - prev_low).abs() / prev_low.abs().replace(0.0, float("nan"))
    ) <= tolerance

    return (bar1_bearish & curr_bullish & equal_lows).fill_null(False).alias("tweezer_bottom")


# ---------------------------------------------------------------------------
# Dark Cloud Cover / Piercing Line
# ---------------------------------------------------------------------------


def is_dark_cloud_cover(
    ohlc: pl.DataFrame,
    body_ratio: float = 0.3,
    penetration: float = 0.5,
) -> pl.Series:
    """Detect dark cloud cover — two-bar bearish reversal pattern.

    The dark cloud cover signals a potential top reversal:

        Bar 1 (i−1): Substantial bullish candle (close > open).
        Bar 2 (i):   Opens above Bar 1's high, then closes below the midpoint
                     of Bar 1's body — demonstrating strong selling pressure.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        body_ratio: Minimum body-to-range ratio for both bars (default 0.3).
        penetration: Minimum fraction of Bar 1's body that Bar 2 must penetrate
                     (default 0.5 = must close below Bar 1's midpoint).

    Returns:
        Boolean Series, ``True`` on Bar 2.  The first bar is always ``False``.
    """
    prev_open = ohlc["open"].shift(1)
    prev_high = ohlc["high"].shift(1)
    prev_close = ohlc["close"].shift(1)
    prev_low = ohlc["low"].shift(1)

    # Bar 1: substantial bullish candle.
    bar1_bullish = (prev_close > prev_open) & (
        _body_range_ratio(prev_open, prev_close, prev_high, prev_low) >= body_ratio
    )

    # Bar 2: substantial bearish candle.
    bar2_bearish = (ohlc["close"] < ohlc["open"]) & (
        _body_range_ratio(ohlc["open"], ohlc["close"], ohlc["high"], ohlc["low"]) >= body_ratio
    )

    # Bar 2 opens above Bar 1's high (gap-up open).
    opens_above = ohlc["open"] > prev_high

    # Bar 2 closes below the penetration level into Bar 1's body.
    penetration_level = prev_close - (prev_close - prev_open) * penetration
    closes_inside = ohlc["close"] < penetration_level

    result = bar1_bullish & bar2_bearish & opens_above & closes_inside
    return result.fill_null(False).alias("dark_cloud_cover")


def is_piercing_line(
    ohlc: pl.DataFrame,
    body_ratio: float = 0.3,
    penetration: float = 0.5,
) -> pl.Series:
    """Detect piercing line — two-bar bullish reversal pattern.

    The piercing line is the bullish mirror of the dark cloud cover:

        Bar 1 (i−1): Substantial bearish candle (close < open).
        Bar 2 (i):   Opens below Bar 1's low, then closes above the midpoint
                     of Bar 1's body — demonstrating strong buying recovery.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        body_ratio: Minimum body-to-range ratio for both bars (default 0.3).
        penetration: Minimum fraction of Bar 1's body that Bar 2 must recover
                     (default 0.5 = must close above Bar 1's midpoint).

    Returns:
        Boolean Series, ``True`` on Bar 2.  The first bar is always ``False``.
    """
    prev_open = ohlc["open"].shift(1)
    prev_high = ohlc["high"].shift(1)
    prev_close = ohlc["close"].shift(1)
    prev_low = ohlc["low"].shift(1)

    # Bar 1: substantial bearish candle.
    bar1_bearish = (prev_close < prev_open) & (
        _body_range_ratio(prev_open, prev_close, prev_high, prev_low) >= body_ratio
    )

    # Bar 2: substantial bullish candle.
    bar2_bullish = (ohlc["close"] > ohlc["open"]) & (
        _body_range_ratio(ohlc["open"], ohlc["close"], ohlc["high"], ohlc["low"]) >= body_ratio
    )

    # Bar 2 opens below Bar 1's low (gap-down open).
    opens_below = ohlc["open"] < prev_low

    # Bar 2 closes above the penetration level into Bar 1's body.
    penetration_level = prev_close + (prev_open - prev_close) * penetration
    closes_inside = ohlc["close"] > penetration_level

    result = bar1_bearish & bar2_bullish & opens_below & closes_inside
    return result.fill_null(False).alias("piercing_line")


# ---------------------------------------------------------------------------
# Rising Three Methods / Falling Three Methods
# ---------------------------------------------------------------------------


def is_rising_three_methods(
    ohlc: pl.DataFrame,
    body_ratio: float = 0.3,
    small_body_ratio: float = 0.3,
) -> pl.Series:
    """Detect rising three methods — five-bar bullish continuation pattern.

    The rising three methods signals that a brief consolidation within an
    uptrend has ended and the trend is resuming:

        Bar 1 (i−4): Large bullish candle.
        Bars 2–4 (i−3 to i−1): Three small bearish candles, all with closes
                     and opens within Bar 1's body range.
        Bar 5 (i):   Large bullish candle closing above Bar 1's close.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        body_ratio: Minimum body-to-range ratio for Bars 1 and 5 (default 0.3).
        small_body_ratio: Maximum body-to-range ratio for the three middle
                          candles (default 0.3).

    Returns:
        Boolean Series, ``True`` on Bar 5.  The first four bars are always ``False``.
    """
    o = ohlc["open"]
    h = ohlc["high"]
    lo = ohlc["low"]
    c = ohlc["close"]

    # Bar 1 (4 bars ago): large bullish.
    o1, h1, l1, c1 = o.shift(4), h.shift(4), lo.shift(4), c.shift(4)
    bar1_bull = (c1 > o1) & (_body_range_ratio(o1, c1, h1, l1) >= body_ratio)

    # Bars 2–4 (i−3 to i−1): small bearish candles contained within Bar 1's body.
    middle_ok = pl.Series([True] * len(o))
    for shift in (3, 2, 1):
        os, hs, ls, cs = o.shift(shift), h.shift(shift), lo.shift(shift), c.shift(shift)
        bar_small = (cs < os) & (_body_range_ratio(os, cs, hs, ls) <= small_body_ratio)
        # Body spans [o1, c1] for bullish bar1; closes above o1 and opens below c1.
        middle_ok = middle_ok & bar_small & (cs >= o1) & (os <= c1)

    # Bar 5 (current): large bullish closing above Bar 1's close.
    bar5_bull = (c > o) & (_body_range_ratio(o, c, h, lo) >= body_ratio) & (c > c1)

    return (
        (bar1_bull & middle_ok & bar5_bull)
        .fill_null(False)
        .alias(  # type: ignore[possibly-undefined]
            "rising_three_methods"
        )
    )


def is_falling_three_methods(
    ohlc: pl.DataFrame,
    body_ratio: float = 0.3,
    small_body_ratio: float = 0.3,
) -> pl.Series:
    """Detect falling three methods — five-bar bearish continuation pattern.

    The falling three methods signals that a brief consolidation within a
    downtrend has ended and the bearish trend is resuming:

        Bar 1 (i−4): Large bearish candle.
        Bars 2–4 (i−3 to i−1): Three small bullish candles, all with closes
                     and opens within Bar 1's body range.
        Bar 5 (i):   Large bearish candle closing below Bar 1's close.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        body_ratio: Minimum body-to-range ratio for Bars 1 and 5 (default 0.3).
        small_body_ratio: Maximum body-to-range ratio for the three middle
                          candles (default 0.3).

    Returns:
        Boolean Series, ``True`` on Bar 5.  The first four bars are always ``False``.
    """
    o = ohlc["open"]
    h = ohlc["high"]
    lo = ohlc["low"]
    c = ohlc["close"]

    # Bar 1 (4 bars ago): large bearish.
    o1, h1, l1, c1 = o.shift(4), h.shift(4), lo.shift(4), c.shift(4)
    bar1_bear = (c1 < o1) & (_body_range_ratio(o1, c1, h1, l1) >= body_ratio)

    # Bars 2–4: small bullish candles contained within Bar 1's body.
    middle_ok = pl.Series([True] * len(o))
    for shift in (3, 2, 1):
        os, hs, ls, cs = o.shift(shift), h.shift(shift), lo.shift(shift), c.shift(shift)
        bar_small = (cs > os) & (_body_range_ratio(os, cs, hs, ls) <= small_body_ratio)
        # Body spans [c1, o1] for bearish bar1; closes below o1 and opens above c1.
        middle_ok = middle_ok & bar_small & (cs <= o1) & (os >= c1)

    # Bar 5 (current): large bearish closing below Bar 1's close.
    bar5_bear = (c < o) & (_body_range_ratio(o, c, h, lo) >= body_ratio) & (c < c1)

    return (
        (bar1_bear & middle_ok & bar5_bear)
        .fill_null(False)
        .alias(  # type: ignore[possibly-undefined]
            "falling_three_methods"
        )
    )
