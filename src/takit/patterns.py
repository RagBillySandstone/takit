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
is_bullish_engulfing    Bearish candle followed by a larger bullish candle
is_bearish_engulfing    Bullish candle followed by a larger bearish candle
is_pin_bar_bullish      Hammer: small body near top, long lower wick
is_pin_bar_bearish      Shooting star: small body near bottom, long upper wick
is_inside_bar           Current bar's range is fully contained by the prior bar
is_doji                 Open ≈ Close (indecision candle)
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

    # Lower wick = min(open, close) - low.
    lower_wick = (
        pl.Series(
            [
                min(o, c)
                for o, c in zip(ohlc["open"].to_list(), ohlc["close"].to_list(), strict=True)
            ]
        )
        - ohlc["low"]
    )

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

    # Upper wick = high - max(open, close).
    upper_wick = ohlc["high"] - pl.Series(
        [max(o, c) for o, c in zip(ohlc["open"].to_list(), ohlc["close"].to_list(), strict=True)]
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
