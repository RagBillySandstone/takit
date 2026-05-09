"""
Price-level indicators: pivot points and support/resistance levels.

All functions accept a prior-session OHLC summary (a single set of values,
not a rolling window) and return a ``pl.DataFrame`` of level Series aligned
with the input length.

Typical usage: compute the prior daily session's OHLC, then broadcast the
resulting pivot levels as constant columns across all intraday bars in that
session.

Functions
---------
pivot_points_floor      Classic floor-trader pivot points (PP, S1-S3, R1-R3)
pivot_points_camarilla  Camarilla pivot points (S1-S4, R1-R4)
pivot_points_fibonacci  Fibonacci pivot points (PP ± 0.382/0.618/1.0 × range)
pivot_points_woodie     Woodie pivot points (weights close more than floor)
pivot_points_demark     DeMark pivot points (conditional on open vs. close)
"""

from __future__ import annotations

import polars as pl


def pivot_points_floor(
    prev_high: pl.Series,
    prev_low: pl.Series,
    prev_close: pl.Series,
) -> pl.DataFrame:
    """Classic floor-trader pivot points.

    Computes seven price levels from the prior session's high, low, and close.
    These levels are widely used as intraday support and resistance.

    Formulas:
        PP  = (prev_high + prev_low + prev_close) / 3
        R1  = 2 × PP − prev_low
        R2  = PP + (prev_high − prev_low)
        R3  = prev_high + 2 × (PP − prev_low)
        S1  = 2 × PP − prev_high
        S2  = PP − (prev_high − prev_low)
        S3  = prev_low − 2 × (prev_high − PP)

    Args:
        prev_high:  Prior-session high series (one value per current-session bar,
                    or a scalar broadcast — caller controls the alignment).
        prev_low:   Prior-session low series.
        prev_close: Prior-session close series.

    Returns:
        DataFrame with columns ``pp``, ``r1``, ``r2``, ``r3``,
        ``s1``, ``s2``, ``s3``.
    """
    pp = (prev_high + prev_low + prev_close) / 3.0
    hl_range = prev_high - prev_low

    r1 = 2.0 * pp - prev_low
    r2 = pp + hl_range
    r3 = prev_high + 2.0 * (pp - prev_low)

    s1 = 2.0 * pp - prev_high
    s2 = pp - hl_range
    s3 = prev_low - 2.0 * (prev_high - pp)

    return pl.DataFrame({"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3})


def pivot_points_camarilla(
    prev_high: pl.Series,
    prev_low: pl.Series,
    prev_close: pl.Series,
) -> pl.DataFrame:
    """Camarilla pivot points.

    Camarilla pivots weight price action more heavily toward the close and
    produce tighter, intraday-focused levels.  R3/S3 are commonly used for
    mean-reversion scalps; R4/S4 are breakout levels.

    Formulas (Camarilla equation with multiplier 1.1):
        R1  = prev_close + range × (1.1 / 12)
        R2  = prev_close + range × (1.1 / 6)
        R3  = prev_close + range × (1.1 / 4)
        R4  = prev_close + range × (1.1 / 2)
        S1  = prev_close − range × (1.1 / 12)
        S2  = prev_close − range × (1.1 / 6)
        S3  = prev_close − range × (1.1 / 4)
        S4  = prev_close − range × (1.1 / 2)

    where range = prev_high − prev_low.

    Args:
        prev_high:  Prior-session high series.
        prev_low:   Prior-session low series.
        prev_close: Prior-session close series.

    Returns:
        DataFrame with columns ``cam_r1`` … ``cam_r4``, ``cam_s1`` … ``cam_s4``.
    """
    hl_range = prev_high - prev_low

    r1 = prev_close + hl_range * (1.1 / 12)
    r2 = prev_close + hl_range * (1.1 / 6)
    r3 = prev_close + hl_range * (1.1 / 4)
    r4 = prev_close + hl_range * (1.1 / 2)

    s1 = prev_close - hl_range * (1.1 / 12)
    s2 = prev_close - hl_range * (1.1 / 6)
    s3 = prev_close - hl_range * (1.1 / 4)
    s4 = prev_close - hl_range * (1.1 / 2)

    return pl.DataFrame(
        {
            "cam_r1": r1,
            "cam_r2": r2,
            "cam_r3": r3,
            "cam_r4": r4,
            "cam_s1": s1,
            "cam_s2": s2,
            "cam_s3": s3,
            "cam_s4": s4,
        }
    )


def pivot_points_fibonacci(
    prev_high: pl.Series,
    prev_low: pl.Series,
    prev_close: pl.Series,
) -> pl.DataFrame:
    """Fibonacci pivot points.

    Extends the classic floor pivot with Fibonacci-ratio levels instead of
    simple multiples.  The 0.382, 0.618, and 1.000 ratios correspond to
    the three main Fibonacci retracement levels.

    Formulas:
        PP  = (prev_high + prev_low + prev_close) / 3
        R1  = PP + 0.382 × (prev_high − prev_low)
        R2  = PP + 0.618 × (prev_high − prev_low)
        R3  = PP + 1.000 × (prev_high − prev_low)
        S1  = PP − 0.382 × (prev_high − prev_low)
        S2  = PP − 0.618 × (prev_high − prev_low)
        S3  = PP − 1.000 × (prev_high − prev_low)

    Args:
        prev_high:  Prior-session high series.
        prev_low:   Prior-session low series.
        prev_close: Prior-session close series.

    Returns:
        DataFrame with columns ``fib_pp``, ``fib_r1``, ``fib_r2``, ``fib_r3``,
        ``fib_s1``, ``fib_s2``, ``fib_s3``.
    """
    pp = (prev_high + prev_low + prev_close) / 3.0
    hl_range = prev_high - prev_low

    r1 = pp + 0.382 * hl_range
    r2 = pp + 0.618 * hl_range
    r3 = pp + 1.000 * hl_range

    s1 = pp - 0.382 * hl_range
    s2 = pp - 0.618 * hl_range
    s3 = pp - 1.000 * hl_range

    return pl.DataFrame(
        {
            "fib_pp": pp,
            "fib_r1": r1,
            "fib_r2": r2,
            "fib_r3": r3,
            "fib_s1": s1,
            "fib_s2": s2,
            "fib_s3": s3,
        }
    )


def pivot_points_woodie(
    prev_high: pl.Series,
    prev_low: pl.Series,
    prev_close: pl.Series,
) -> pl.DataFrame:
    """Woodie pivot points — weights the close more than the floor formula.

    Woodie's formula uses a 4-component pivot that double-counts the prior
    close, making the PP more responsive to the closing price.  Support and
    resistance levels are simpler (only two tiers) compared to floor pivots.

    Formulas:
        PP = (prev_high + prev_low + 2 × prev_close) / 4
        R1 = 2 × PP − prev_low
        R2 = PP + (prev_high − prev_low)
        S1 = 2 × PP − prev_high
        S2 = PP − (prev_high − prev_low)

    Args:
        prev_high:  Prior-session high series.
        prev_low:   Prior-session low series.
        prev_close: Prior-session close series.

    Returns:
        DataFrame with columns ``wood_pp``, ``wood_r1``, ``wood_r2``,
        ``wood_s1``, ``wood_s2``.
    """
    pp = (prev_high + prev_low + 2.0 * prev_close) / 4.0
    hl_range = prev_high - prev_low

    r1 = 2.0 * pp - prev_low
    r2 = pp + hl_range
    s1 = 2.0 * pp - prev_high
    s2 = pp - hl_range

    return pl.DataFrame(
        {
            "wood_pp": pp,
            "wood_r1": r1,
            "wood_r2": r2,
            "wood_s1": s1,
            "wood_s2": s2,
        }
    )


def pivot_points_demark(
    prev_open: pl.Series,
    prev_high: pl.Series,
    prev_low: pl.Series,
    prev_close: pl.Series,
) -> pl.DataFrame:
    """DeMark pivot points — conditional on prior open vs. close relationship.

    Tom DeMark's method adapts the pivot formula based on whether the prior
    session closed above, below, or equal to its open.  This produces a
    single resistance (R1) and support (S1) level rather than multiple tiers.

    Formula (conditional on X):
        If prev_close < prev_open:  X = prev_high + 2 × prev_low + prev_close
        If prev_close > prev_open:  X = 2 × prev_high + prev_low + prev_close
        If prev_close == prev_open: X = prev_high + prev_low + 2 × prev_close

        PP = X / 4
        R1 = X / 2 − prev_low
        S1 = X / 2 − prev_high

    Args:
        prev_open:  Prior-session open series.
        prev_high:  Prior-session high series.
        prev_low:   Prior-session low series.
        prev_close: Prior-session close series.

    Returns:
        DataFrame with columns ``dm_pp``, ``dm_r1``, ``dm_s1``.
    """
    # Compute all three X variants and select based on open/close relationship.
    x_bearish = prev_high + 2.0 * prev_low + prev_close  # close < open
    x_bullish = 2.0 * prev_high + prev_low + prev_close  # close > open
    x_neutral = prev_high + prev_low + 2.0 * prev_close  # close == open

    # pl.when/then with Series input requires pl.select to materialise.
    x = pl.select(
        pl.when(prev_close < prev_open)
        .then(x_bearish)
        .when(prev_close > prev_open)
        .then(x_bullish)
        .otherwise(x_neutral)
    ).to_series()

    pp = x / 4.0
    r1 = x / 2.0 - prev_low
    s1 = x / 2.0 - prev_high

    return pl.DataFrame({"dm_pp": pp, "dm_r1": r1, "dm_s1": s1})
