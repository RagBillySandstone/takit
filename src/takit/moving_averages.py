"""
Moving average indicators.

All functions accept a ``pl.Series`` and return a ``pl.Series``.
The first ``(period - 1)`` values of any windowed output are ``null``
rather than zero, preserving null-propagation semantics in Polars.

Functions
---------
sma             Simple Moving Average
ema             Exponential Moving Average  (α = 2 / (n + 1))
wma             Weighted Moving Average     (linearly weighted)
wilder_smooth   Wilder's Smoothing / RMA    (α = 1 / n)
dema            Double EMA                  (2·EMA - EMA(EMA))
tema            Triple EMA                  (3·EMA - 3·EMA(EMA) + EMA(EMA(EMA)))
"""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_period(period: int, name: str, min_period: int = 1) -> None:
    """Raise ``ValueError`` when *period* is below *min_period*.

    Args:
        period: The period value supplied by the caller.
        name: Indicator name used in the error message.
        min_period: Minimum acceptable value (default 1).

    Raises:
        ValueError: If ``period < min_period``.
    """
    if period < min_period:
        raise ValueError(f"{name} period must be at least {min_period}, got {period}.")


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


def sma(series: pl.Series, period: int) -> pl.Series:
    """Simple Moving Average over a rolling window.

    Computes the arithmetic mean of the last *period* values.  The first
    ``period - 1`` values are ``null``.

    Args:
        series: Input price series.
        period: Lookback window (number of bars).

    Returns:
        Series of SMA values aligned with the input.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "SMA")
    return series.rolling_mean(window_size=period, min_samples=period).alias(f"sma_{period}")


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def ema(series: pl.Series, period: int) -> pl.Series:
    """Exponential Moving Average using the standard smoothing factor.

    Smoothing factor: α = 2 / (period + 1).  The series is seeded with
    the SMA of the first *period* bars (``adjust=False, min_periods=period``).
    Values before the seed point are ``null``.

    Args:
        series: Input price series.
        period: Lookback period — larger values produce more smoothing.

    Returns:
        Series of EMA values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "EMA")
    alpha = 2.0 / (period + 1)
    return series.ewm_mean(
        alpha=alpha,
        adjust=False,
        min_samples=period,
    ).alias(f"ema_{period}")


# ---------------------------------------------------------------------------
# WMA
# ---------------------------------------------------------------------------


def wma(series: pl.Series, period: int) -> pl.Series:
    """Weighted Moving Average with linearly increasing weights.

    Each bar within the window receives a weight proportional to its
    position: the most recent bar has weight *period*, the oldest has
    weight 1.  The sum of weights for a window of size *n* is
    ``n * (n + 1) / 2``.

    The first ``period - 1`` values are ``null``.

    Args:
        series: Input price series.
        period: Lookback window.

    Returns:
        Series of WMA values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "WMA")

    weights = list(range(1, period + 1))
    weight_sum = sum(weights)

    # Sum shifted copies of the series, each multiplied by its linear weight.
    # shift(k) moves the series forward k bars, so shift(0) = most recent,
    # shift(period-1) = oldest bar in the window.
    weighted = sum(series.shift(period - 1 - i) * weights[i] for i in range(period))

    # Null out the warm-up period to match the convention of every other indicator.
    null_prefix: list[float | None] = [None] * (period - 1)
    ones_suffix: list[float | None] = [1.0] * (len(series) - (period - 1))
    mask = pl.Series(null_prefix + ones_suffix)
    return ((weighted / weight_sum) * mask).alias(f"wma_{period}")


# ---------------------------------------------------------------------------
# Wilder's Smoothing (RMA)
# ---------------------------------------------------------------------------


def wilder_smooth(series: pl.Series, period: int) -> pl.Series:
    """Wilder's Smoothing, also known as the Running Moving Average (RMA).

    Uses a smoothing factor of α = 1 / period, which is more conservative
    than standard EMA (α = 2 / (n + 1)).  Used internally by RSI and ATR
    to match Wilder's original formulas.

    Args:
        series: Input series.
        period: Lookback period.

    Returns:
        Series of Wilder-smoothed values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Wilder smooth")
    alpha = 1.0 / period
    return series.ewm_mean(
        alpha=alpha,
        adjust=False,
        min_samples=period,
    ).alias(f"wilder_{period}")


# ---------------------------------------------------------------------------
# DEMA
# ---------------------------------------------------------------------------


def dema(series: pl.Series, period: int) -> pl.Series:
    """Double Exponential Moving Average.

    DEMA reduces the lag of a standard EMA by applying a correction:

        DEMA = 2 · EMA(series, n) − EMA(EMA(series, n), n)

    The warm-up period is ``2 * (period - 1)`` bars, after which both the
    outer EMA and the EMA-of-EMA are defined.

    Args:
        series: Input price series.
        period: EMA period.

    Returns:
        Series of DEMA values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "DEMA")
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    return (2.0 * ema1 - ema2).alias(f"dema_{period}")


# ---------------------------------------------------------------------------
# TEMA
# ---------------------------------------------------------------------------


def tema(series: pl.Series, period: int) -> pl.Series:
    """Triple Exponential Moving Average.

    Further reduces lag compared to DEMA:

        TEMA = 3 · EMA(series, n)
             − 3 · EMA(EMA(series, n), n)
             +     EMA(EMA(EMA(series, n), n), n)

    The warm-up period is ``3 * (period - 1)`` bars.

    Args:
        series: Input price series.
        period: EMA period.

    Returns:
        Series of TEMA values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "TEMA")
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return (3.0 * ema1 - 3.0 * ema2 + ema3).alias(f"tema_{period}")
