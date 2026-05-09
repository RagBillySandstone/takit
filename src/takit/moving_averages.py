"""
Moving average indicators.

All functions accept a ``pl.Series`` and return a ``pl.Series``.
The first ``(period - 1)`` values of any windowed output are ``null``
rather than zero, preserving null-propagation semantics in Polars.

Functions
---------
sma               Simple Moving Average
ema               Exponential Moving Average  (α = 2 / (n + 1))
wma               Weighted Moving Average     (linearly weighted)
wilder_smooth     Wilder's Smoothing / RMA    (α = 1 / n)
dema              Double EMA                  (2·EMA - EMA(EMA))
tema              Triple EMA                  (3·EMA - 3·EMA(EMA) + EMA(EMA(EMA)))
hma               Hull Moving Average         (WMA of 2·WMA(n/2) - WMA(n))
vwma              Volume-Weighted Moving Average
mcginley_dynamic  McGinley Dynamic            (self-adjusting, tracks price closely)
"""

from __future__ import annotations

import operator
from functools import reduce

import polars as pl

from takit._validate import _validate_period

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
    # shift(period-1) = oldest bar in the window.  Null propagates naturally
    # through the summation — no explicit warm-up mask is needed.
    weighted: pl.Series = reduce(
        operator.add,
        (series.shift(period - 1 - i) * weights[i] for i in range(period)),
    )
    return (weighted / weight_sum).alias(f"wma_{period}")


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


# ---------------------------------------------------------------------------
# HMA
# ---------------------------------------------------------------------------


def hma(series: pl.Series, period: int) -> pl.Series:
    """Hull Moving Average — reduces lag while remaining smooth.

    HMA applies a WMA over a derived series that halves conventional lag:

        raw      = 2 · WMA(series, n/2) − WMA(series, n)
        HMA(n)   = WMA(raw, √n)

    The ``n/2`` half-period uses integer division; the smoothing window
    uses ``round(√n)``.  The warm-up is ``(period - 1) + (round(√period) - 1)``
    bars, after which both underlying WMAs and the final WMA are defined.

    Args:
        series: Input price series.
        period: Primary lookback window.

    Returns:
        Series of HMA values.

    Raises:
        ValueError: If ``period < 2`` (need at least two bars for a half-period).
    """
    _validate_period(period, "HMA", min_period=2)

    half_period = period // 2
    sqrt_period = round(period**0.5)

    # The intermediate series amplifies the short-term WMA to reduce lag.
    raw = 2.0 * wma(series, half_period) - wma(series, period)

    return wma(raw, sqrt_period).alias(f"hma_{period}")


# ---------------------------------------------------------------------------
# VWMA
# ---------------------------------------------------------------------------


def vwma(price: pl.Series, volume: pl.Series, period: int) -> pl.Series:
    """Volume-Weighted Moving Average.

    Each bar within the window is weighted by its volume rather than by
    its position (as in WMA) or equally (as in SMA).  VWMA tracks price
    more closely during high-volume bars and gives less weight to
    low-volume noise.

        VWMA(n) = Σ(price[i] × volume[i]) / Σ(volume[i])   over the last n bars

    The first ``period - 1`` values are ``null``.

    Args:
        price: Price series (e.g. close).
        volume: Volume series aligned bar-for-bar with *price*.
        period: Lookback window.

    Returns:
        Series of VWMA values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "VWMA")

    pv_sum = (price * volume.cast(pl.Float64)).rolling_sum(window_size=period, min_samples=period)
    vol_sum = volume.cast(pl.Float64).rolling_sum(window_size=period, min_samples=period)

    # fill_nan handles the edge case of a window with zero total volume.
    return (pv_sum / vol_sum).fill_nan(None).alias(f"vwma_{period}")


# ---------------------------------------------------------------------------
# McGinley Dynamic
# ---------------------------------------------------------------------------


def mcginley_dynamic(series: pl.Series, period: int) -> pl.Series:
    """McGinley Dynamic — self-adjusting moving average.

    The McGinley Dynamic adapts its speed to market velocity so that it
    stays closer to price during fast moves and is smoother during slow
    ones.  Unlike EMA or WMA it does not require the trader to choose a
    period that matches the current market rhythm — the formula
    self-corrects:

        MD[t] = MD[t-1] + (price - MD[t-1]) / (N × (price / MD[t-1])⁴)

    The indicator is seeded with the SMA of the first *period* bars.
    Values before the seed point are ``null``.

    Args:
        series: Input price series.
        period: Nominal period (controls the denominator constant N).

    Returns:
        Series of McGinley Dynamic values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "McGinley Dynamic")

    raw_values: list[float | None] = series.to_list()
    output: list[float | None] = [None] * len(raw_values)

    # Seed with the SMA of the first period bars; skip if any input is null.
    seed_window = [v for v in raw_values[:period] if v is not None]
    if len(seed_window) < period:
        # Not enough non-null data to seed — return all-null series.
        return pl.Series(f"mcginley_{period}", output, dtype=pl.Float64)

    md = sum(seed_window) / period
    output[period - 1] = md

    for idx in range(period, len(raw_values)):
        price = raw_values[idx]
        if price is None or md == 0.0:
            # Propagate null on missing data or degenerate seed.
            output[idx] = None
            continue
        # The (price/MD)^4 term makes the denominator larger (slower) when
        # price is far above MD and smaller (faster) when price is far below.
        md = md + (price - md) / (period * (price / md) ** 4)
        output[idx] = md

    return pl.Series(f"mcginley_{period}", output, dtype=pl.Float64)
