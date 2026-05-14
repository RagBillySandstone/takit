"""
General-purpose signal and return utilities.

Functions
---------
crossover       Bullish cross: fast series crosses above slow series
crossunder      Bearish cross: fast series crosses below slow series
log_returns     Bar-to-bar log returns: ln(price[t] / price[t-1])
simple_returns  Bar-to-bar simple returns: (price[t] - price[t-1]) / price[t-1]
rolling_highest Rolling n-period maximum
rolling_lowest  Rolling n-period minimum
rolling_std     Rolling n-period sample standard deviation
percent_rank    Rolling percentile rank of current value within last n bars
"""

from __future__ import annotations

import math

import polars as pl

from polarticks._validate import _validate_period


def crossover(fast: pl.Series, slow: pl.Series, atol: float = 0.0) -> pl.Series:
    """Detect a bullish crossover: fast crosses above slow.

    Returns ``True`` on the single bar where ``fast`` transitions from being
    at or below ``slow`` to strictly above it.  All other bars are ``False``.

    The optional *atol* parameter handles floating-point equality at the cross
    point.  A bar is considered "above" only when ``fast - slow > atol``, so
    tiny numerical noise near the crossing level cannot trigger a spurious
    double-signal.  Leave at the default ``0.0`` for exact comparison.

    Args:
        fast: The faster (more responsive) series.
        slow: The slower (less responsive) series.
        atol: Absolute tolerance for equality (default 0.0).

    Returns:
        Boolean Series, ``True`` only on the crossover bar.
    """
    diff = fast - slow
    # Treat values within atol of zero as "not yet above".
    above_now = diff > atol
    # Use a large negative sentinel for the null on bar 0 so the first bar
    # can legitimately fire as a crossover if it starts above.
    above_prev = diff.shift(1).fill_null(-1.0) > atol
    return (above_now & ~above_prev).alias("crossover")


def crossunder(fast: pl.Series, slow: pl.Series, atol: float = 0.0) -> pl.Series:
    """Detect a bearish crossunder: fast crosses below slow.

    Returns ``True`` on the single bar where ``fast`` transitions from being
    at or above ``slow`` to strictly below it.

    The optional *atol* parameter mirrors the behaviour of :func:`crossover`:
    a bar is "below" only when ``slow - fast > atol``.

    Args:
        fast: The faster (more responsive) series.
        slow: The slower (less responsive) series.
        atol: Absolute tolerance for equality (default 0.0).

    Returns:
        Boolean Series, ``True`` only on the crossunder bar.
    """
    diff = slow - fast
    below_now = diff > atol
    below_prev = diff.shift(1).fill_null(-1.0) > atol
    return (below_now & ~below_prev).alias("crossunder")


def log_returns(series: pl.Series) -> pl.Series:
    """Compute bar-to-bar log returns: ln(price[t] / price[t-1]).

    Log returns are preferred in statistical research because they are
    time-additive and more normally distributed than simple returns.
    The first value is ``null``.

    Args:
        series: Price series.

    Returns:
        Series of log return values.
    """
    return (series / series.shift(1)).log(base=math.e).alias("log_returns")


def simple_returns(series: pl.Series) -> pl.Series:
    """Compute bar-to-bar simple (arithmetic) returns.

    Simple return = (price[t] - price[t-1]) / price[t-1].
    The first value is ``null``.

    Args:
        series: Price series.

    Returns:
        Series of simple return values.
    """
    return series.pct_change().alias("simple_returns")


# ---------------------------------------------------------------------------
# Rolling Highest
# ---------------------------------------------------------------------------


def rolling_highest(series: pl.Series, period: int) -> pl.Series:
    """Rolling n-period maximum (highest value in the lookback window).

    A direct thin wrapper around Polars ``rolling_max`` that enforces the
    library's null-prefix contract (``period − 1`` leading nulls) and adds
    a consistent alias.  Useful as a standalone building block when the
    caller needs the raw high series without bundling it into a wider
    indicator output.

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Input series (e.g., high prices).
        period: Lookback window length.

    Returns:
        Series of rolling maximum values named ``highest_{period}``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Rolling Highest")
    return series.rolling_max(window_size=period, min_samples=period).alias(f"highest_{period}")


# ---------------------------------------------------------------------------
# Rolling Lowest
# ---------------------------------------------------------------------------


def rolling_lowest(series: pl.Series, period: int) -> pl.Series:
    """Rolling n-period minimum (lowest value in the lookback window).

    Mirror of :func:`rolling_highest` for the rolling minimum.  Enforces the
    ``period − 1`` leading-null contract via ``min_samples=period``.

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Input series (e.g., low prices).
        period: Lookback window length.

    Returns:
        Series of rolling minimum values named ``lowest_{period}``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Rolling Lowest")
    return series.rolling_min(window_size=period, min_samples=period).alias(f"lowest_{period}")


# ---------------------------------------------------------------------------
# Rolling Standard Deviation
# ---------------------------------------------------------------------------


def rolling_std(series: pl.Series, period: int) -> pl.Series:
    """Rolling n-period sample standard deviation (ddof=1).

    Wraps Polars ``rolling_std`` with ``min_samples=period`` to guarantee the
    null-prefix contract.  The sample std (ddof=1) is used for consistency
    with the rest of the library (e.g., Bollinger Bands).

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Input series.
        period: Lookback window length (must be ≥ 2 for a meaningful std).

    Returns:
        Series of rolling standard deviation values named ``std_{period}``.

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "Rolling Std", min_period=2)
    return series.rolling_std(window_size=period, min_samples=period).alias(f"std_{period}")


# ---------------------------------------------------------------------------
# Percent Rank
# ---------------------------------------------------------------------------


def percent_rank(series: pl.Series, period: int) -> pl.Series:
    """Rolling percentile rank — what fraction of the last n bars are ≤ current value.

    At each bar, counts how many of the last *period* values (inclusive of the
    current bar) are less than or equal to the current value, then normalises
    by *period* to give a result in [0, 100].  A value of 100 means the
    current bar is the highest in the window; 0 means it is the lowest.

    Implemented via ``rolling_map`` since Polars has no native rolling-rank
    expression.  This is O(n × period); avoid very large periods on long series.

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Input series to rank.
        period: Rolling window length.

    Returns:
        Series of percent-rank values in [0, 100] named ``prank_{period}``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Percent Rank")

    # rolling_map passes each window as a pl.Series; we rank the last element.
    result = series.rolling_map(
        function=lambda s: (s <= s[-1]).sum() / len(s) * 100.0,
        window_size=period,
        min_samples=period,
    )
    return result.alias(f"prank_{period}")
