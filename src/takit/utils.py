"""
General-purpose signal and return utilities.

Functions
---------
crossover       Bullish cross: fast series crosses above slow series
crossunder      Bearish cross: fast series crosses below slow series
log_returns     Bar-to-bar log returns: ln(price[t] / price[t-1])
simple_returns  Bar-to-bar simple returns: (price[t] - price[t-1]) / price[t-1]
"""

from __future__ import annotations

import polars as pl


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
    return (series / series.shift(1)).log(base=2.718281828459045).alias("log_returns")


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
