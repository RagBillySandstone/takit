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


def crossover(fast: pl.Series, slow: pl.Series) -> pl.Series:
    """Detect a bullish crossover: fast crosses above slow.

    Returns ``True`` on the single bar where ``fast`` transitions from being
    at or below ``slow`` to strictly above it.  All other bars are ``False``.

    Args:
        fast: The faster (more responsive) series.
        slow: The slower (less responsive) series.

    Returns:
        Boolean Series, ``True`` only on the crossover bar.
    """
    above_now = fast > slow
    above_prev = (fast > slow).shift(1).fill_null(value=False)
    return (above_now & ~above_prev).alias("crossover")


def crossunder(fast: pl.Series, slow: pl.Series) -> pl.Series:
    """Detect a bearish crossunder: fast crosses below slow.

    Returns ``True`` on the single bar where ``fast`` transitions from being
    at or above ``slow`` to strictly below it.

    Args:
        fast: The faster (more responsive) series.
        slow: The slower (less responsive) series.

    Returns:
        Boolean Series, ``True`` only on the crossunder bar.
    """
    below_now = fast < slow
    below_prev = (fast < slow).shift(1).fill_null(value=False)
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
