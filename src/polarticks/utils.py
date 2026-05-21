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
rolling_zscore  Rolling Z-score: (value − mean) / std
rolling_beta    Rolling beta: OLS regression coefficient vs a benchmark series
hurst_exponent  Rolling Hurst Exponent — trending vs. mean-reverting regime detection
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

    Bar 0 is always ``False``: a crossing requires a prior bar to cross *from*,
    so the first bar can never be a crossover regardless of the values.

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
    # fill_null(True) on the boolean: bar 0 is treated as "was already above",
    # so ~above_prev is False and crossover cannot fire without a prior bar.
    above_prev = (diff.shift(1) > atol).fill_null(True)
    return (above_now & ~above_prev).alias("crossover")


def crossunder(fast: pl.Series, slow: pl.Series, atol: float = 0.0) -> pl.Series:
    """Detect a bearish crossunder: fast crosses below slow.

    Returns ``True`` on the single bar where ``fast`` transitions from being
    at or above ``slow`` to strictly below it.

    The optional *atol* parameter mirrors the behaviour of :func:`crossover`:
    a bar is "below" only when ``slow - fast > atol``.

    Bar 0 is always ``False``: a crossing requires a prior bar to cross *from*,
    so the first bar can never be a crossunder regardless of the values.

    Args:
        fast: The faster (more responsive) series.
        slow: The slower (less responsive) series.
        atol: Absolute tolerance for equality (default 0.0).

    Returns:
        Boolean Series, ``True`` only on the crossunder bar.
    """
    diff = slow - fast
    below_now = diff > atol
    # fill_null(True): bar 0 is treated as "was already below",
    # so ~below_prev is False and crossunder cannot fire without a prior bar.
    below_prev = (diff.shift(1) > atol).fill_null(True)
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


# ---------------------------------------------------------------------------
# Rolling Z-Score
# ---------------------------------------------------------------------------


def rolling_zscore(series: pl.Series, period: int) -> pl.Series:
    """Rolling Z-score — number of standard deviations above/below the rolling mean.

    At each bar, normalises the current value relative to the rolling window's
    mean and sample standard deviation:

        z[t] = (series[t] − rolling_mean(series, period)[t])
                / rolling_std(series, period)[t]

    Null-prefix: ``period − 1`` bars (driven by the rolling std warm-up).

    Args:
        series: Input series.
        period: Rolling window length (must be ≥ 2).

    Returns:
        Series named ``zscore_{period}``.

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "Rolling Z-Score", min_period=2)

    mean = series.rolling_mean(window_size=period, min_samples=period)
    std = series.rolling_std(window_size=period, min_samples=period)

    # fill_nan converts division by zero (constant window) to null.
    return ((series - mean) / std).fill_nan(None).alias(f"zscore_{period}")


# ---------------------------------------------------------------------------
# Rolling Beta
# ---------------------------------------------------------------------------


def rolling_beta(series: pl.Series, benchmark: pl.Series, period: int) -> pl.Series:
    """Rolling beta — sensitivity of returns to a benchmark series.

    Beta is the OLS regression slope of the series' log returns against the
    benchmark's log returns over a rolling window.  Beta > 1 means the series
    moves more than the benchmark; 0 < Beta < 1 means it moves less; negative
    Beta means it moves inversely.

    Algorithm:
        ret_a = log(series[t] / series[t-1])
        ret_b = log(benchmark[t] / benchmark[t-1])
        beta  = Cov(ret_a, ret_b, period) / Var(ret_b, period)

    Null-prefix: ``period`` bars (one extra from the log-return diff).

    Args:
        series: Target price series.
        benchmark: Benchmark price series (must be the same length as ``series``).
        period: Rolling window length (must be ≥ 2).

    Returns:
        Series named ``beta_{period}``.

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "Rolling Beta", min_period=2)

    # Log returns; first bar is null for both.
    ret_a = (series / series.shift(1)).log(base=math.e)
    ret_b = (benchmark / benchmark.shift(1)).log(base=math.e)

    # Rolling covariance via the identity Cov(a,b) = E[ab] - E[a]*E[b].
    ab = ret_a * ret_b
    mean_ab = ab.rolling_mean(window_size=period, min_samples=period)
    mean_a = ret_a.rolling_mean(window_size=period, min_samples=period)
    mean_b = ret_b.rolling_mean(window_size=period, min_samples=period)
    mean_b2 = (ret_b**2).rolling_mean(window_size=period, min_samples=period)

    cov_ab = mean_ab - mean_a * mean_b
    var_b = mean_b2 - mean_b**2

    # fill_nan handles zero-variance benchmark windows (constant benchmark price).
    return (cov_ab / var_b).fill_nan(None).alias(f"beta_{period}")


# ---------------------------------------------------------------------------
# Hurst Exponent
# ---------------------------------------------------------------------------


def hurst_exponent(series: pl.Series, period: int = 100) -> pl.Series:
    """Rolling Hurst Exponent — trending vs. mean-reverting regime detection.

    The Hurst Exponent H characterises the long-memory property of the series:
        H > 0.5 → persistent (trending),
        H = 0.5 → random walk,
        H < 0.5 → anti-persistent (mean-reverting).

    Estimated via the rescaled range (R/S) method applied to log returns in
    each rolling window:
        returns   = log(price[t] / price[t-1])  in the window
        cumdev[t] = Σ(returns[0..t] − mean(returns))
        R         = max(cumdev) − min(cumdev)
        S         = std(returns)
        H         = log(R / S) / log(len(window))

    Null-prefix: ``period`` bars (one extra from the log-return diff).

    Args:
        series: Input price series.
        period: Rolling window length for each R/S estimate (default 100; must be ≥ 10).

    Returns:
        Series named ``hurst_{period}``.

    Raises:
        ValueError: If ``period < 10``.
    """
    _validate_period(period, "Hurst Exponent", min_period=10)

    # Log returns; first value is null — rolling_map will see it in windows.
    log_ret = (series / series.shift(1)).log(base=math.e)

    def _hurst(w: pl.Series) -> float:
        """R/S analysis for a single rolling window of log returns."""
        vals = [v for v in w.to_list() if v is not None]
        n = len(vals)
        if n < 2:
            return float("nan")

        # Mean-centred cumulative sum (cumulative deviation).
        mean_r = sum(vals) / n
        cumdev: list[float] = []
        running = 0.0
        for v in vals:
            running += v - mean_r
            cumdev.append(running)

        r = max(cumdev) - min(cumdev)
        if r == 0.0:
            return float("nan")

        std_r = (sum((v - mean_r) ** 2 for v in vals) / n) ** 0.5
        if std_r == 0.0:
            return float("nan")

        return math.log(r / std_r) / math.log(n)

    result = log_ret.rolling_map(function=_hurst, window_size=period, min_samples=period)
    # Convert NaN placeholders (from degenerate windows) to null.
    return result.fill_nan(None).alias(f"hurst_{period}")
