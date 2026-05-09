"""
Momentum and oscillator indicators.

All functions accept a ``pl.Series`` (or ``pl.DataFrame`` for multi-input
indicators) and return a ``pl.Series`` or ``pl.DataFrame``.  Warm-up values
are ``null``.

Functions
---------
rsi         Relative Strength Index (Wilder, default period 14)
macd        MACD line, signal line, and histogram
stochastic  Stochastic Oscillator (%K and %D)
williams_r  Williams %R
cci         Commodity Channel Index
roc         Rate of Change (percentage)
"""

from __future__ import annotations

import polars as pl

from takit.moving_averages import _validate_period, ema, sma, wilder_smooth

# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def rsi(series: pl.Series, period: int = 14) -> pl.Series:
    """Relative Strength Index using Wilder's smoothing.

    RSI measures the speed and magnitude of recent price changes to evaluate
    overbought/oversold conditions.  Values range from 0 to 100; conventional
    thresholds are >70 (overbought) and <30 (oversold).

    Algorithm:
        1. Compute bar-to-bar price deltas.
        2. Separate gains (positive deltas) and losses (absolute negative deltas).
        3. Apply Wilder's smoothing to each.
        4. RS = avg_gain / avg_loss.
        5. RSI = 100 − (100 / (1 + RS)).

    The first *period* values are ``null``.

    Args:
        series: Close price series (or any price series).
        period: Wilder smoothing period (default 14).

    Returns:
        Series of RSI values in the range [0, 100].

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "RSI", min_period=2)

    delta = series.diff(n=1)

    # Gains: only positive changes; losses set to zero.
    gains = delta.clip(lower_bound=0.0)
    # Losses: absolute value of negative changes; gains set to zero.
    losses = (-delta).clip(lower_bound=0.0)

    avg_gain = wilder_smooth(gains, period)
    avg_loss = wilder_smooth(losses, period)

    rs = avg_gain / avg_loss
    rsi_values = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss == 0 (no losses in the window), RS is inf → RSI should be 100.
    return rsi_values.fill_nan(100.0).alias(f"rsi_{period}")


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    series: pl.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.DataFrame:
    """Moving Average Convergence Divergence.

    Computes three series that together describe trend momentum:

        macd_line    = EMA(fast) − EMA(slow)
        signal_line  = EMA(macd_line, signal)
        histogram    = macd_line − signal_line

    Args:
        series: Close price series.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        DataFrame with columns ``macd_line``, ``macd_signal``, ``macd_histogram``.

    Raises:
        ValueError: If ``fast >= slow``.
    """
    if fast >= slow:
        raise ValueError(f"MACD fast period ({fast}) must be less than slow period ({slow}).")

    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)

    macd_line = (fast_ema - slow_ema).alias("macd_line")
    signal_line = ema(macd_line, signal).alias("macd_signal")
    histogram = (macd_line - signal_line).alias("macd_histogram")

    return pl.DataFrame(
        {"macd_line": macd_line, "macd_signal": signal_line, "macd_histogram": histogram}
    )


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------


def stochastic(
    ohlc: pl.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pl.DataFrame:
    """Stochastic Oscillator (%K and %D).

    Compares the closing price to the high-low range over a lookback window.
    Values oscillate between 0 and 100; conventional thresholds are >80
    (overbought) and <20 (oversold).

        %K = 100 × (close − lowest_low) / (highest_high − lowest_low)
        %D = SMA(%K, d_period)

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        k_period: %K lookback period (default 14).
        d_period: %D smoothing period (default 3).

    Returns:
        DataFrame with columns ``stoch_k`` and ``stoch_d``.

    Raises:
        ValueError: If ``k_period < 1`` or ``d_period < 1``.
    """
    _validate_period(k_period, "Stochastic %K")
    _validate_period(d_period, "Stochastic %D")

    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    highest_high = high.rolling_max(window_size=k_period, min_samples=k_period)
    lowest_low = low.rolling_min(window_size=k_period, min_samples=k_period)

    hl_range = highest_high - lowest_low

    # Fill NaN for the flat-market edge case (range == 0) with midpoint 50.
    k = (100.0 * (close - lowest_low) / hl_range).fill_nan(50.0).alias("stoch_k")
    d = sma(k, d_period).alias("stoch_d")

    return pl.DataFrame({"stoch_k": k, "stoch_d": d})


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------


def williams_r(ohlc: pl.DataFrame, period: int = 14) -> pl.Series:
    """Williams Percent Range (%R).

    A momentum indicator that measures overbought/oversold levels on a scale
    from 0 to -100.  Readings from 0 to -20 are considered overbought;
    readings from -80 to -100 are considered oversold.

        %R = −100 × (highest_high − close) / (highest_high − lowest_low)

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: Lookback period (default 14).

    Returns:
        Series of Williams %R values in the range [−100, 0].

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Williams %R")

    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    highest_high = high.rolling_max(window_size=period, min_samples=period)
    lowest_low = low.rolling_min(window_size=period, min_samples=period)

    hl_range = highest_high - lowest_low

    # Fill NaN for flat market (range == 0) with -50 (neutral midpoint).
    result = (-100.0 * (highest_high - close) / hl_range).fill_nan(-50.0)
    return result.alias(f"williams_r_{period}")


# ---------------------------------------------------------------------------
# CCI
# ---------------------------------------------------------------------------


def cci(ohlc: pl.DataFrame, period: int = 20) -> pl.Series:
    """Commodity Channel Index.

    Measures deviation of the typical price from its moving average,
    normalised by mean absolute deviation (MAD).  Values above +100 suggest
    an overbought market; values below -100 suggest oversold.

        typical_price = (high + low + close) / 3
        CCI = (typical_price − SMA(typical_price, n)) / (0.015 × MAD)

    The constant 0.015 ensures that roughly 70-80% of CCI values fall
    between ±100 in a typical market.

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: Lookback period (default 20).

    Returns:
        Series of CCI values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "CCI")

    tp = ((ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3.0).alias("tp")

    tp_sma = tp.rolling_mean(window_size=period, min_samples=period)

    # Mean absolute deviation computed via a rolling map.
    # For each window, MAD = mean(|tp_i − mean(tp)|).
    mad = tp.rolling_map(
        function=lambda s: (s - s.mean()).abs().mean(),
        window_size=period,
        min_samples=period,
    )

    # Protect against a perfectly flat window where MAD == 0.
    result = (tp - tp_sma) / (0.015 * mad)
    return result.alias(f"cci_{period}")


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------


def roc(series: pl.Series, period: int = 10) -> pl.Series:
    """Rate of Change (percentage).

    Measures the percentage change in price over *period* bars:

        ROC = 100 × (close − close[n]) / close[n]

    A positive ROC indicates upward momentum; negative indicates downward.

    Args:
        series: Close price series.
        period: Lookback period (default 10).

    Returns:
        Series of ROC values in percentage terms.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "ROC")
    past = series.shift(period)
    return (100.0 * (series - past) / past).alias(f"roc_{period}")
