"""
Momentum and oscillator indicators.

All functions accept a ``pl.Series`` (or ``pl.DataFrame`` for multi-input
indicators) and return a ``pl.Series`` or ``pl.DataFrame``.  Warm-up values
are ``null``.

Functions
---------
rsi                 Relative Strength Index (Wilder, default period 14)
macd                MACD line, signal line, and histogram
stochastic          Stochastic Oscillator (%K and %D)
williams_r          Williams %R
cci                 Commodity Channel Index
roc                 Rate of Change (percentage)
mfi                 Money Flow Index (volume-weighted RSI)
cmf                 Chaikin Money Flow
tsi                 True Strength Index (double-smoothed momentum)
ultimate_oscillator Weighted blend of three time-frame oscillators
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


# ---------------------------------------------------------------------------
# MFI
# ---------------------------------------------------------------------------


def mfi(ohlc_vol: pl.DataFrame, period: int = 14) -> pl.Series:
    """Money Flow Index — volume-weighted RSI.

    MFI combines price and volume to identify overbought/oversold conditions.
    Like RSI, readings above 80 suggest overbought and below 20 suggest
    oversold, but MFI is more sensitive to volume divergence.

    Algorithm:
        typical_price = (high + low + close) / 3
        money_flow    = typical_price × volume
        Positive money flow: bars where typical_price > prior typical_price.
        Negative money flow: bars where typical_price ≤ prior typical_price.
        MFI = 100 − 100 / (1 + sum_pos_mf / sum_neg_mf)   over period bars.

    Args:
        ohlc_vol: DataFrame with columns ``high``, ``low``, ``close``, ``volume``.
        period: Rolling window length (default 14).

    Returns:
        Series of MFI values in [0, 100].

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "MFI", min_period=2)

    high = ohlc_vol["high"]
    low = ohlc_vol["low"]
    close = ohlc_vol["close"]
    volume = ohlc_vol["volume"].cast(pl.Float64)

    tp = (high + low + close) / 3.0
    money_flow = tp * volume
    tp_prev = tp.shift(1)

    # Allocate each bar's money flow to the positive or negative bucket.
    pos_mf = (
        pl.select(pl.when(tp > tp_prev).then(money_flow).otherwise(0.0)).to_series().fill_null(0.0)
    )
    neg_mf = (
        pl.select(pl.when(tp <= tp_prev).then(money_flow).otherwise(0.0)).to_series().fill_null(0.0)
    )

    pos_sum = pos_mf.rolling_sum(window_size=period, min_samples=period)
    neg_sum = neg_mf.rolling_sum(window_size=period, min_samples=period)

    mfr = pos_sum / neg_sum
    # When neg_sum == 0 (no selling pressure), MFR is inf → MFI = 100.
    return (100.0 - 100.0 / (1.0 + mfr)).fill_nan(100.0).alias(f"mfi_{period}")


# ---------------------------------------------------------------------------
# CMF
# ---------------------------------------------------------------------------


def cmf(ohlc_vol: pl.DataFrame, period: int = 20) -> pl.Series:
    """Chaikin Money Flow.

    CMF measures buying and selling pressure by combining the Money Flow
    Multiplier (position of close within the bar's range) with volume.
    Values above 0 indicate accumulation (buying pressure); below 0
    indicate distribution (selling pressure).

    Algorithm:
        money_flow_multiplier = (2 × close − high − low) / (high − low)
        money_flow_volume     = multiplier × volume
        CMF = Σ(money_flow_volume, n) / Σ(volume, n)

    Args:
        ohlc_vol: DataFrame with columns ``high``, ``low``, ``close``, ``volume``.
        period: Rolling window length (default 20).

    Returns:
        Series of CMF values in the range [-1, 1].

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "CMF")

    high = ohlc_vol["high"]
    low = ohlc_vol["low"]
    close = ohlc_vol["close"]
    volume = ohlc_vol["volume"].cast(pl.Float64)

    hl_range = high - low
    # fill_nan handles zero-range (doji-like) bars by zeroing their contribution.
    mfm = ((2.0 * close - high - low) / hl_range).fill_nan(0.0)
    mfv = mfm * volume

    mfv_sum = mfv.rolling_sum(window_size=period, min_samples=period)
    vol_sum = volume.rolling_sum(window_size=period, min_samples=period)

    return (mfv_sum / vol_sum).alias(f"cmf_{period}")


# ---------------------------------------------------------------------------
# TSI
# ---------------------------------------------------------------------------


def tsi(
    series: pl.Series,
    slow: int = 25,
    fast: int = 13,
) -> pl.Series:
    """True Strength Index — double-smoothed momentum oscillator.

    TSI measures trend direction and magnitude using double EMA smoothing
    of raw momentum (price changes).  Values above 0 are bullish; below 0
    are bearish.  Typical signal line: EMA(TSI, 7).

        momentum            = close[t] − close[t-1]
        double_smooth       = EMA(EMA(momentum, slow), fast)
        double_smooth_abs   = EMA(EMA(|momentum|, slow), fast)
        TSI                 = 100 × double_smooth / double_smooth_abs

    Args:
        series: Close price series.
        slow: Period of the first (slower) EMA smoothing pass (default 25).
        fast: Period of the second (faster) EMA smoothing pass (default 13).

    Returns:
        Series of TSI values in the range (-100, +100).

    Raises:
        ValueError: If ``slow < 1`` or ``fast < 1``.
    """
    _validate_period(slow, "TSI slow")
    _validate_period(fast, "TSI fast")

    delta = series.diff(n=1)
    abs_delta = delta.abs()

    double_smooth = ema(ema(delta, slow), fast)
    double_smooth_abs = ema(ema(abs_delta, slow), fast)

    # fill_nan handles the edge case where abs momentum is zero throughout.
    return (100.0 * double_smooth / double_smooth_abs).fill_nan(0.0).alias(f"tsi_{slow}_{fast}")


# ---------------------------------------------------------------------------
# Ultimate Oscillator
# ---------------------------------------------------------------------------


def ultimate_oscillator(
    ohlc: pl.DataFrame,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
) -> pl.Series:
    """Ultimate Oscillator — weighted blend of three time-frame oscillators.

    Combines buying pressure and true range over three different periods
    to reduce false signals from any single time frame.  Values above 70
    indicate overbought; below 30 indicate oversold.

    Algorithm (Larry Williams, 1976):
        buying_pressure = close − min(low, prev_close)
        true_range      = max(high, prev_close) − min(low, prev_close)
        avg_n = Σ(BP, n) / Σ(TR, n)
        UO = 100 × (4 × avg1 + 2 × avg2 + avg3) / 7

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period1: Shortest lookback period (default 7).
        period2: Medium lookback period (default 14).
        period3: Longest lookback period (default 28).

    Returns:
        Series of Ultimate Oscillator values in [0, 100].

    Raises:
        ValueError: If any period is less than 1.
    """
    _validate_period(period1, "Ultimate Oscillator period1")
    _validate_period(period2, "Ultimate Oscillator period2")
    _validate_period(period3, "Ultimate Oscillator period3")

    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]
    prev_close = close.shift(1)

    # Buying pressure: how far close moved above the effective low.
    true_low = pl.select(pl.min_horizontal(low, prev_close)).to_series()
    true_high = pl.select(pl.max_horizontal(high, prev_close)).to_series()

    buying_pressure = close - true_low
    true_range_vals = true_high - true_low

    def _avg(period: int) -> pl.Series:
        """Rolling BP/TR ratio for a given period."""
        bp_sum = buying_pressure.rolling_sum(window_size=period, min_samples=period)
        tr_sum = true_range_vals.rolling_sum(window_size=period, min_samples=period)
        return (bp_sum / tr_sum).fill_nan(0.5)

    avg1 = _avg(period1)
    avg2 = _avg(period2)
    avg3 = _avg(period3)

    return (100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0).alias(
        f"uo_{period1}_{period2}_{period3}"
    )
