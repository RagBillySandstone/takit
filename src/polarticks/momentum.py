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
stoch_rsi           Stochastic applied to RSI (StochRSI %K and %D)
ppo                 Percentage Price Oscillator (MACD normalised as %)
williams_r          Williams %R
cci                 Commodity Channel Index
roc                 Rate of Change (percentage)
mfi                 Money Flow Index (volume-weighted RSI)
cmf                 Chaikin Money Flow
tsi                 True Strength Index (double-smoothed momentum)
ultimate_oscillator Weighted blend of three time-frame oscillators
cmo                 Chande Momentum Oscillator (net momentum as % of total movement)
dpo                 Detrended Price Oscillator (removes trend to isolate price cycles)
kst                 Know Sure Thing (weighted sum of four smoothed ROC oscillators)
coppock             Coppock Curve (WMA of combined ROC; long-term bottom detector)
fisher_transform    Fisher Transform — arctanh normalisation of HL midpoint
awesome_oscillator  Awesome Oscillator — SMA(midpoint,5) minus SMA(midpoint,34)
accelerator_oscillator  Accelerator Oscillator — AO minus SMA(AO,5)
smi                 Stochastic Momentum Index — double-EMA stochastic variant
rvi                 Relative Vigor Index — open/close vs high/low momentum
bop                 Balance of Power — close position within open-close range
qqe                 Quantitative Qualitative Estimation — RSI-based trailing trend line
crsi                Connors RSI — composite of RSI, streak-RSI, and percent-rank
qstick              Q-Stick — EMA of intraday close-minus-open directionality
psy_line            Psychological Line — percentage of rising bars in a rolling window
rocr                Rate of Change Ratio — close / close[n] (ratio form of ROC)
disparity_index     Disparity Index — % deviation of price from its moving average
apo                 Absolute Price Oscillator — difference of two EMAs in price units
asi                 Accumulative Swing Index — Wilder's cumulative directional indicator
pmo                 Price Momentum Oscillator — double-smoothed rate-of-change
chande_trend_score  Chande Trend Score — % of prior bars the current close exceeds
dss                 Double-smoothed Stochastic — Bressert's two-pass stochastic oscillator
vwrsi               Volume-weighted RSI — Wilder RSI with volume-scaled gains/losses
"""

from __future__ import annotations

import math

import polars as pl

from polarticks._validate import _validate_period
from polarticks.moving_averages import (  # noqa: F401 (wma used by coppock)
    ema,
    sma,
    wilder_smooth,
    wma,
)
from polarticks.utils import percent_rank

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

    # rolling_map invokes a Python callback per window — no vectorised MAD
    # expression exists in Polars. Accepted performance trade-off for correctness.
    mad = tp.rolling_map(
        function=lambda s: (s - s.mean()).abs().mean(),
        window_size=period,
        min_samples=period,
    )

    # Perfectly flat windows produce MAD = 0 → 0/0 = NaN.  CCI is
    # conventionally 0 in a directionless market.
    result = ((tp - tp_sma) / (0.015 * mad)).fill_nan(0.0)
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
    # fill_nan converts inf/nan (zero past price) to null rather than silently
    # propagating an undefined value.
    return (100.0 * (series - past) / past).fill_nan(None).alias(f"roc_{period}")


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
        pl.select(pl.when(tp < tp_prev).then(money_flow).otherwise(0.0)).to_series().fill_null(0.0)
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


# ---------------------------------------------------------------------------
# StochRSI
# ---------------------------------------------------------------------------


def stoch_rsi(
    series: pl.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> pl.DataFrame:
    """Stochastic RSI — stochastic oscillator applied to RSI values.

    StochRSI generates overbought/oversold signals more frequently than raw
    RSI because it measures RSI relative to its own range rather than price.
    Values are in [0, 100]; readings above 80 suggest overbought, below 20
    suggest oversold.

    Algorithm:
        rsi_vals          = RSI(series, rsi_period)
        lowest_rsi[t]     = min(rsi_vals, stoch_period)
        highest_rsi[t]    = max(rsi_vals, stoch_period)
        raw_%K[t]         = 100 × (rsi_vals[t] − lowest_rsi[t])
                            / (highest_rsi[t] − lowest_rsi[t])
        %K                = SMA(raw_%K, k_period)        (smoothed fast line)
        %D                = SMA(%K, d_period)            (signal / slow line)

    Null-prefix for ``stoch_rsi_k``:
        ``rsi_period + stoch_period + k_period - 2`` bars.
    Null-prefix for ``stoch_rsi_d``:
        ``rsi_period + stoch_period + k_period + d_period - 3`` bars.

    Args:
        series: Close price series (or any price series).
        rsi_period: Period for the underlying RSI calculation (default 14).
        stoch_period: Lookback window applied to RSI values (default 14).
        k_period: Smoothing period for the fast %K line (default 3).
        d_period: Smoothing period for the slow %D signal line (default 3).

    Returns:
        DataFrame with columns ``stoch_rsi_k`` and ``stoch_rsi_d``.

    Raises:
        ValueError: If any period is below its minimum (all must be ≥ 1,
            ``rsi_period`` must be ≥ 2).
    """
    _validate_period(rsi_period, "StochRSI rsi_period", min_period=2)
    _validate_period(stoch_period, "StochRSI stoch_period")
    _validate_period(k_period, "StochRSI k_period")
    _validate_period(d_period, "StochRSI d_period")

    rsi_vals = rsi(series, rsi_period)

    lowest_rsi = rsi_vals.rolling_min(window_size=stoch_period, min_samples=stoch_period)
    highest_rsi = rsi_vals.rolling_max(window_size=stoch_period, min_samples=stoch_period)

    rsi_range = highest_rsi - lowest_rsi

    # fill_nan handles the degenerate case where RSI is constant over the window
    # (range == 0); 50 is the neutral midpoint.
    raw_k = (100.0 * (rsi_vals - lowest_rsi) / rsi_range).fill_nan(50.0)

    k = sma(raw_k, k_period).alias("stoch_rsi_k")
    d = sma(k, d_period).alias("stoch_rsi_d")

    return pl.DataFrame({"stoch_rsi_k": k, "stoch_rsi_d": d})


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


def ppo(
    series: pl.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.DataFrame:
    """Percentage Price Oscillator — MACD normalised as a percentage of price.

    PPO expresses the MACD line as a fraction of the slow EMA, making it
    directly comparable across instruments with different price levels.
    A value of +1.5 means the fast EMA is 1.5% above the slow EMA.

    Algorithm:
        ppo_line      = 100 × (EMA(fast) − EMA(slow)) / EMA(slow)
        ppo_signal    = EMA(ppo_line, signal)
        ppo_histogram = ppo_line − ppo_signal

    Null-prefix:
        ``ppo_line``      — ``slow − 1`` bars.
        ``ppo_signal``    — ``(slow − 1) + (signal − 1)`` bars.
        ``ppo_histogram`` — same as ``ppo_signal``.

    Args:
        series: Close price series.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        DataFrame with columns ``ppo_line``, ``ppo_signal``, ``ppo_histogram``.

    Raises:
        ValueError: If ``fast >= slow``.
    """
    if fast >= slow:
        raise ValueError(f"PPO fast period ({fast}) must be less than slow period ({slow}).")

    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)

    # fill_nan handles the rare zero-slow-EMA edge case.
    ppo_line = (100.0 * (fast_ema - slow_ema) / slow_ema).fill_nan(None).alias("ppo_line")

    signal_line = ema(ppo_line, signal).alias("ppo_signal")
    histogram = (ppo_line - signal_line).alias("ppo_histogram")

    return pl.DataFrame(
        {"ppo_line": ppo_line, "ppo_signal": signal_line, "ppo_histogram": histogram}
    )


# ---------------------------------------------------------------------------
# CMO
# ---------------------------------------------------------------------------


def cmo(series: pl.Series, period: int = 14) -> pl.Series:
    """Chande Momentum Oscillator — net momentum as a percentage of total movement.

    CMO compares the sum of up-moves against total price movement (up + down)
    over the lookback period.  It oscillates between −100 (all down) and
    +100 (all up).  Unlike RSI, it does not smooth the gains/losses — the
    raw daily deltas are used directly.

    Algorithm:
        delta    = price.diff(1)
        sum_up   = Σ max(delta, 0)  over *period* bars
        sum_down = Σ max(−delta, 0) over *period* bars
        CMO      = 100 × (sum_up − sum_down) / (sum_up + sum_down)

    Null-prefix: ``period`` bars (diff contributes 1; rolling sum adds period − 1).

    Args:
        series: Input price series (e.g., close).
        period: Lookback window for the rolling sums (default 14).

    Returns:
        Series of CMO values in (−100, +100).
        The first ``period`` values are ``null``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "CMO")

    delta = series.diff(1)
    sum_up = delta.clip(lower_bound=0.0).rolling_sum(window_size=period, min_samples=period)
    sum_down = (-delta).clip(lower_bound=0.0).rolling_sum(window_size=period, min_samples=period)

    # fill_nan handles the degenerate case where all bars are flat (denominator = 0).
    result = (100.0 * (sum_up - sum_down) / (sum_up + sum_down)).fill_nan(None)
    return result.alias(f"cmo_{period}")


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


def dpo(series: pl.Series, period: int = 20) -> pl.Series:
    """Detrended Price Oscillator — isolates short-term price cycles.

    DPO removes the dominant trend by subtracting a displaced SMA from the
    current price.  The SMA is shifted back by ``period // 2 + 1`` bars so
    it represents the "centre of gravity" of the lookback window rather than
    a trailing average.  The result oscillates around zero; peaks and troughs
    correspond to cyclical highs and lows.

    Algorithm:
        displacement = period // 2 + 1
        DPO[t]       = price[t] − SMA(price, period)[t − displacement]

    Null-prefix: ``(period − 1) + displacement`` bars.

    Args:
        series: Input price series (e.g., close).
        period: SMA lookback period (default 20).

    Returns:
        Series of DPO values.
        The first ``(period − 1) + period // 2 + 1`` values are ``null``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "DPO")

    displacement = period // 2 + 1
    # shift(displacement) moves the SMA back in time so the current price is
    # compared against a centred historical average rather than a trailing one.
    displaced_sma = sma(series, period).shift(displacement)
    return (series - displaced_sma).alias(f"dpo_{period}")


# ---------------------------------------------------------------------------
# KST
# ---------------------------------------------------------------------------


def kst(
    series: pl.Series,
    roc1: int = 10,
    roc2: int = 13,
    roc3: int = 14,
    roc4: int = 24,
    sma1: int = 10,
    sma2: int = 13,
    sma3: int = 14,
    sma4: int = 24,
    signal: int = 9,
) -> pl.DataFrame:
    """Know Sure Thing (KST) — weighted sum of smoothed ROC oscillators.

    KST (Martin Pring, 1992) sums four Rate-of-Change series — each smoothed
    with an SMA — at progressively longer time-frames, weighted so that longer
    cycles contribute more.  It is designed to identify major trend changes.

    Algorithm:
        RCMA_k[t] = SMA(ROC(price, roc_k), sma_k)
        KST[t]    = 1·RCMA_1 + 2·RCMA_2 + 3·RCMA_3 + 4·RCMA_4
        signal[t] = SMA(KST, signal)

    Default daily parameters (Pring 1992):
        roc1/sma1 = 10/10, roc2/sma2 = 13/13, roc3/sma3 = 14/14, roc4/sma4 = 24/24

    Null-prefix for ``kst_line``:   ``roc4 + sma4 − 1`` bars (slowest component).
    Null-prefix for ``kst_signal``: kst_line nulls + ``signal − 1`` bars.

    Args:
        series: Close price series.
        roc1..roc4: ROC lookback periods (default 10, 13, 14, 24).
        sma1..sma4: SMA smoothing periods applied to each ROC (default 10, 13, 14, 24).
        signal: SMA period for the signal line (default 9).

    Returns:
        DataFrame with columns ``kst_line`` and ``kst_signal``.
    """
    # Build four smoothed ROC components with increasing weights.
    rcma1 = sma(roc(series, roc1), sma1)
    rcma2 = sma(roc(series, roc2), sma2) * 2.0
    rcma3 = sma(roc(series, roc3), sma3) * 3.0
    rcma4 = sma(roc(series, roc4), sma4) * 4.0

    kst_line = (rcma1 + rcma2 + rcma3 + rcma4).alias("kst_line")
    kst_signal = sma(kst_line, signal).alias("kst_signal")

    return pl.DataFrame({"kst_line": kst_line, "kst_signal": kst_signal})


# ---------------------------------------------------------------------------
# Coppock Curve
# ---------------------------------------------------------------------------


def coppock(
    series: pl.Series,
    long_roc: int = 14,
    short_roc: int = 11,
    wma_period: int = 10,
) -> pl.Series:
    """Coppock Curve — long-term momentum oscillator for major market bottoms.

    The Coppock Curve (Edwin Coppock, 1962) adds two ROC values at different
    time-frames and smooths the result with a Weighted Moving Average.
    Originally designed on monthly data to identify buying opportunities after
    major bear markets; a cross from negative to positive is the buy signal.

    Algorithm:
        Coppock = WMA(ROC(price, long_roc) + ROC(price, short_roc), wma_period)

    Null-prefix: ``long_roc + wma_period − 1`` bars
        (long_roc nulls from the slower ROC; wma_period − 1 from the WMA).

    Args:
        series: Close price series.
        long_roc: Longer ROC period (default 14).
        short_roc: Shorter ROC period (default 11).
        wma_period: WMA smoothing period (default 10).

    Returns:
        Series named ``coppock``.

    Raises:
        ValueError: If any period < 1.
    """
    _validate_period(long_roc, "Coppock long_roc")
    _validate_period(short_roc, "Coppock short_roc")
    _validate_period(wma_period, "Coppock wma_period")

    r1 = roc(series, long_roc)
    r2 = roc(series, short_roc)
    # WMA of the sum of both ROCs; null propagation handles the warm-up span.
    return wma(r1 + r2, wma_period).alias("coppock")


# ---------------------------------------------------------------------------
# Fisher Transform
# ---------------------------------------------------------------------------


def fisher_transform(ohlc: pl.DataFrame, period: int = 9) -> pl.DataFrame:
    """Fisher Transform — normalises the HL midpoint to a near-Gaussian distribution.

    The Fisher Transform (John Ehlers, 2002) applies the inverse hyperbolic
    tangent (arctanh) to a price series normalised to the range (−1, 1).
    The resulting distribution is nearly Gaussian, making extreme readings
    statistically significant and easier to identify as turning points.

    Algorithm:
        hl2[t]       = (high[t] + low[t]) / 2
        highest[t]   = max(hl2, period)
        lowest[t]    = min(hl2, period)
        value[t]     = 2 × (hl2 − lowest) / (highest − lowest) − 1
        value        = clamp(value, −0.999, 0.999)   # avoid arctanh singularity
        fisher[t]    = 0.5 × ln((1 + value) / (1 − value))
        signal[t]    = fisher[t − 1]

    Null-prefix: ``period − 1`` bars for ``fisher``; ``period`` bars for ``signal``.

    Args:
        ohlc: DataFrame with columns ``high`` and ``low``.
        period: Rolling window for normalisation (default 9).

    Returns:
        DataFrame with columns ``fisher`` and ``fisher_signal``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Fisher Transform")

    # Midpoint of the bar's range.
    hl2 = (ohlc["high"] + ohlc["low"]) / 2.0

    highest = hl2.rolling_max(window_size=period, min_samples=period)
    lowest = hl2.rolling_min(window_size=period, min_samples=period)
    hl_range = highest - lowest

    # Normalise to (−1, 1); flat windows produce NaN from 0/0 → treat as 0.
    value = (2.0 * (hl2 - lowest) / hl_range - 1.0).fill_nan(0.0).clip(-0.999, 0.999)

    # arctanh via the logarithm identity: arctanh(x) = 0.5 * ln((1+x)/(1-x)).
    fisher = (0.5 * ((1.0 + value) / (1.0 - value)).log(base=math.e)).alias("fisher")
    signal = fisher.shift(1).alias("fisher_signal")

    return pl.DataFrame({"fisher": fisher, "fisher_signal": signal})


# ---------------------------------------------------------------------------
# Awesome Oscillator
# ---------------------------------------------------------------------------


def awesome_oscillator(
    ohlc: pl.DataFrame,
    fast: int = 5,
    slow: int = 34,
) -> pl.Series:
    """Awesome Oscillator (Bill Williams) — SMA difference of bar midpoints.

    AO compares a fast and slow simple moving average of the bar's median
    price ``(high + low) / 2`` to gauge momentum.  Values above zero signal
    bullish momentum; below zero, bearish momentum.  A cross of zero is a
    primary buy/sell signal; the "saucer" and twin-peak setups are secondary.

    Algorithm:
        midpoint = (high + low) / 2
        AO = SMA(midpoint, fast) − SMA(midpoint, slow)

    Null-prefix: ``slow − 1`` bars.

    Args:
        ohlc: DataFrame with columns ``high`` and ``low``.
        fast: Fast SMA period (default 5).
        slow: Slow SMA period (default 34).

    Returns:
        Series named ``ao``.

    Raises:
        ValueError: If ``fast >= slow``.
    """
    if fast >= slow:
        raise ValueError(f"AO fast period ({fast}) must be less than slow period ({slow}).")

    midpoint = (ohlc["high"] + ohlc["low"]) / 2.0
    # Null-propagation aligns naturally: slow SMA drives the warm-up.
    return (sma(midpoint, fast) - sma(midpoint, slow)).alias("ao")


# ---------------------------------------------------------------------------
# Accelerator Oscillator
# ---------------------------------------------------------------------------


def accelerator_oscillator(
    ohlc: pl.DataFrame,
    fast: int = 5,
    slow: int = 34,
    signal: int = 5,
) -> pl.Series:
    """Accelerator Oscillator (Bill Williams) — momentum of the Awesome Oscillator.

    AC is the difference between the Awesome Oscillator and a moving average
    of AO itself.  It changes direction before AO, providing an earlier signal
    in Bill Williams' three-screen trading system.

    Algorithm:
        AO = SMA(midpoint, fast) − SMA(midpoint, slow)
        AC = AO − SMA(AO, signal)

    Null-prefix: ``slow + signal − 2`` bars.

    Args:
        ohlc: DataFrame with columns ``high`` and ``low``.
        fast: AO fast SMA period (default 5).
        slow: AO slow SMA period (default 34).
        signal: SMA period applied to AO (default 5).

    Returns:
        Series named ``ac``.

    Raises:
        ValueError: If ``fast >= slow`` or ``signal < 1``.
    """
    _validate_period(signal, "AC signal")
    ao = awesome_oscillator(ohlc, fast, slow)
    return (ao - sma(ao, signal)).alias("ac")


# ---------------------------------------------------------------------------
# Stochastic Momentum Index
# ---------------------------------------------------------------------------


def smi(
    ohlc: pl.DataFrame,
    period: int = 14,
    smooth1: int = 3,
    smooth2: int = 3,
    signal: int = 9,
) -> pl.DataFrame:
    """Stochastic Momentum Index — double-EMA smoothed stochastic oscillator.

    SMI (Blau, 1993) reduces the noise of the classic stochastic by applying
    two rounds of EMA smoothing to both the numerator (distance of close from
    the midpoint of the period's range) and the denominator (the range itself).
    It oscillates between −100 and +100.

    Algorithm:
        HH   = rolling_max(high,  period)
        LL   = rolling_min(low,   period)
        D    = close − (HH + LL) / 2
        HLS  = HH − LL
        DS   = EMA(EMA(D,  smooth1), smooth2)   (double-smoothed distance)
        HLSS = EMA(EMA(HLS, smooth1), smooth2)  (double-smoothed range)
        SMI  = 100 × DS / (0.5 × HLSS)
        signal = EMA(SMI, signal)

    Null-prefix for ``smi``:    ``(period − 1) + 2 × max(smooth1, smooth2) − 2`` bars.

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: Lookback for rolling high/low (default 14).
        smooth1: Period of the first EMA smoothing pass (default 3).
        smooth2: Period of the second EMA smoothing pass (default 3).
        signal: EMA period for the signal line (default 9).

    Returns:
        DataFrame with columns ``smi`` and ``smi_signal``.

    Raises:
        ValueError: If any period < 1.
    """
    _validate_period(period, "SMI")
    _validate_period(smooth1, "SMI smooth1")
    _validate_period(smooth2, "SMI smooth2")
    _validate_period(signal, "SMI signal")

    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    # Rolling extremes of the period window.
    hh = high.rolling_max(window_size=period, min_samples=period)
    ll = low.rolling_min(window_size=period, min_samples=period)

    # Distance from close to the midpoint; range of the window.
    d = close - (hh + ll) / 2.0
    hls = hh - ll

    # Two EMA passes smooth each series independently.
    ds = ema(ema(d, smooth1), smooth2)
    hlss = ema(ema(hls, smooth1), smooth2)

    # fill_nan handles perfectly flat markets where HLSS == 0.
    smi_vals = (100.0 * ds / (0.5 * hlss)).fill_nan(0.0).alias("smi")
    smi_signal = ema(smi_vals, signal).alias("smi_signal")

    return pl.DataFrame({"smi": smi_vals, "smi_signal": smi_signal})


# ---------------------------------------------------------------------------
# Relative Vigor Index
# ---------------------------------------------------------------------------


def rvi(ohlc: pl.DataFrame, period: int = 10) -> pl.DataFrame:
    """Relative Vigor Index — symmetrically weighted open/close momentum.

    RVI (Ehlers, 2002) measures the tendency for prices to close higher in
    a rising market by using a symmetric four-bar triangular weighted average
    of ``(close − open)`` relative to ``(high − low)``.  Values above zero
    are bullish; below zero are bearish.

    Algorithm:
        num_raw = (CO + 2×CO[−1] + 2×CO[−2] + CO[−3]) / 6   where CO = close − open
        den_raw = (HL + 2×HL[−1] + 2×HL[−2] + HL[−3]) / 6   where HL = high − low
        RVI     = SMA(num_raw, period) / SMA(den_raw, period)
        signal  = (RVI + 2×RVI[−1] + 2×RVI[−2] + RVI[−3]) / 6

    Null-prefix for ``rvi``:    ``period + 2`` bars.
    Null-prefix for ``rvi_signal``: ``period + 5`` bars.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        period: SMA lookback period (default 10).

    Returns:
        DataFrame with columns ``rvi`` and ``rvi_signal``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "RVI")

    o = ohlc["open"]
    h = ohlc["high"]
    lo = ohlc["low"]
    c = ohlc["close"]

    co = c - o
    hl = h - lo

    # Symmetric 4-bar triangular weights (1, 2, 2, 1) normalised to sum 6.
    num_raw = (co + 2.0 * co.shift(1) + 2.0 * co.shift(2) + co.shift(3)) / 6.0
    den_raw = (hl + 2.0 * hl.shift(1) + 2.0 * hl.shift(2) + hl.shift(3)) / 6.0

    # fill_nan handles zero-range bars (identical high and low throughout period).
    rvi_vals = (sma(num_raw, period) / sma(den_raw, period)).fill_nan(0.0).alias("rvi")

    # Signal: same 4-bar symmetric average applied to the RVI line.
    rvi_signal = (
        (rvi_vals + 2.0 * rvi_vals.shift(1) + 2.0 * rvi_vals.shift(2) + rvi_vals.shift(3)) / 6.0
    ).alias("rvi_signal")

    return pl.DataFrame({"rvi": rvi_vals, "rvi_signal": rvi_signal})


# ---------------------------------------------------------------------------
# Balance of Power
# ---------------------------------------------------------------------------


def bop(ohlc: pl.DataFrame, period: int = 14) -> pl.Series:
    """Balance of Power — close position within the bar's open-close range.

    BOP (Igor Livshin) measures the relative strength between buyers and
    sellers by comparing how far the close moved from the open, normalised
    by the high-low range.  Values near +1 indicate strong buying pressure;
    values near −1 indicate strong selling pressure.

        BOP = (close − open) / (high − low)

    Optional SMA smoothing reduces noise.

    Null-prefix (period > 1): ``period − 1`` bars.
    Null-prefix (period = 1): 0 bars (raw, no smoothing).

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        period: SMA smoothing period; use 1 for unsmoothed raw values (default 14).

    Returns:
        Series named ``bop_{period}``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "BOP")

    hl_range = ohlc["high"] - ohlc["low"]
    # fill_nan: a zero-range bar (open == high == low == close) → BOP is 0.
    raw_bop = ((ohlc["close"] - ohlc["open"]) / hl_range).fill_nan(0.0)

    if period == 1:
        return raw_bop.alias("bop_1")

    return sma(raw_bop, period).alias(f"bop_{period}")


# ---------------------------------------------------------------------------
# QQE
# ---------------------------------------------------------------------------


def qqe(
    series: pl.Series,
    rsi_period: int = 14,
    sf: int = 5,
    qqe_factor: float = 4.236,
) -> pl.DataFrame:
    """Quantitative Qualitative Estimation — RSI-derived adaptive trailing trend line.

    QQE smooths RSI with an EMA and then builds an adaptive trailing stop
    from the ATR of the smoothed RSI (double Wilder-smoothed absolute delta).
    The QQE line is a ratcheting band: it can only move in the direction of
    the prevailing RSI trend and flips when the RSI crosses to the opposite
    side.  Price above the QQE line is bullish; below is bearish.

    Algorithm:
        rsi_ma   = EMA(RSI(series, rsi_period), sf)
        atr_rsi  = |rsi_ma[t] − rsi_ma[t−1]|
        dar      = Wilder(Wilder(atr_rsi, sf×2−1), sf×2−1) × qqe_factor
        # Stateful trailing bands:
        long_band[t]  = max(rsi_ma[t] − dar[t], long_band[t−1])  while rsi_ma trending up
        short_band[t] = min(rsi_ma[t] + dar[t], short_band[t−1]) while rsi_ma trending down
        qqe_line = long_band or short_band depending on which side rsi_ma is on

    Null-prefix: ``rsi_period + 2 × (sf × 2 − 1) − 1`` bars (dominated by the
    double Wilder smooth of ``dar``).

    Args:
        series: Close price series.
        rsi_period: RSI period (default 14; must be ≥ 2).
        sf: EMA smoothing factor period (default 5).
        qqe_factor: ATR multiplier controlling band width (default 4.236).

    Returns:
        DataFrame with columns ``qqe_line`` (trailing stop) and
        ``qqe_fast`` (smoothed RSI oscillator).

    Raises:
        ValueError: If ``rsi_period < 2`` or ``sf < 1``.
    """
    _validate_period(rsi_period, "QQE rsi_period", min_period=2)
    _validate_period(sf, "QQE sf")

    rsi_vals = rsi(series, rsi_period)
    rsi_ma = ema(rsi_vals, sf)

    # Pseudo-ATR of smoothed RSI via double Wilder smoothing.
    atr_rsi = rsi_ma.diff(1).abs()
    slow = sf * 2 - 1
    dar = wilder_smooth(wilder_smooth(atr_rsi, slow), slow) * qqe_factor

    rma_list: list[float | None] = rsi_ma.to_list()
    dar_list: list[float | None] = dar.to_list()
    n = len(rma_list)
    qqe_line: list[float | None] = [None] * n

    # Locate the first bar where both rsi_ma and dar are valid.
    start = 0
    while start < n and (rma_list[start] is None or dar_list[start] is None):
        start += 1

    if start >= n:
        return pl.DataFrame({"qqe_line": qqe_line, "qqe_fast": rma_list})

    # Seed the trailing bands at the first valid bar.
    long_prev: float = rma_list[start] - dar_list[start]  # type: ignore[operator]
    short_prev: float = rma_list[start] + dar_list[start]  # type: ignore[operator]
    qqe_prev: float = rma_list[start]  # type: ignore[assignment]
    qqe_line[start] = qqe_prev

    for idx in range(start + 1, n):
        rma = rma_list[idx]
        d = dar_list[idx]
        if rma is None or d is None:
            qqe_line[idx] = None
            continue

        rma_prev = rma_list[idx - 1]

        # Ratchet long band: advances up when price is above it, resets otherwise.
        if rma > long_prev and rma_prev is not None and rma_prev > long_prev:
            long_curr = max(rma - d, long_prev)
        else:
            long_curr = rma - d

        # Ratchet short band: advances down when price is below it, resets otherwise.
        if rma < short_prev and rma_prev is not None and rma_prev < short_prev:
            short_curr = min(rma + d, short_prev)
        else:
            short_curr = rma + d

        # Assign QQE line to whichever side RSI is currently on.
        if rma > short_curr:
            qqe_curr = long_curr
        elif rma < long_curr:
            qqe_curr = short_curr
        elif qqe_prev == long_prev:
            qqe_curr = long_curr
        else:
            qqe_curr = short_curr

        qqe_line[idx] = qqe_curr
        long_prev, short_prev, qqe_prev = long_curr, short_curr, qqe_curr

    return pl.DataFrame({"qqe_line": qqe_line, "qqe_fast": rma_list})


# ---------------------------------------------------------------------------
# Connors RSI
# ---------------------------------------------------------------------------


def crsi(
    series: pl.Series,
    rsi_period: int = 3,
    streak_period: int = 2,
    rank_period: int = 100,
) -> pl.Series:
    """Connors RSI — composite momentum oscillator blending three RSI-family signals.

    Connors RSI averages three independently computed oscillators to produce a
    composite that is more responsive to short-term momentum than the classic
    14-period RSI while reducing noise via the diversification of components:

    1. **RSI(rsi_period)** — a short-period RSI applied directly to the close.
    2. **Streak RSI** — RSI(streak_period) applied to the up/down streak count:
       the streak is +n after n consecutive up bars and −n after n consecutive
       down bars.
    3. **Percent Rank(rank_period)** — the rolling percentile rank of the current
       close within the last rank_period closes.

    Connors RSI = (RSI + StreakRSI + PercentRank) / 3

    Null-prefix: ``max(rsi_period, streak_period, rank_period − 1)`` bars.
    With default parameters this is ``rank_period − 1 = 99`` bars.

    Args:
        series: Close price series.
        rsi_period: Period for the first RSI component (default 3).
        streak_period: Period for the streak RSI component (default 2).
        rank_period: Window for the percent-rank component (default 100).

    Returns:
        Series of Connors RSI values in [0, 100] named
        ``crsi_{rsi_period}_{streak_period}_{rank_period}``.

    Raises:
        ValueError: If any period is below its minimum.

    References:
        - Connors, L. A. & Alvarez, C. *Short-Term Trading Strategies
          That Work* (2009).
        - Investopedia — Connors RSI:
          https://www.investopedia.com/terms/c/connorsrsi.asp
    """
    _validate_period(rsi_period, "Connors RSI rsi_period", min_period=2)
    _validate_period(streak_period, "Connors RSI streak_period", min_period=2)
    _validate_period(rank_period, "Connors RSI rank_period")

    # Component 1: short-period RSI on the close.
    rsi_vals = rsi(series, rsi_period)

    # Component 2: RSI applied to the consecutive-up/down streak.
    # Build the streak in a Python loop since each bar depends on the prior value.
    raw: list[float | None] = series.to_list()
    n = len(raw)
    streak: list[float] = [0.0] * n
    for i in range(1, n):
        curr, prev_val = raw[i], raw[i - 1]
        if curr is None or prev_val is None:
            streak[i] = 0.0
        elif curr > prev_val:
            # Extend positive streak or start a new one.
            streak[i] = streak[i - 1] + 1.0 if streak[i - 1] > 0.0 else 1.0
        elif curr < prev_val:
            # Extend negative streak or start a new one.
            streak[i] = streak[i - 1] - 1.0 if streak[i - 1] < 0.0 else -1.0
        else:
            streak[i] = 0.0
    streak_series = pl.Series("streak", streak, dtype=pl.Float64)
    streak_rsi_vals = rsi(streak_series, streak_period)

    # Component 3: rolling percentile rank of the close (0 = lowest, 100 = highest).
    pr_vals = percent_rank(series, rank_period)

    # Average the three components; null propagates wherever any component is null.
    result = (rsi_vals + streak_rsi_vals + pr_vals) / 3.0
    return result.alias(f"crsi_{rsi_period}_{streak_period}_{rank_period}")


# ---------------------------------------------------------------------------
# Q-Stick
# ---------------------------------------------------------------------------


def qstick(ohlc: pl.DataFrame, period: int = 8) -> pl.Series:
    """Q-Stick — EMA of intraday close-minus-open directionality.

    Q-Stick, introduced by Tushar Chande, measures the average direction of
    intraday price movement by smoothing the difference between the close and
    the open.  Positive values indicate that closes are typically above opens
    (buying pressure); negative values indicate selling pressure.  Zero crossings
    signal changes in the prevailing intraday trend.

    Algorithm:
        body_diff = close − open   (positive for bullish, negative for bearish bars)
        Q-Stick   = EMA(body_diff, period)

    Null-prefix: ``period − 1`` bars.

    Args:
        ohlc: DataFrame with columns ``open`` and ``close``.
        period: EMA smoothing period (default 8).

    Returns:
        Series of Q-Stick values named ``qstick_{period}``.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - Chande, T. S. *Beyond Technical Analysis* (1997), pp. 155–157.
        - Investopedia — Q-Stick Indicator:
          https://www.investopedia.com/terms/q/qstick.asp
    """
    _validate_period(period, "Q-Stick")
    body_diff = ohlc["close"] - ohlc["open"]
    return ema(body_diff, period).alias(f"qstick_{period}")


# ---------------------------------------------------------------------------
# Psychological Line
# ---------------------------------------------------------------------------


def psy_line(series: pl.Series, period: int = 14) -> pl.Series:
    """Psychological Line — percentage of rising bars in a rolling window.

    The Psychological Line (PSY) counts how many of the last *period* bars
    closed higher than the previous bar, then expresses that count as a
    percentage.  Values above 50 indicate that bulls dominated the majority
    of recent bars; values below 50 indicate bear dominance.  Readings near
    100 or 0 may signal overbought/oversold conditions.

    Algorithm:
        is_up[t]    = 1 if close[t] > close[t-1] else 0   (null on bar 0)
        PSY[t]      = rolling_sum(is_up, period) / period × 100

    Null-prefix: ``period`` bars (the diff adds one extra null at bar 0).

    Args:
        series: Close price series.
        period: Rolling window length (default 14).

    Returns:
        Series of PSY values in [0, 100] named ``psy_{period}``.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - Investopedia — Psychological Line:
          https://www.investopedia.com/terms/p/psychological-line.asp
    """
    _validate_period(period, "Psychological Line")
    # diff[0] is null; the comparison propagates null, keeping bar 0 null.
    # rolling_sum with min_samples=period therefore waits for period non-null
    # values, producing exactly period leading nulls.
    diff = series.diff(1)
    is_up = (diff > 0).cast(pl.Float64)
    psy = is_up.rolling_sum(window_size=period, min_samples=period) / period * 100.0
    return psy.alias(f"psy_{period}")


# ---------------------------------------------------------------------------
# Rate of Change Ratio (ROCR)
# ---------------------------------------------------------------------------


def rocr(series: pl.Series, period: int = 10) -> pl.Series:
    """Rate of Change Ratio — close / close[n] (ratio form of ROC).

    ROCR expresses the current close as a ratio relative to the close
    *period* bars ago, producing a dimensionless multiplier.  A value of 1.0
    means no change; values > 1.0 indicate appreciation; values < 1.0
    indicate depreciation.  ROCR = 1 + ROC / 100.

    Algorithm:
        ROCR[t] = close[t] / close[t − period]

    Null-prefix: ``period`` bars.

    Args:
        series: Close price series.
        period: Lookback shift (default 10).

    Returns:
        Series of ROCR values named ``rocr_{period}``.
        The first ``period`` values are ``null``.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - TA-Lib ROCR: https://ta-lib.org/function.html#ROCR
        - Investopedia — Rate of Change:
          https://www.investopedia.com/terms/r/rateofchange.asp
    """
    _validate_period(period, "ROCR")
    prev = series.shift(period)
    # Avoid division by zero when a prior close is zero (rare, but possible).
    safe_prev = prev.replace(0.0, float("nan"))
    return (series / safe_prev).alias(f"rocr_{period}")


# ---------------------------------------------------------------------------
# Disparity Index
# ---------------------------------------------------------------------------


def disparity_index(series: pl.Series, period: int = 14) -> pl.Series:
    """Disparity Index — percentage deviation of price from its moving average.

    The Disparity Index (DI) measures how far the current price has strayed
    from its simple moving average, expressed as a percentage.  Positive
    values mean price is above the SMA (potential overbought); negative values
    mean price is below (potential oversold).  Unlike oscillators such as RSI,
    DI is price-scale agnostic and normalises deviation by the SMA itself.

    Algorithm:
        SMA[t]       = SMA(close, period)
        DI[t]        = (close[t] − SMA[t]) / SMA[t] × 100

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Close price series.
        period: SMA lookback (default 14).

    Returns:
        Series of Disparity Index values (%) named ``disparity_{period}``.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - Investopedia — Disparity Index:
          https://www.investopedia.com/terms/d/disparityindex.asp
        - Kaufman, P. J. *Trading Systems and Methods*, 5th ed. (2013), p. 70.
    """
    _validate_period(period, "Disparity Index")
    sma_vals = sma(series, period)
    safe_sma = sma_vals.replace(0.0, float("nan"))
    return ((series - sma_vals) / safe_sma * 100.0).alias(f"disparity_{period}")


# ---------------------------------------------------------------------------
# Absolute Price Oscillator (APO)
# ---------------------------------------------------------------------------


def apo(series: pl.Series, fast: int = 12, slow: int = 26) -> pl.Series:
    """Absolute Price Oscillator — difference of two EMAs in price units.

    The APO (sometimes called the Price Oscillator) subtracts a slow EMA from
    a fast EMA.  Unlike the PPO it reports the absolute difference in price
    units rather than a percentage, making it easier to overlay on a price
    chart.  A positive APO means the fast EMA is above the slow EMA
    (bullish momentum); a negative APO indicates bearish momentum.

    Algorithm:
        APO[t] = EMA(close, fast) − EMA(close, slow)

    Null-prefix: ``slow − 1`` bars (the slower EMA dominates).

    Args:
        series: Close price series.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).

    Returns:
        Series of APO values named ``apo_{fast}_{slow}``.

    Raises:
        ValueError: If ``fast < 1``, ``slow < 1``, or ``fast >= slow``.

    References:
        - Investopedia — Price Oscillator:
          https://www.investopedia.com/terms/p/ppo.asp
        - TA-Lib APO: https://ta-lib.org/function.html#APO
    """
    _validate_period(fast, "APO fast")
    _validate_period(slow, "APO slow")
    if fast >= slow:
        raise ValueError(f"APO fast ({fast}) must be less than slow ({slow}).")
    return (ema(series, fast) - ema(series, slow)).alias(f"apo_{fast}_{slow}")


# ---------------------------------------------------------------------------
# Accumulative Swing Index (ASI)
# ---------------------------------------------------------------------------


def asi(ohlc: pl.DataFrame, limit_move: float = 3.0) -> pl.Series:
    """Accumulative Swing Index — Wilder's cumulative directional indicator.

    The Swing Index (SI) quantifies intraday price action on a single bar by
    comparing the current bar's close to the prior bar's close, open, and range,
    with a normalisation factor ``R`` that scales the sensitivity.  The
    Accumulative Swing Index (ASI) is the running cumulative sum of SI and is
    used to confirm breakouts: a breakout in price should be accompanied by a
    new high or low in ASI.

    Wilder's SI formula (1978):
        K         = max(|H − Cp|, |L − Cp|)
        R chosen by the largest of |H − Cp|, |L − Cp|, |H − L|:
            case 1 (|H − Cp| ≥ both): R = |H − Cp| − 0.5|L − Cp| + 0.25|Cp − Op|
            case 2 (|L − Cp| ≥ both): R = |L − Cp| − 0.5|H − Cp| + 0.25|Cp − Op|
            case 3 (otherwise):        R = |H − L|  + 0.25|Cp − Op|
        SI[t] = 50 × (C − Cp + 0.5(C − O) + 0.25(Cp − Op)) / R × (K / T)
        ASI[t] = cumulative sum of SI

    Null-prefix: ``1`` bar (no prior OHLC data at bar 0).

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        limit_move: Maximum allowable daily price move (T in Wilder's formula).
                    Use the exchange-defined limit; default 3.0 is a common
                    placeholder for equities.

    Returns:
        Series of ASI values named ``asi``.  The first value is ``null``.

    References:
        - Wilder, J. W. *New Concepts in Technical Trading Systems* (1978),
          Chapter 9.
        - Investopedia — Accumulative Swing Index:
          https://www.investopedia.com/terms/a/asi.asp
    """
    h = ohlc["high"]
    lo = ohlc["low"]
    c = ohlc["close"]
    o = ohlc["open"]
    cp = c.shift(1)
    op = o.shift(1)

    # Absolute distances used throughout the formula.
    a = (h - cp).abs()
    b = (lo - cp).abs()
    c_hl = (h - lo).abs()
    d = (cp - op).abs()

    # K = larger of |H − Cp| and |L − Cp|.
    k = pl.select(pl.max_horizontal(a, b)).to_series()

    # R selection via the dominant distance.
    r = (
        pl.when((a >= b) & (a >= c_hl))
        .then(a - 0.5 * b + 0.25 * d)
        .when((b >= a) & (b >= c_hl))
        .then(b - 0.5 * a + 0.25 * d)
        .otherwise(c_hl + 0.25 * d)
    )
    safe_r = r.replace(0.0, float("nan"))

    numerator = c - cp + 0.5 * (c - o) + 0.25 * (cp - op)
    si = 50.0 * numerator / safe_r * (k / limit_move)

    # Fill NaN (degenerate zero-range bars) to 0 before accumulation.
    # Restore null at bar 0 (no prior bar), then accumulate.
    si_clean = si.fill_nan(0.0)
    asi_vals = si_clean.fill_null(0.0).cum_sum()
    # Re-apply the leading null: bar 0 has no prior context.
    result = pl.when(si.is_null()).then(pl.lit(None, dtype=pl.Float64)).otherwise(asi_vals)
    return pl.select(result).to_series().alias("asi")


# ---------------------------------------------------------------------------
# Price Momentum Oscillator (PMO)
# ---------------------------------------------------------------------------


def pmo(series: pl.Series, fast: int = 35, slow: int = 20) -> pl.Series:
    """Price Momentum Oscillator — double-smoothed rate-of-change.

    The PMO (Carl Swenlin, 1994) takes the daily percentage rate-of-change,
    scales it by 10, then applies two successive EMAs to reduce noise.  It is
    momentum-based like MACD but uses ROC as input rather than raw price,
    making it more responsive to acceleration and deceleration of price moves.

    Algorithm:
        ROC1[t]  = (close[t] / close[t−1] − 1) × 100        (1-period % change)
        PMO1[t]  = EMA(ROC1 × 10, fast)                      (first smoothing)
        PMO[t]   = EMA(PMO1, slow)                           (second smoothing)

    Null-prefix: ``fast + slow − 1`` bars.
    - ROC1 contributes 1 null (shift by 1).
    - First EMA adds ``fast − 1`` nulls → ``fast`` total.
    - Second EMA adds ``slow − 1`` nulls → ``fast + slow − 1`` total.

    Args:
        series: Close price series.
        fast: Period of the first EMA smoothing (default 35).
        slow: Period of the second EMA smoothing (default 20).

    Returns:
        Series of PMO values named ``pmo_{fast}_{slow}``.

    Raises:
        ValueError: If ``fast < 1`` or ``slow < 1``.

    References:
        - Swenlin, C. "Price Momentum Oscillator," *Technical Analysis of
          Stocks & Commodities* (1994).
        - StockCharts — PMO:
          https://school.stockcharts.com/doku.php?id=technical_indicators:price_momentum_oscillator_pmo
    """
    _validate_period(fast, "PMO fast")
    _validate_period(slow, "PMO slow")
    # ROC as a fraction × 10; shift(1) introduces the first null.
    prev = series.shift(1)
    safe_prev = prev.replace(0.0, float("nan"))
    roc1 = (series / safe_prev - 1.0) * 1000.0  # ×100 for % then ×10 = ×1000
    pmo1 = ema(roc1, fast)
    return ema(pmo1, slow).alias(f"pmo_{fast}_{slow}")


# ---------------------------------------------------------------------------
# Chande Trend Score
# ---------------------------------------------------------------------------


def chande_trend_score(series: pl.Series, period: int = 20) -> pl.Series:
    """Chande Trend Score — percentage of prior bars the current close exceeds.

    For each bar, the Chande Trend Score (CTS) looks back over the preceding
    *period* bars and counts how many of them had a close below the current
    bar's close.  The count is divided by *period* and expressed as a
    percentage.  A score near 100 means the current bar is above virtually
    all recent history (strong uptrend); near 0 means the opposite.

    Unlike the Psychological Line (which counts rising bar-over-bar changes),
    CTS compares the current price directly to each of the *period* prior
    prices, measuring the current bar's percentile within its own lookback.

    Algorithm (rolling window of size ``period + 1``):
        window = [close[t−period], …, close[t−1], close[t]]
        CTS[t] = (# prior bars with close < close[t]) / period × 100

    Null-prefix: ``period`` bars.

    Args:
        series: Close price series.
        period: Number of prior bars to compare against (default 20).

    Returns:
        Series of CTS values in [0, 100] named ``cts_{period}``.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - Chande, T. S. & Kroll, S. *The New Technical Trader* (1994),
          Chapter 4.
        - Kaufman, P. J. *Trading Systems and Methods*, 5th ed. (2013).
    """
    _validate_period(period, "Chande Trend Score")

    def _cts_window(w: pl.Series) -> float:
        """Count how many of the first `period` elements are below the last."""
        vals: list[float] = w.to_list()
        current = vals[-1]
        # Count prior bars strictly below current close.
        count = sum(1 for v in vals[:-1] if v < current)
        return count / period * 100.0

    # Window = period prior bars + current bar → size period + 1.
    return series.rolling_map(
        function=_cts_window,
        window_size=period + 1,
        min_samples=period + 1,
    ).alias(f"cts_{period}")


# ---------------------------------------------------------------------------
# Double-smoothed Stochastic (Bressert DSS)
# ---------------------------------------------------------------------------


def dss(
    ohlc: pl.DataFrame,
    period: int = 13,
    smooth: int = 8,
) -> pl.DataFrame:
    """Double-smoothed Stochastic — Bressert DSS oscillator.

    The Bressert DSS applies the stochastic formula twice in sequence, with
    EMA smoothing between passes, producing a smoother oscillator than the
    classic stochastic that retains sensitivity to genuine turning points.
    Values oscillate between 0 and 100; readings above 80 are considered
    overbought and below 20 oversold; zero-line crossovers of the signal
    line are entry signals.

    Algorithm:
        1. First stochastic over *period* bars:
               HH   = rolling_max(high, period)
               LL   = rolling_min(low,  period)
               %K   = (close − LL) / (HH − LL) × 100
        2. First EMA smoothing:
               ema1 = EMA(%K, smooth)
        3. Second stochastic applied to ema1:
               HH2  = rolling_max(ema1, period)
               LL2  = rolling_min(ema1, period)
               DSS  = (ema1 − LL2) / (HH2 − LL2) × 100
        4. Signal line:
               signal = EMA(DSS, smooth)

    Null-prefix: ``2 × (period − 1) + 2 × (smooth − 1)`` bars for the signal
    (two rolling windows + two EMA passes).

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: Stochastic lookback window (default 13).
        smooth: EMA smoothing period applied at each of the two passes (default 8).

    Returns:
        DataFrame with columns ``dss`` and ``dss_signal``.

    Raises:
        ValueError: If ``period < 1`` or ``smooth < 1``.

    References:
        - Bressert, W. "Stochastic Pop and Drop," *Stocks & Commodities* (1991).
        - Investopedia — Stochastic Oscillator:
          https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """
    _validate_period(period, "DSS")
    _validate_period(smooth, "DSS smooth")

    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    # First stochastic pass: normalise close within the period's H/L range.
    hh = high.rolling_max(window_size=period, min_samples=period)
    ll = low.rolling_min(window_size=period, min_samples=period)
    hl_range = hh - ll
    # Flat markets (hh == ll) produce 0/0 = NaN; treat as 50 (neutral midpoint).
    stoch_k = ((close - ll) / hl_range.replace(0.0, float("nan")) * 100.0).fill_nan(50.0)

    # First EMA smoothing pass reduces noise before the second stochastic.
    ema1 = ema(stoch_k, smooth)

    # Second stochastic applied to the smoothed stochastic.
    hh2 = ema1.rolling_max(window_size=period, min_samples=period)
    ll2 = ema1.rolling_min(window_size=period, min_samples=period)
    hl_range2 = hh2 - ll2
    dss_vals = (
        ((ema1 - ll2) / hl_range2.replace(0.0, float("nan")) * 100.0).fill_nan(50.0).alias("dss")
    )

    # Signal line from a final EMA smoothing of the DSS output.
    dss_signal = ema(dss_vals, smooth).alias("dss_signal")

    return pl.DataFrame({"dss": dss_vals, "dss_signal": dss_signal})


# ---------------------------------------------------------------------------
# Volume-weighted RSI
# ---------------------------------------------------------------------------


def vwrsi(ohlc_vol: pl.DataFrame, period: int = 14) -> pl.Series:
    """Volume-weighted RSI (VW-RSI).

    VW-RSI replaces the plain price change used in standard RSI with a
    volume-weighted price change: each bar's up-move or down-move is
    multiplied by the bar's volume before Wilder smoothing.  Bars with high
    volume on up-moves push the oscillator toward 100; high volume on
    down-moves push it toward 0.  This links oscillator momentum to actual
    market participation, making divergences more meaningful than with plain RSI.

    Algorithm:
        delta[t]   = close[t] − close[t−1]
        up_vw[t]   = max(delta[t], 0) × volume[t]   (volume on up bars)
        down_vw[t] = max(−delta[t], 0) × volume[t]  (volume on down bars)
        avg_up     = Wilder_smooth(up_vw,   period)
        avg_down   = Wilder_smooth(down_vw, period)
        RS         = avg_up / avg_down
        VW-RSI     = 100 − 100 / (1 + RS)

    Null-prefix: ``period`` bars (1 from diff + period − 1 from Wilder smoothing).

    Args:
        ohlc_vol: DataFrame with columns ``close`` and ``volume``.
        period: Wilder smoothing period (default 14; must be ≥ 2).

    Returns:
        Series of VW-RSI values in [0, 100] named ``vwrsi_{period}``.

    Raises:
        ValueError: If ``period < 2``.

    References:
        - Blau, W. *Momentum, Direction, and Divergence* (1995), Chapter 6.
        - Investopedia — Relative Strength Index (RSI):
          https://www.investopedia.com/terms/r/rsi.asp
    """
    _validate_period(period, "VW-RSI", min_period=2)

    close = ohlc_vol["close"]
    vol = ohlc_vol["volume"].cast(pl.Float64)

    # Bar-to-bar price changes; null propagates from diff at bar 0.
    delta = close.diff(n=1)

    # Volume-weighted directional flows; null × volume → null for bar 0.
    up_vw = delta.clip(lower_bound=0.0) * vol
    down_vw = (-delta).clip(lower_bound=0.0) * vol

    avg_up = wilder_smooth(up_vw, period)
    avg_down = wilder_smooth(down_vw, period)

    # All-up bars → avg_down == 0 → RS = inf → VW-RSI = 100.
    rs = avg_up / avg_down.replace(0.0, float("nan"))
    return (100.0 - 100.0 / (1.0 + rs)).fill_nan(100.0).alias(f"vwrsi_{period}")
