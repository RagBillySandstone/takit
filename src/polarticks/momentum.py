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
"""

from __future__ import annotations

import polars as pl

from polarticks._validate import _validate_period
from polarticks.moving_averages import ema, sma, wilder_smooth, wma

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
