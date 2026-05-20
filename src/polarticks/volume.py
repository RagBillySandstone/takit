"""
Volume and price-volume indicators.

Functions
---------
vwap              Session-anchored Volume Weighted Average Price
vwap_bands        VWAP with ±1σ / ±2σ standard-deviation bands
obv               On-Balance Volume (running signed cumulative volume)
ad_line           Accumulation/Distribution Line (volume-weighted cumulative flow)
kvo               Klinger Volume Oscillator (trend-aligned cumulative volume force)
eom               Ease of Movement (price change relative to volume pressure)
pvt               Price Volume Trend (cumulative volume scaled by % price change)
force_index       Elder's Force Index — EMA of signed price-change × volume
nvi               Negative Volume Index — cumulates price change on falling-volume days
pvi               Positive Volume Index — cumulates price change on rising-volume days
chaikin_osc       Chaikin Oscillator — EMA difference of the A/D Line
volume_oscillator Volume Oscillator — percentage difference of two volume EMAs
twap              Time-Weighted Average Price (cumulative or rolling equal-weight)
rvol              Relative Volume — current volume relative to its rolling average
obv_osc           OBV Oscillator — EMA spread of On-Balance Volume
volume_roc        Volume Rate of Change — percentage change in trading volume
"""

from __future__ import annotations

import math

import polars as pl

from polarticks._validate import _validate_period
from polarticks.moving_averages import ema


def vwap(
    ohlc_vol: pl.DataFrame,
    session_start_hour: int = 22,
) -> pl.Series:
    """Session-anchored Volume Weighted Average Price (VWAP).

    VWAP is the cumulative sum of (typical_price × volume) divided by the
    cumulative volume, reset at the start of each trading session.  It acts
    as a dynamic fair-value benchmark; price above VWAP is considered
    bullish context, below is bearish.

    Session boundary:
        A new session begins on every bar whose UTC hour equals
        *session_start_hour*.  For FX this is conventionally 22:00 UTC
        (New York close / Wellington open), but any hour can be supplied.
        If no ``time`` column is present, the entire series is treated as
        a single session.

    Typical price = (high + low + close) / 3.

    Args:
        ohlc_vol: DataFrame with columns ``high``, ``low``, ``close``,
                  ``volume``, and optionally ``time`` (Datetime).
        session_start_hour: UTC hour at which a new session begins (default 22).

    Returns:
        Series of VWAP values aligned with the input rows.
    """
    high = ohlc_vol["high"]
    low = ohlc_vol["low"]
    close = ohlc_vol["close"]
    volume = ohlc_vol["volume"].cast(pl.Float64)

    tp = (high + low + close) / 3.0
    tp_vol = tp * volume

    # Detect session boundaries using the time column when available.
    if "time" in ohlc_vol.columns:
        hours = ohlc_vol["time"].dt.hour()
        # A session starts on the first bar of the dataset OR on bars whose
        # hour matches session_start_hour.
        is_session_start = (hours == session_start_hour) | (
            pl.Series([True] + [False] * (len(ohlc_vol) - 1))
        )
    else:
        # No time column: entire series is one session — vectorise directly.
        return (tp_vol.cum_sum() / volume.cum_sum()).alias("vwap")

    # Walk through rows accumulating cumulative sums, resetting on each session
    # boundary.  A vectorised alternative (group_by session ID + cum_sum) is
    # possible but requires restructuring the data; the loop is kept for clarity.
    tp_vol_list = tp_vol.to_list()
    vol_list = volume.to_list()
    starts = is_session_start.to_list()

    vwap_values: list[float] = []
    running_tp_vol = 0.0
    running_vol = 0.0

    for tpv, vol, is_start in zip(tp_vol_list, vol_list, starts, strict=True):
        if is_start:
            running_tp_vol = 0.0
            running_vol = 0.0
        running_tp_vol += tpv
        running_vol += vol
        # Avoid division by zero on bars with zero volume.
        vwap_values.append(running_tp_vol / running_vol if running_vol > 0 else float("nan"))

    return pl.Series("vwap", vwap_values, dtype=pl.Float64)


def vwap_bands(
    ohlc_vol: pl.DataFrame,
    session_start_hour: int = 22,
) -> pl.DataFrame:
    """Session-anchored VWAP with ±1σ and ±2σ standard-deviation bands.

    Extends VWAP by also computing the cumulative session variance of the
    typical price weighted by volume.  The standard deviation at each bar is:

        σ = sqrt(Σ(tp² × vol) / Σ(vol)  −  VWAP²)

    This is the population standard deviation of the volume-weighted typical
    price distribution up to that bar within the session.

    Session handling follows :func:`vwap` exactly: resets at
    ``session_start_hour`` UTC when a ``time`` column is present; otherwise
    treats the entire series as one session.

    Args:
        ohlc_vol: DataFrame with columns ``high``, ``low``, ``close``,
                  ``volume``, and optionally ``time`` (Datetime).
        session_start_hour: UTC hour at which a new session begins (default 22).

    Returns:
        DataFrame with columns ``vwap``, ``upper_1``, ``lower_1``,
        ``upper_2``, and ``lower_2``.
    """
    high = ohlc_vol["high"]
    low = ohlc_vol["low"]
    close = ohlc_vol["close"]
    volume = ohlc_vol["volume"].cast(pl.Float64)

    tp = (high + low + close) / 3.0
    tp_vol = tp * volume
    tp2_vol = tp * tp * volume

    if "time" in ohlc_vol.columns:
        hours = ohlc_vol["time"].dt.hour()
        is_session_start = (hours == session_start_hour) | (
            pl.Series([True] + [False] * (len(ohlc_vol) - 1))
        )
    else:
        # No time column: entire series is one session — vectorise directly.
        cum_tp_vol_s = tp_vol.cum_sum()
        cum_tp2_vol_s = tp2_vol.cum_sum()
        cum_vol_s = volume.cum_sum()
        vwap_vals = cum_tp_vol_s / cum_vol_s
        # Population variance: E[tp²] − E[tp]²; clamp to 0 for float rounding.
        variance = (cum_tp2_vol_s / cum_vol_s - vwap_vals**2).clip(lower_bound=0.0)
        std_vals = variance**0.5
        return pl.DataFrame(
            {
                "vwap": vwap_vals,
                "upper_1": vwap_vals + std_vals,
                "lower_1": vwap_vals - std_vals,
                "upper_2": vwap_vals + 2.0 * std_vals,
                "lower_2": vwap_vals - 2.0 * std_vals,
            },
            schema={
                "vwap": pl.Float64,
                "upper_1": pl.Float64,
                "lower_1": pl.Float64,
                "upper_2": pl.Float64,
                "lower_2": pl.Float64,
            },
        )

    tp_vol_list = tp_vol.to_list()
    tp2_vol_list = tp2_vol.to_list()
    vol_list = volume.to_list()
    starts = is_session_start.to_list()

    vwap_values: list[float] = []
    upper_1: list[float] = []
    lower_1: list[float] = []
    upper_2: list[float] = []
    lower_2: list[float] = []

    cum_tp_vol = 0.0
    cum_tp2_vol = 0.0
    cum_vol = 0.0

    for tpv, tp2v, vol, is_start in zip(tp_vol_list, tp2_vol_list, vol_list, starts, strict=True):
        if is_start:
            cum_tp_vol = 0.0
            cum_tp2_vol = 0.0
            cum_vol = 0.0

        cum_tp_vol += tpv
        cum_tp2_vol += tp2v
        cum_vol += vol

        if cum_vol > 0:
            vwap_val = cum_tp_vol / cum_vol
            # Population variance: E[tp²] - E[tp]² (volume-weighted).
            variance = max(0.0, cum_tp2_vol / cum_vol - vwap_val * vwap_val)
            std = math.sqrt(variance)
        else:
            vwap_val = float("nan")
            std = float("nan")

        vwap_values.append(vwap_val)
        upper_1.append(vwap_val + std)
        lower_1.append(vwap_val - std)
        upper_2.append(vwap_val + 2.0 * std)
        lower_2.append(vwap_val - 2.0 * std)

    return pl.DataFrame(
        {
            "vwap": vwap_values,
            "upper_1": upper_1,
            "lower_1": lower_1,
            "upper_2": upper_2,
            "lower_2": lower_2,
        },
        schema={
            "vwap": pl.Float64,
            "upper_1": pl.Float64,
            "lower_1": pl.Float64,
            "upper_2": pl.Float64,
            "lower_2": pl.Float64,
        },
    )


def obv(ohlc_vol: pl.DataFrame) -> pl.Series:
    """On-Balance Volume (OBV) — running signed cumulative volume.

    OBV accumulates volume with a sign determined by whether the close
    is higher or lower than the prior close:

        - close[t] > close[t-1]: add volume (buying pressure).
        - close[t] < close[t-1]: subtract volume (selling pressure).
        - close[t] = close[t-1]: no change.

    The first bar contributes zero volume (no prior close exists to compare).

    OBV is used to confirm price trends: rising OBV alongside rising price
    confirms an uptrend; divergence may signal weakening momentum.

    Args:
        ohlc_vol: DataFrame with columns ``close`` and ``volume``.

    Returns:
        Series of cumulative OBV values (dtype Float64).
    """
    close = ohlc_vol["close"]
    volume = ohlc_vol["volume"].cast(pl.Float64)

    prev_close = close.shift(1)

    # pl.when/then/otherwise with Series operands produces a lazy Expr; materialise
    # it via pl.select() so the result is a concrete Series.
    signed_volume: pl.Series = pl.select(
        pl.when(close > prev_close)
        .then(volume)
        .when(close < prev_close)
        .then(-volume)
        .otherwise(pl.lit(0.0))
    ).to_series()

    # Bar 0 has no prior close, so its signed volume is null → treat as zero.
    return signed_volume.fill_null(0.0).cum_sum().alias("obv")


# ---------------------------------------------------------------------------
# A/D Line
# ---------------------------------------------------------------------------


def ad_line(ohlc_vol: pl.DataFrame) -> pl.Series:
    """Accumulation/Distribution Line — volume-weighted cumulative money flow.

    The A/D Line accumulates volume scaled by the *Money Flow Multiplier*,
    which measures where the close sits within the bar's high-low range.
    A close at the top of the range (strong buying) adds the full volume;
    a close at the bottom (strong selling) subtracts it.

    Algorithm:
        money_flow_multiplier = (2 × close − high − low) / (high − low)
        money_flow_volume     = multiplier × volume
        A/D[t]               = Σ money_flow_volume[0..t]

    Doji bars (high == low, range = 0) contribute zero to the line.  The
    A/D Line is closely related to OBV but is more nuanced: a bar where the
    close is above the midpoint contributes positive flow even if price fell
    from the prior close.

    No leading nulls — the line starts accumulating from bar 0.

    Args:
        ohlc_vol: DataFrame with columns ``high``, ``low``, ``close``,
            ``volume``.

    Returns:
        Series of cumulative A/D values (dtype Float64).
    """
    high = ohlc_vol["high"]
    low = ohlc_vol["low"]
    close = ohlc_vol["close"]
    volume = ohlc_vol["volume"].cast(pl.Float64)

    hl_range = high - low

    # fill_nan zeroes out zero-range (doji-like) bars rather than propagating NaN.
    mfm = ((2.0 * close - high - low) / hl_range).fill_nan(0.0)
    mfv = mfm * volume

    return mfv.cum_sum().alias("ad_line")


# ---------------------------------------------------------------------------
# Klinger Volume Oscillator
# ---------------------------------------------------------------------------


def kvo(
    ohlc_vol: pl.DataFrame,
    fast: int = 34,
    slow: int = 55,
    signal: int = 13,
) -> pl.DataFrame:
    """Klinger Volume Oscillator — trend-aligned cumulative volume force.

    The KVO (Stephen Klinger, 1997) constructs a Volume Force (VF) that
    scales each bar's volume by the price-trend direction and by how much
    the daily range falls inside or outside a cumulative range window.  Short
    and long EMAs of VF are subtracted to form the oscillator.

    Algorithm:
        trend[t]   = +1 if (H+L+C)[t] > (H+L+C)[t-1] else −1
        dm[t]      = high[t] − low[t]
        cm[t]      = cm[t-1] + dm[t]        if trend unchanged
                   = dm[t-1] + dm[t]        if trend reversed
        vf[t]      = 100 × volume × trend × |2 × dm / cm − 1|
        kvo_line   = EMA(vf, fast) − EMA(vf, slow)
        kvo_signal = EMA(kvo_line, signal)

    Null-prefix for ``kvo_line``:   ``slow − 1`` bars.
    Null-prefix for ``kvo_signal``: ``slow + signal − 2`` bars.

    Args:
        ohlc_vol: DataFrame with columns ``high``, ``low``, ``close``, ``volume``.
        fast: Fast EMA period (default 34).
        slow: Slow EMA period (default 55).
        signal: Signal line EMA period (default 13).

    Returns:
        DataFrame with columns ``kvo_line`` and ``kvo_signal``.

    Raises:
        ValueError: If ``fast >= slow`` or any period < 1.
    """
    _validate_period(fast, "KVO fast")
    _validate_period(slow, "KVO slow")
    _validate_period(signal, "KVO signal")
    if fast >= slow:
        raise ValueError(f"KVO fast ({fast}) must be less than slow ({slow}).")

    highs = ohlc_vol["high"].to_list()
    lows = ohlc_vol["low"].to_list()
    closes = ohlc_vol["close"].to_list()
    vols = ohlc_vol["volume"].cast(pl.Float64).to_list()
    n = len(highs)

    vf_list: list[float] = [0.0] * n  # bar 0 has no prior bar → zero contribution
    prev_trend = 1
    cm = float(highs[0] - lows[0]) if (highs[0] is not None and lows[0] is not None) else 0.0

    for i in range(1, n):
        h, lo, c, v = highs[i], lows[i], closes[i], vols[i]
        ph, plo = highs[i - 1], lows[i - 1]
        if any(x is None for x in (h, lo, c, v, ph, plo, closes[i - 1])):
            continue

        trend = 1 if (h + lo + c) > (ph + plo + closes[i - 1]) else -1
        dm = h - lo  # today's high-low range
        dm_prev = ph - plo  # prior bar's range

        if trend == prev_trend:
            cm += dm
        else:
            cm = dm_prev + dm

        prev_trend = trend

        if cm != 0.0:
            vf_list[i] = 100.0 * v * trend * abs(2.0 * dm / cm - 1.0)

    vf = pl.Series("vf", vf_list, dtype=pl.Float64)
    kvo_line = (ema(vf, fast) - ema(vf, slow)).alias("kvo_line")
    kvo_signal_s = ema(kvo_line, signal).alias("kvo_signal")

    return pl.DataFrame({"kvo_line": kvo_line, "kvo_signal": kvo_signal_s})


# ---------------------------------------------------------------------------
# Ease of Movement
# ---------------------------------------------------------------------------


def eom(
    ohlc_vol: pl.DataFrame,
    period: int = 14,
    divisor: float = 10_000.0,
) -> pl.Series:
    """Ease of Movement — price change relative to volume pressure.

    EOM (Richard Arms, 1989) relates how far price midpoints move to the
    volume required per unit of price range.  A large positive EOM means
    prices advanced easily (little volume needed); a large negative value
    means prices fell with little resistance.

    Algorithm:
        midpoint  = (high + low) / 2
        distance  = midpoint[t] − midpoint[t-1]
        box_ratio = (volume / divisor) / (high − low)
        raw_eom   = distance / box_ratio
        EOM[t]    = SMA(raw_eom, period)

    Zero-range bars (high = low) contribute 0 to the rolling mean.
    Null-prefix: ``period`` bars (1 from midpoint shift; period − 1 from SMA).

    Args:
        ohlc_vol: DataFrame with columns ``high``, ``low``, ``volume``.
        period: SMA smoothing period (default 14).
        divisor: Volume scale factor to bring raw EOM into a convenient range
                 (default 10 000).

    Returns:
        Series named ``eom_{period}``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "EOM")

    mid = (ohlc_vol["high"] + ohlc_vol["low"]) / 2.0
    distance = mid - mid.shift(1)

    hl_range = ohlc_vol["high"] - ohlc_vol["low"]
    # replace 0.0 with NaN so division produces NaN for zero-range bars.
    safe_range = hl_range.replace(0.0, float("nan"))
    box_ratio = (ohlc_vol["volume"].cast(pl.Float64) / divisor) / safe_range

    # fill_nan(0.0): zero-range bars did not move the midpoint meaningfully.
    raw_eom = (distance / box_ratio).fill_nan(0.0)

    return raw_eom.rolling_mean(window_size=period, min_samples=period).alias(f"eom_{period}")


# ---------------------------------------------------------------------------
# Price Volume Trend
# ---------------------------------------------------------------------------


def pvt(ohlc_vol: pl.DataFrame) -> pl.Series:
    """Price Volume Trend — cumulative volume scaled by fractional price change.

    PVT accumulates volume multiplied by the percentage price change at each
    bar.  Unlike OBV (which uses only the sign of the price change), PVT
    weights volume by the magnitude of the move, making it more sensitive to
    large-range sessions.

    Algorithm:
        pct_change = (close[t] − close[t-1]) / close[t-1]
        PVT[t]     = PVT[t-1] + volume × pct_change

    The first bar has no prior close → pct_change is treated as 0.
    No leading nulls — the line starts accumulating from bar 0.

    Args:
        ohlc_vol: DataFrame with columns ``close`` and ``volume``.

    Returns:
        Series named ``pvt`` (dtype Float64).
    """
    close = ohlc_vol["close"]
    volume = ohlc_vol["volume"].cast(pl.Float64)

    prev_close = close.shift(1)
    # fill_nan handles the zero-prev-close edge case (avoids inf).
    pct_change = ((close - prev_close) / prev_close).fill_nan(0.0)
    # Bar 0: prev_close is null → pct_change is null → treat as zero so PVT starts at 0.
    return (volume * pct_change.fill_null(0.0)).cum_sum().alias("pvt")


# ---------------------------------------------------------------------------
# Force Index
# ---------------------------------------------------------------------------


def force_index(ohlc_vol: pl.DataFrame, period: int = 13) -> pl.Series:
    """Elder's Force Index — EMA-smoothed signed price-change times volume.

    The Force Index (Dr Alexander Elder) measures the power behind a price
    move by combining direction (close − previous close), magnitude, and
    volume.  Positive values indicate buying force; negative values indicate
    selling force.  The EMA smoothing filters out single-bar noise.

    Algorithm:
        raw_force[t]  = (close[t] − close[t-1]) × volume[t]
        force_index[t] = EMA(raw_force, period)

    The raw force for bar 0 is treated as zero (no prior close exists).
    Null-prefix: ``period − 1`` bars (from the EMA warm-up).

    Args:
        ohlc_vol: DataFrame with columns ``close`` and ``volume``.
        period: EMA smoothing period (default 13; use 2 for a faster signal).

    Returns:
        Series named ``force_index_{period}``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Force Index")

    close = ohlc_vol["close"]
    volume = ohlc_vol["volume"].cast(pl.Float64)

    # Bar 0 has no prior close; fill null to zero so the EMA seeds cleanly.
    raw = ((close - close.shift(1)) * volume).fill_null(0.0)

    return ema(raw, period).alias(f"force_index_{period}")


# ---------------------------------------------------------------------------
# Negative Volume Index
# ---------------------------------------------------------------------------


def nvi(ohlc_vol: pl.DataFrame) -> pl.Series:
    """Negative Volume Index (NVI) — cumulates price change on falling-volume bars.

    The NVI (Paul Dysart, popularised by Norman Fosback) is based on the
    premise that the "smart money" (informed traders) tends to be active on
    low-volume days.  The index only moves when volume is lower than the
    prior bar; on rising-volume days the index is unchanged.

    Algorithm:
        NVI[0] = 1000
        If volume[t] < volume[t-1]:
            NVI[t] = NVI[t-1] × (1 + (close[t] − close[t-1]) / close[t-1])
        Else:
            NVI[t] = NVI[t-1]

    No leading nulls — the index is defined for every bar from bar 0.

    Args:
        ohlc_vol: DataFrame with columns ``close`` and ``volume``.

    Returns:
        Series named ``nvi`` (dtype Float64), starting at 1000.
    """
    close_list = ohlc_vol["close"].to_list()
    vol_list = ohlc_vol["volume"].cast(pl.Float64).to_list()
    n = len(close_list)

    result: list[float] = [1000.0] * n

    for i in range(1, n):
        c, pc = close_list[i], close_list[i - 1]
        v, pv = vol_list[i], vol_list[i - 1]

        # Skip bars with missing data; carry the prior value forward.
        if c is None or pc is None or v is None or pv is None or pc == 0.0:
            result[i] = result[i - 1]
        elif v < pv:
            # Volume fell: accumulate price change.
            result[i] = result[i - 1] * (1.0 + (c - pc) / pc)
        else:
            result[i] = result[i - 1]

    return pl.Series("nvi", result, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# Positive Volume Index
# ---------------------------------------------------------------------------


def pvi(ohlc_vol: pl.DataFrame) -> pl.Series:
    """Positive Volume Index (PVI) — cumulates price change on rising-volume bars.

    The PVI (Paul Dysart) mirrors the NVI: it tracks what the "crowd" (less
    informed, momentum-driven traders) does on high-volume days.  The index
    only moves when volume is higher than the prior bar; it is unchanged on
    falling-volume days.

    Algorithm:
        PVI[0] = 1000
        If volume[t] > volume[t-1]:
            PVI[t] = PVI[t-1] × (1 + (close[t] − close[t-1]) / close[t-1])
        Else:
            PVI[t] = PVI[t-1]

    No leading nulls — the index is defined for every bar from bar 0.

    Args:
        ohlc_vol: DataFrame with columns ``close`` and ``volume``.

    Returns:
        Series named ``pvi`` (dtype Float64), starting at 1000.
    """
    close_list = ohlc_vol["close"].to_list()
    vol_list = ohlc_vol["volume"].cast(pl.Float64).to_list()
    n = len(close_list)

    result: list[float] = [1000.0] * n

    for i in range(1, n):
        c, pc = close_list[i], close_list[i - 1]
        v, pv = vol_list[i], vol_list[i - 1]

        # Skip bars with missing data; carry the prior value forward.
        if c is None or pc is None or v is None or pv is None or pc == 0.0:
            result[i] = result[i - 1]
        elif v > pv:
            # Volume rose: accumulate price change.
            result[i] = result[i - 1] * (1.0 + (c - pc) / pc)
        else:
            result[i] = result[i - 1]

    return pl.Series("pvi", result, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# Chaikin Oscillator
# ---------------------------------------------------------------------------


def chaikin_osc(
    ohlc_vol: pl.DataFrame,
    fast: int = 3,
    slow: int = 10,
) -> pl.Series:
    """Chaikin Oscillator — EMA divergence of the Accumulation/Distribution Line.

    The Chaikin Oscillator (Marc Chaikin) applies MACD-style EMA differencing
    to the A/D Line to measure momentum of money flow.  Values above zero
    indicate accumulation; below zero indicate distribution.

        ChaikinOsc = EMA(AD_Line, fast) − EMA(AD_Line, slow)

    Null-prefix: ``slow − 1`` bars.

    Args:
        ohlc_vol: DataFrame with columns ``high``, ``low``, ``close``, ``volume``.
        fast: Fast EMA period (default 3).
        slow: Slow EMA period (default 10).

    Returns:
        Series named ``chaikin_osc``.

    Raises:
        ValueError: If ``fast >= slow``.
    """
    if fast >= slow:
        raise ValueError(f"Chaikin Oscillator fast ({fast}) must be less than slow ({slow}).")

    adl = ad_line(ohlc_vol)
    return (ema(adl, fast) - ema(adl, slow)).alias("chaikin_osc")


# ---------------------------------------------------------------------------
# Volume Oscillator
# ---------------------------------------------------------------------------


def volume_oscillator(
    volume: pl.Series,
    fast: int = 5,
    slow: int = 10,
) -> pl.Series:
    """Volume Oscillator — percentage difference between two volume EMAs.

    Measures volume momentum by comparing a fast EMA of volume to a slow EMA.
    Positive values confirm price moves with rising volume; negative values
    suggest weakening volume that may precede a reversal.

        VO = 100 × (EMA(volume, fast) − EMA(volume, slow)) / EMA(volume, slow)

    Null-prefix: ``slow − 1`` bars.

    Args:
        volume: Volume series.
        fast: Fast EMA period (default 5).
        slow: Slow EMA period (default 10).

    Returns:
        Series named ``vol_osc``.

    Raises:
        ValueError: If ``fast >= slow``.
    """
    if fast >= slow:
        raise ValueError(f"Volume Oscillator fast ({fast}) must be less than slow ({slow}).")

    v = volume.cast(pl.Float64)
    fast_ema = ema(v, fast)
    slow_ema = ema(v, slow)

    # fill_nan handles the rare case where slow EMA is zero.
    return (100.0 * (fast_ema - slow_ema) / slow_ema).fill_nan(None).alias("vol_osc")


# ---------------------------------------------------------------------------
# TWAP
# ---------------------------------------------------------------------------


def twap(
    ohlc_vol: pl.DataFrame,
    period: int | None = None,
) -> pl.Series:
    """Time-Weighted Average Price — equal-weight rolling or cumulative average.

    TWAP weights each time bar equally, regardless of volume.  In the rolling
    form (``period`` specified), it is the SMA of the typical price.  In the
    cumulative form (``period=None``), it is the running mean from bar 0.

        typical_price = (high + low + close) / 3
        TWAP (rolling)    = SMA(typical_price, period)
        TWAP (cumulative) = cumsum(typical_price) / bar_count

    Null-prefix (rolling):    ``period − 1`` bars.
    Null-prefix (cumulative): 0 bars.

    Args:
        ohlc_vol: DataFrame with columns ``high``, ``low``, ``close``.
        period: Rolling window length; ``None`` for cumulative from bar 0 (default None).

    Returns:
        Series named ``twap_{period}`` (rolling) or ``twap`` (cumulative).

    Raises:
        ValueError: If ``period`` is specified and ``< 1``.
    """
    tp = (ohlc_vol["high"] + ohlc_vol["low"] + ohlc_vol["close"]) / 3.0

    if period is None:
        # Cumulative average: sum(tp) / number of bars seen so far.
        count = pl.Series(list(range(1, len(tp) + 1)), dtype=pl.Float64)
        return (tp.cum_sum() / count).alias("twap")

    _validate_period(period, "TWAP")
    return tp.rolling_mean(window_size=period, min_samples=period).alias(f"twap_{period}")


# ---------------------------------------------------------------------------
# Relative Volume (RVOL)
# ---------------------------------------------------------------------------


def rvol(ohlc_vol: pl.DataFrame, period: int = 20) -> pl.Series:
    """Relative Volume — current volume relative to its rolling SMA.

    RVOL normalises the current bar's volume by the simple moving average of
    volume over the last *period* bars, producing a dimensionless ratio.
    Values above 1.0 indicate above-average activity; values below 1.0
    indicate below-average activity.  Spikes in RVOL often accompany
    breakouts or news-driven moves.

    Algorithm:
        avg_vol[t] = SMA(volume, period)
        RVOL[t]    = volume[t] / avg_vol[t]

    Null-prefix: ``period − 1`` bars.

    Args:
        ohlc_vol: DataFrame with a ``volume`` column.
        period: Rolling SMA window for average volume (default 20).

    Returns:
        Series of RVOL values named ``rvol_{period}``.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - Investopedia — Relative Volume:
          https://www.investopedia.com/terms/r/relative-volume.asp
    """
    _validate_period(period, "RVOL")
    vol = ohlc_vol["volume"]
    avg_vol = vol.rolling_mean(window_size=period, min_samples=period)
    safe_avg = avg_vol.replace(0.0, float("nan"))
    return (vol / safe_avg).alias(f"rvol_{period}")


# ---------------------------------------------------------------------------
# OBV Oscillator
# ---------------------------------------------------------------------------


def obv_osc(ohlc_vol: pl.DataFrame, fast: int = 5, slow: int = 10) -> pl.Series:
    """OBV Oscillator — EMA spread of On-Balance Volume.

    The OBV Oscillator subtracts a slow EMA of OBV from a fast EMA of OBV,
    producing a momentum-style histogram centred around zero.  Positive values
    mean the fast EMA is above the slow EMA (accumulation accelerating);
    negative values mean the slow EMA is dominant (distribution).  Zero
    crossings are entry/exit signals.

    Algorithm:
        OBV[t]     = cumulative signed volume (see ``obv``)
        fast_ema   = EMA(OBV, fast)
        slow_ema   = EMA(OBV, slow)
        OSC[t]     = fast_ema[t] − slow_ema[t]

    Null-prefix: ``slow − 1`` bars.

    Args:
        ohlc_vol: DataFrame with columns ``close`` and ``volume``.
        fast: Fast EMA period for OBV (default 5).
        slow: Slow EMA period for OBV (default 10).

    Returns:
        Series of OBV Oscillator values named ``obv_osc_{fast}_{slow}``.

    Raises:
        ValueError: If ``fast < 1``, ``slow < 1``, or ``fast >= slow``.

    References:
        - Granville, J. *New Key to Stock Market Profits* (1963).
        - Investopedia — On-Balance Volume (OBV):
          https://www.investopedia.com/terms/o/onbalancevolume.asp
    """
    _validate_period(fast, "OBV Oscillator fast")
    _validate_period(slow, "OBV Oscillator slow")
    if fast >= slow:
        raise ValueError(f"OBV Oscillator fast ({fast}) must be less than slow ({slow}).")
    obv_vals = obv(ohlc_vol)
    fast_ema = ema(obv_vals, fast)
    slow_ema = ema(obv_vals, slow)
    return (fast_ema - slow_ema).alias(f"obv_osc_{fast}_{slow}")


# ---------------------------------------------------------------------------
# Volume Rate of Change (VROC)
# ---------------------------------------------------------------------------


def volume_roc(ohlc_vol: pl.DataFrame, period: int = 14) -> pl.Series:
    """Volume Rate of Change — percentage change in trading volume.

    VROC measures how much the current bar's volume has changed relative to
    the volume *period* bars ago, expressed as a percentage.  Surging VROC
    confirms breakouts; declining VROC during a price move may signal weak
    conviction.

    Algorithm:
        VROC[t] = (volume[t] − volume[t-period]) / volume[t-period] × 100

    Null-prefix: ``period`` bars.

    Args:
        ohlc_vol: DataFrame with a ``volume`` column.
        period: Lookback shift (default 14).

    Returns:
        Series of VROC values (%) named ``vroc_{period}``.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - Investopedia — Rate of Change (ROC):
          https://www.investopedia.com/terms/r/rateofchange.asp
    """
    _validate_period(period, "Volume ROC")
    vol = ohlc_vol["volume"]
    prev_vol = vol.shift(period)
    safe_prev = prev_vol.replace(0.0, float("nan"))
    return ((vol - prev_vol) / safe_prev * 100.0).alias(f"vroc_{period}")
