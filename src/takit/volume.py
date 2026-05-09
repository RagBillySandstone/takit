"""
Volume and price-volume indicators.

Functions
---------
vwap        Session-anchored Volume Weighted Average Price
vwap_bands  VWAP with ±1σ / ±2σ standard-deviation bands
obv         On-Balance Volume (running signed cumulative volume)
"""

from __future__ import annotations

import math

import polars as pl


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

    # Walk through rows and accumulate cumulative sums, resetting at each
    # session boundary.  Sequential by nature — resets cannot be vectorised.
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
