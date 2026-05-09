"""
Volume and price-volume indicators.

Functions
---------
vwap    Session-anchored Volume Weighted Average Price
"""

from __future__ import annotations

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
        # No time column: treat entire series as one session.
        is_session_start = pl.Series([True] + [False] * (len(ohlc_vol) - 1))

    # Walk through rows and accumulate cumulative sums, resetting at each
    # session boundary.  This is inherently sequential so a Python loop is used;
    # for M1/M5 data the number of sessions is small relative to bar count.
    cum_tp_vol = tp_vol.to_list()
    cum_vol = volume.to_list()
    starts = is_session_start.to_list()

    vwap_values: list[float] = []
    running_tp_vol = 0.0
    running_vol = 0.0

    for tpv, vol, is_start in zip(cum_tp_vol, cum_vol, starts, strict=True):
        if is_start:
            running_tp_vol = 0.0
            running_vol = 0.0
        running_tp_vol += tpv
        running_vol += vol
        # Avoid division by zero on bars with zero volume.
        vwap_values.append(running_tp_vol / running_vol if running_vol > 0 else float("nan"))

    return pl.Series("vwap", vwap_values, dtype=pl.Float64)
