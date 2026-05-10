"""
Trend-following indicators.

Functions
---------
donchian_channels   Highest high / lowest low channel over a rolling window
adx                 Average Directional Index with +DI and -DI components
supertrend          ATR-based trailing stop/trend indicator
parabolic_sar       Parabolic SAR — acceleration-factor dot plot
ichimoku            Ichimoku Cloud — five-component trend/support/resistance system
"""

from __future__ import annotations

import polars as pl

from polarticks._validate import _validate_period
from polarticks.moving_averages import wilder_smooth
from polarticks.volatility import atr


def donchian_channels(ohlc: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """Donchian Channels: rolling highest high and lowest low.

    Originally used by Richard Donchian for trend-following breakout systems.
    A close above the upper channel signals bullish breakout; a close below
    the lower channel signals bearish breakout.  The middle band is the
    midpoint of upper and lower.

    Args:
        ohlc: DataFrame with columns ``high`` and ``low``.
        period: Lookback window (default 20).

    Returns:
        DataFrame with columns ``dc_upper_{period}``, ``dc_lower_{period}``,
        ``dc_middle_{period}``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Donchian Channels")

    upper = ohlc["high"].rolling_max(window_size=period, min_samples=period)
    lower = ohlc["low"].rolling_min(window_size=period, min_samples=period)
    middle = (upper + lower) / 2.0

    return pl.DataFrame(
        {
            f"dc_upper_{period}": upper,
            f"dc_lower_{period}": lower,
            f"dc_middle_{period}": middle,
        }
    )


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


def adx(ohlc: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Average Directional Index (ADX) with +DI and -DI components.

    ADX quantifies the *strength* of a trend irrespective of direction.
    Readings above 25 indicate a trending market; below 20 suggest
    consolidation.  +DI > -DI indicates bullish momentum; -DI > +DI
    indicates bearish momentum.

    Algorithm (Wilder, 1978):
        1. Compute +DM and -DM directional movement for each bar.
        2. Apply Wilder smoothing to +DM, -DM, and True Range.
        3. +DI = 100 × Wilder(+DM) / Wilder(TR).
        4. -DI = 100 × Wilder(-DM) / Wilder(TR).
        5. DX  = 100 × |+DI − -DI| / (+DI + -DI).
        6. ADX = Wilder(DX, period).

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: Wilder smoothing period (default 14).

    Returns:
        DataFrame with columns ``adx_{period}``, ``plus_di_{period}``,
        ``minus_di_{period}``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "ADX")

    high = ohlc["high"]
    low = ohlc["low"]

    # Raw directional movement: only the larger up or down move counts per bar.
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # pl.when/then produces an Expr; pl.select() materialises it to a Series.
    plus_dm = (
        pl.select(pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0))
        .to_series()
        .fill_null(0.0)
    )
    minus_dm = (
        pl.select(pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0))
        .to_series()
        .fill_null(0.0)
    )

    # Wilder-smooth DM components and ATR in one pass each.
    atr_values = atr(ohlc, period)
    smoothed_plus = wilder_smooth(plus_dm, period)
    smoothed_minus = wilder_smooth(minus_dm, period)

    plus_di = (100.0 * smoothed_plus / atr_values).fill_nan(0.0)
    minus_di = (100.0 * smoothed_minus / atr_values).fill_nan(0.0)

    dx_num = (plus_di - minus_di).abs()
    dx_den = plus_di + minus_di
    # fill_nan covers the rare case where both DI values are zero.
    dx = (100.0 * dx_num / dx_den).fill_nan(0.0)

    adx_values = wilder_smooth(dx, period)

    return pl.DataFrame(
        {
            f"adx_{period}": adx_values,
            f"plus_di_{period}": plus_di,
            f"minus_di_{period}": minus_di,
        }
    )


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------


def supertrend(
    ohlc: pl.DataFrame,
    period: int = 7,
    multiplier: float = 3.0,
) -> pl.DataFrame:
    """Supertrend — ATR-based trailing stop and trend direction indicator.

    The Supertrend line acts as a dynamic support (uptrend) or resistance
    (downtrend).  When price crosses the band it signals a trend flip.

    Algorithm:
        basic_upper = (high + low) / 2 + multiplier × ATR
        basic_lower = (high + low) / 2 − multiplier × ATR

        final_upper[t]:
            If basic_upper[t] < final_upper[t-1] OR close[t-1] > final_upper[t-1]:
                final_upper[t] = basic_upper[t]
            Else:
                final_upper[t] = final_upper[t-1]

        final_lower[t]:
            If basic_lower[t] > final_lower[t-1] OR close[t-1] < final_lower[t-1]:
                final_lower[t] = basic_lower[t]
            Else:
                final_lower[t] = final_lower[t-1]

        direction: +1 when close > final_upper (bullish), -1 otherwise.

    The first ``period`` bars are ``null`` (ATR warm-up).

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: ATR lookback period (default 7).
        multiplier: ATR multiplier for band width (default 3.0).

    Returns:
        DataFrame with columns ``supertrend`` (band level) and
        ``supertrend_direction`` (+1 / -1).

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Supertrend")

    high = ohlc["high"].to_list()
    low = ohlc["low"].to_list()
    close = ohlc["close"].to_list()
    atr_vals = atr(ohlc, period).to_list()

    n = len(close)
    band: list[float | None] = [None] * n
    direction: list[int | None] = [None] * n

    # Track the running final bands across bars (sequential by nature).
    final_upper = 0.0
    final_lower = 0.0

    for idx in range(n):
        if atr_vals[idx] is None:
            # Still in the ATR warm-up period.
            continue

        hl2 = (high[idx] + low[idx]) / 2.0
        basic_upper = hl2 + multiplier * atr_vals[idx]
        basic_lower = hl2 - multiplier * atr_vals[idx]

        if idx == period - 1:
            # First bar with a valid ATR — initialise both bands.
            final_upper = basic_upper
            final_lower = basic_lower
        else:
            prev_close = close[idx - 1]
            # Upper band ratchets down (tightens) unless price broke above it.
            final_upper = (
                basic_upper
                if (basic_upper < final_upper or prev_close > final_upper)
                else final_upper
            )
            # Lower band ratchets up (tightens) unless price broke below it.
            final_lower = (
                basic_lower
                if (basic_lower > final_lower or prev_close < final_lower)
                else final_lower
            )

        # Trend direction: bullish when close is above the upper band.
        if close[idx] > final_upper:
            direction[idx] = 1
            band[idx] = final_lower
        else:
            direction[idx] = -1
            band[idx] = final_upper

    return pl.DataFrame(
        {
            "supertrend": pl.Series("supertrend", band, dtype=pl.Float64),
            "supertrend_direction": pl.Series("supertrend_direction", direction, dtype=pl.Int8),
        }
    )


# ---------------------------------------------------------------------------
# Parabolic SAR
# ---------------------------------------------------------------------------


def parabolic_sar(
    ohlc: pl.DataFrame,
    initial_af: float = 0.02,
    step_af: float = 0.02,
    max_af: float = 0.20,
) -> pl.DataFrame:
    """Parabolic SAR — acceleration-factor trailing stop.

    The SAR (Stop And Reverse) dot trails price and accelerates toward it
    as new extremes are set.  It is primarily used to determine trend
    direction and place trailing stops.

    Algorithm (Wilder, 1978):
        In an uptrend:
            SAR[t] = SAR[t-1] + AF × (EP − SAR[t-1])
            If close[t] < SAR[t]: flip to downtrend, reset AF = initial_af.
            If new high > EP: EP = new high; AF = min(AF + step_af, max_af).

        In a downtrend the same logic applies with highs and lows swapped.

    The indicator initialises on bar 1 (two bars needed to determine the
    first SAR).  Bar 0 is always ``null``.

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        initial_af: Starting acceleration factor (default 0.02).
        step_af: AF increment each time a new extreme is set (default 0.02).
        max_af: Maximum allowed acceleration factor (default 0.20).

    Returns:
        DataFrame with columns ``psar`` (SAR level) and
        ``psar_direction`` (+1 uptrend / -1 downtrend).
    """
    high = ohlc["high"].to_list()
    low = ohlc["low"].to_list()
    close = ohlc["close"].to_list()
    n = len(close)

    sar_out: list[float | None] = [None] * n
    dir_out: list[int | None] = [None] * n

    if n < 2:
        return pl.DataFrame(
            {
                "psar": pl.Series("psar", sar_out, dtype=pl.Float64),
                "psar_direction": pl.Series("psar_direction", dir_out, dtype=pl.Int8),
            }
        )

    # Seed: guess uptrend if bar 1 closes higher than bar 0.
    is_uptrend = close[1] >= close[0]
    af = initial_af
    ep = high[0] if is_uptrend else low[0]
    sar = low[0] if is_uptrend else high[0]

    for idx in range(1, n):
        # Project the SAR one step using the acceleration formula.
        sar = sar + af * (ep - sar)

        # Clamp SAR to the last two prior bars' extremes (Wilder's rule).
        if is_uptrend:
            sar = min(sar, low[idx - 1], low[max(0, idx - 2)])
        else:
            sar = max(sar, high[idx - 1], high[max(0, idx - 2)])

        # Check for trend reversal.
        if is_uptrend and close[idx] < sar:
            is_uptrend = False
            sar = ep  # Flip SAR to the prior extreme point.
            ep = low[idx]
            af = initial_af
        elif not is_uptrend and close[idx] > sar:
            is_uptrend = True
            sar = ep
            ep = high[idx]
            af = initial_af
        else:
            # No reversal — update EP and AF if a new extreme was set.
            if is_uptrend and high[idx] > ep:
                ep = high[idx]
                af = min(af + step_af, max_af)
            elif not is_uptrend and low[idx] < ep:
                ep = low[idx]
                af = min(af + step_af, max_af)

        sar_out[idx] = sar
        dir_out[idx] = 1 if is_uptrend else -1

    return pl.DataFrame(
        {
            "psar": pl.Series("psar", sar_out, dtype=pl.Float64),
            "psar_direction": pl.Series("psar_direction", dir_out, dtype=pl.Int8),
        }
    )


# ---------------------------------------------------------------------------
# Ichimoku Cloud
# ---------------------------------------------------------------------------


def ichimoku(
    ohlc: pl.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    chikou_period: int = 26,
) -> pl.DataFrame:
    """Ichimoku Cloud — five-component trend, momentum, and support/resistance system.

    Developed by Goichi Hosoda, Ichimoku provides at-a-glance information about
    trend direction, momentum, support, and resistance.

    Components (all returned at the current bar index; no forward/backward shift
    is applied — callers can shift as needed for visual display):

        tenkan_sen    Conversion Line: midpoint of the highest high and lowest low
                      over ``tenkan_period`` bars.  Faster trend line.

        kijun_sen     Base Line: same midpoint formula over ``kijun_period`` bars.
                      Slower trend line; crossovers with Tenkan signal trend changes.

        senkou_span_a Leading Span A: (tenkan_sen + kijun_sen) / 2.  Conventionally
                      plotted ``kijun_period`` bars ahead; forms the faster cloud edge.

        senkou_span_b Leading Span B: midpoint of the highest high / lowest low over
                      ``senkou_b_period`` bars.  Conventionally plotted ``kijun_period``
                      bars ahead; forms the slower cloud edge.

        chikou_span   Lagging Span: current close value stored at the current bar.
                      Conventionally plotted ``chikou_period`` bars *behind* the
                      current bar.  Returned here as ``close.shift(-chikou_period)``
                      so that bar ``t`` holds the close from bar ``t + chikou_period``,
                      enabling comparison of today's close to historical price.

    Null-prefix summary (default periods):
        tenkan_sen:    ``tenkan_period - 1``   (8 bars)
        kijun_sen:     ``kijun_period - 1``    (25 bars)
        senkou_span_a: ``kijun_period - 1``    (25 bars, limited by kijun_sen)
        senkou_span_b: ``senkou_b_period - 1`` (51 bars)
        chikou_span:   0 leading nulls; ``chikou_period`` trailing nulls.

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        tenkan_period: Conversion Line lookback period (default 9).
        kijun_period: Base Line lookback period (default 26).
        senkou_b_period: Leading Span B lookback period (default 52).
        chikou_period: Number of bars the Lagging Span is offset (default 26).

    Returns:
        DataFrame with columns ``tenkan_sen``, ``kijun_sen``, ``senkou_span_a``,
        ``senkou_span_b``, ``chikou_span``.

    Raises:
        ValueError: If any period is less than 1.
    """
    _validate_period(tenkan_period, "Ichimoku tenkan_period")
    _validate_period(kijun_period, "Ichimoku kijun_period")
    _validate_period(senkou_b_period, "Ichimoku senkou_b_period")
    _validate_period(chikou_period, "Ichimoku chikou_period")

    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    # Tenkan-sen: midpoint of the tenkan_period range.
    tenkan_high = high.rolling_max(window_size=tenkan_period, min_samples=tenkan_period)
    tenkan_low = low.rolling_min(window_size=tenkan_period, min_samples=tenkan_period)
    tenkan_sen = ((tenkan_high + tenkan_low) / 2.0).alias("tenkan_sen")

    # Kijun-sen: midpoint of the kijun_period range.
    kijun_high = high.rolling_max(window_size=kijun_period, min_samples=kijun_period)
    kijun_low = low.rolling_min(window_size=kijun_period, min_samples=kijun_period)
    kijun_sen = ((kijun_high + kijun_low) / 2.0).alias("kijun_sen")

    # Senkou Span A: average of Tenkan and Kijun; inherits the wider null prefix.
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).alias("senkou_span_a")

    # Senkou Span B: midpoint of the longest range window.
    senkou_b_high = high.rolling_max(window_size=senkou_b_period, min_samples=senkou_b_period)
    senkou_b_low = low.rolling_min(window_size=senkou_b_period, min_samples=senkou_b_period)
    senkou_span_b = ((senkou_b_high + senkou_b_low) / 2.0).alias("senkou_span_b")

    # Chikou Span: close shifted forward so bar t holds the close from t+chikou_period.
    # This gives 0 leading nulls and chikou_period trailing nulls.
    chikou_span = close.shift(-chikou_period).alias("chikou_span")

    return pl.DataFrame(
        {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span,
        }
    )
