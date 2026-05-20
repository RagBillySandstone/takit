"""
Trend-following indicators.

Functions
---------
donchian_channels           Highest high / lowest low channel over a rolling window
adx                         Average Directional Index with +DI and -DI components
supertrend                  ATR-based trailing stop/trend indicator
parabolic_sar               Parabolic SAR — acceleration-factor dot plot
ichimoku                    Ichimoku Cloud — five-component trend/support/resistance system
aroon                       Aroon Up/Down/Oscillator — time-since-extreme trend indicator
vortex                      Vortex Indicator — VI+ and VI− directional-movement lines
linreg_slope                Rolling linear regression slope coefficient
stc                         Schaff Trend Cycle — stochastic of MACD for cycle detection
elder_ray                   Elder Ray Index — Bull Power and Bear Power vs EMA
alligator                   Bill Williams Alligator — three offset Wilder-smoothed lines
fractal                     Williams Fractal — 5-bar pivot high and pivot low detector
linreg_channel              Rolling linear regression channel with RMSE-based bands
tsf                         Time Series Forecast — linreg projected one bar ahead
chande_kroll_stop           Chande Kroll Stop — two-stage ATR trailing stop
vhf                         Vertical Horizontal Filter — trending vs. ranging regime detector
pfe                         Polarized Fractal Efficiency — EMA-smoothed path-efficiency × direction
chande_forecast_oscillator  Chande Forecast Oscillator — % deviation of close from TSF
linreg_r2                   Rolling linear regression R² (coefficient of determination)
tii                         Trend Intensity Index — fraction of closes above/below the SMA
ma_envelope                 Moving Average Envelope — MA ± percentage bands
linreg_intercept            Rolling OLS y-intercept (constant term at x = 0)
standard_error_bands        Linear regression line ± 2 × residual standard error
cog                         Centre of Gravity oscillator — Ehlers' price-weighted lag estimator
rwi                         Random Walk Index — tests whether price moves exceed a random walk
"""

from __future__ import annotations

import operator
from functools import reduce

import polars as pl

from polarticks._validate import _validate_period
from polarticks.moving_averages import ema, wilder_smooth
from polarticks.volatility import atr, true_range


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


# ---------------------------------------------------------------------------
# Aroon
# ---------------------------------------------------------------------------


def aroon(ohlc: pl.DataFrame, period: int = 25) -> pl.DataFrame:
    """Aroon Indicator — measures time since the last extreme high or low.

    Aroon quantifies how recently within a lookback window the highest high
    and lowest low occurred.  Values of 100 on Aroon Up mean a new high was
    made on the current bar; 0 means the high was made *period* bars ago.
    Aroon Down works the same way for the lowest low.

    Algorithm:
        aroon_up[t]   = 100 × index_of_highest_high_in_window / period
        aroon_down[t] = 100 × index_of_lowest_low_in_window  / period
        aroon_osc[t]  = aroon_up[t] − aroon_down[t]

    The window contains ``period + 1`` bars so that the current bar occupies
    index *period* (newest) and the oldest bar occupies index 0.  When the
    arg_max/min is at index *period* (current bar), Aroon Up/Down = 100.

    The first *period* output values are ``null``.

    Args:
        ohlc: DataFrame with columns ``high`` and ``low``.
        period: Number of bars to look back (default 25).

    Returns:
        DataFrame with columns ``aroon_up_{period}``, ``aroon_down_{period}``,
        ``aroon_osc_{period}``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Aroon")

    high = ohlc["high"]
    low = ohlc["low"]

    # rolling_map invokes a Python callback per window of size period + 1.
    # index 0 = oldest bar, index period = current bar → arg_max() / period = Aroon Up.
    aroon_up = high.rolling_map(
        function=lambda window: 100.0 * (window.arg_max() or 0) / (len(window) - 1),
        window_size=period + 1,
        min_samples=period + 1,
    ).alias(f"aroon_up_{period}")

    aroon_down = low.rolling_map(
        function=lambda window: 100.0 * (window.arg_min() or 0) / (len(window) - 1),
        window_size=period + 1,
        min_samples=period + 1,
    ).alias(f"aroon_down_{period}")

    aroon_osc = (aroon_up - aroon_down).alias(f"aroon_osc_{period}")

    return pl.DataFrame(
        {
            f"aroon_up_{period}": aroon_up,
            f"aroon_down_{period}": aroon_down,
            f"aroon_osc_{period}": aroon_osc,
        }
    )


# ---------------------------------------------------------------------------
# Vortex Indicator
# ---------------------------------------------------------------------------


def vortex(ohlc: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Vortex Indicator — directional-movement lines VI+ and VI−.

    The Vortex Indicator (Etienne Botes & Douglas Siepman, 2010) compares
    upward and downward price movements to True Range, producing two
    oscillating lines that signal trend direction and strength.

    Algorithm:
        vm_plus[t]    = |high[t] − low[t-1]|   (positive vortex movement)
        vm_minus[t]   = |low[t]  − high[t-1]|  (negative vortex movement)
        (bar 0 values for both are set to 0 — no prior bar)
        VI+[t] = Σ(vm_plus,  period) / Σ(TrueRange, period)
        VI−[t] = Σ(vm_minus, period) / Σ(TrueRange, period)

    When VI+ > VI− the market is in an uptrend; when VI− > VI+ it is in a
    downtrend.  A cross of the two lines signals a trend change.

    The first ``period − 1`` output values are ``null``.

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: Rolling sum period (default 14).

    Returns:
        DataFrame with columns ``vi_plus_{period}`` and ``vi_minus_{period}``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Vortex")

    high = ohlc["high"]
    low = ohlc["low"]

    # |high[t] − low[t-1]|: bar 0 has no prior bar → fill to 0.
    vm_plus = (high - low.shift(1)).abs().fill_null(0.0)
    # |low[t] − high[t-1]|: same boundary treatment.
    vm_minus = (low - high.shift(1)).abs().fill_null(0.0)

    tr_vals = true_range(ohlc)

    vm_plus_sum = vm_plus.rolling_sum(window_size=period, min_samples=period)
    vm_minus_sum = vm_minus.rolling_sum(window_size=period, min_samples=period)
    tr_sum = tr_vals.rolling_sum(window_size=period, min_samples=period)

    vi_plus = (vm_plus_sum / tr_sum).alias(f"vi_plus_{period}")
    vi_minus = (vm_minus_sum / tr_sum).alias(f"vi_minus_{period}")

    return pl.DataFrame(
        {
            f"vi_plus_{period}": vi_plus,
            f"vi_minus_{period}": vi_minus,
        }
    )


# ---------------------------------------------------------------------------
# Linear Regression Slope
# ---------------------------------------------------------------------------


def linreg_slope(series: pl.Series, period: int = 14) -> pl.Series:
    """Rolling ordinary-least-squares regression slope.

    Fits a straight line to the last *period* bars (using bar position 0…n−1
    as the x-axis) and returns the slope coefficient.  A positive slope
    indicates an uptrend; a negative slope indicates a downtrend.  The
    magnitude reflects the rate of price change per bar.

    Algorithm (for window positions k = 0, 1, … period − 1):
        sum_x   = n(n−1)/2                (constant for any window)
        sum_x²  = n(n−1)(2n−1)/6         (constant)
        sum_xy  = Σ k × price[t−n+1+k]   (weighted shift sum)
        sum_y   = rolling_sum(price, n)
        slope   = (n·sum_xy − sum_x·sum_y) / (n·sum_x² − sum_x²)

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Input price series (e.g., close).
        period: Lookback window; must be at least 2 (default 14).

    Returns:
        Series named ``linreg_slope_{period}``.
        The first ``period − 1`` values are ``null``.

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "Linear Regression Slope", min_period=2)

    n = period
    # x-axis statistics are constant for a fixed-width window.
    sum_x = n * (n - 1) / 2.0
    sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0
    denom = float(n) * sum_x2 - sum_x**2

    # Weighted sum: Σ(k × price[t−n+1+k]) for k = 1…n−1 (k=0 contributes zero).
    # shift(n−1−k) aligns the bar at window-position k to the current index.
    sum_xy: pl.Series = reduce(
        operator.add,
        (series.shift(n - 1 - k) * float(k) for k in range(1, n)),
    )
    sum_y = series.rolling_sum(window_size=n, min_samples=n)

    return ((float(n) * sum_xy - sum_x * sum_y) / denom).alias(f"linreg_slope_{period}")


# ---------------------------------------------------------------------------
# Schaff Trend Cycle (STC)
# ---------------------------------------------------------------------------


def stc(
    ohlc: pl.DataFrame,
    fast: int = 23,
    slow: int = 50,
    stoch_period: int = 10,
    smooth: int = 3,
) -> pl.Series:
    """Schaff Trend Cycle (STC) — stochastic of MACD for faster cycle detection.

    STC (Doug Schaff, 1999) applies the stochastic formula twice to a MACD
    line, smoothing each intermediate %K with an EMA.  The result oscillates
    between 0 and 100 and generates buy/sell signals at the 25/75 thresholds
    — typically earlier than a raw MACD cross.

    Algorithm:
        macd_line = EMA(close, fast) − EMA(close, slow)
        %K1       = fast-stoch(macd_line, stoch_period)  (clipped at 0, fill_nan=50)
        %D1       = EMA(%K1, smooth)
        %K2       = fast-stoch(%D1, stoch_period)        (clipped at 0, fill_nan=50)
        STC       = EMA(%K2, smooth).clip(0, 100)

    A reading above 75 suggests overbought; below 25 suggests oversold.

    Null-prefix: ``slow + 2×stoch_period + 2×smooth − 5`` bars.

    Args:
        ohlc: DataFrame with column ``close``.
        fast: Fast EMA period (default 23).
        slow: Slow EMA period (default 50).
        stoch_period: Rolling window for each stochastic pass (default 10).
        smooth: EMA smoothing period for each intermediate %K (default 3).

    Returns:
        Series named ``stc``, values clamped to [0, 100].

    Raises:
        ValueError: If ``fast >= slow`` or any period < 1.
    """
    _validate_period(fast, "STC fast")
    _validate_period(slow, "STC slow")
    _validate_period(stoch_period, "STC stoch_period")
    _validate_period(smooth, "STC smooth")
    if fast >= slow:
        raise ValueError(f"STC fast ({fast}) must be less than slow ({slow}).")

    close = ohlc["close"]

    # Step 1: MACD line as the input to both stochastic passes.
    macd_line = ema(close, fast) - ema(close, slow)

    # Step 2: First stochastic pass over the MACD line.
    min1 = macd_line.rolling_min(window_size=stoch_period, min_samples=stoch_period)
    max1 = macd_line.rolling_max(window_size=stoch_period, min_samples=stoch_period)
    # fill_nan converts division-by-zero (flat MACD region) to mid-range (50).
    k1 = (100.0 * (macd_line - min1) / (max1 - min1)).fill_nan(50.0)
    d1 = ema(k1, smooth)

    # Step 3: Second stochastic pass over the smoothed %K.
    min2 = d1.rolling_min(window_size=stoch_period, min_samples=stoch_period)
    max2 = d1.rolling_max(window_size=stoch_period, min_samples=stoch_period)
    k2 = (100.0 * (d1 - min2) / (max2 - min2)).fill_nan(50.0)
    stc_raw = ema(k2, smooth)

    # Clamp to [0, 100] to prevent floating-point overshoot at the boundaries.
    return stc_raw.clip(lower_bound=0.0, upper_bound=100.0).alias("stc")


# ---------------------------------------------------------------------------
# Elder Ray Index
# ---------------------------------------------------------------------------


def elder_ray(ohlc: pl.DataFrame, period: int = 13) -> pl.DataFrame:
    """Elder Ray Index — Bull Power and Bear Power relative to an EMA.

    Developed by Dr Alexander Elder, the indicator splits market force into
    two components: the bulls' ability to push price above the consensus EMA
    (Bull Power) and the bears' ability to push price below it (Bear Power).

    Algorithm:
        ema_close   = EMA(close, period)
        bull_power  = high  − ema_close
        bear_power  = low   − ema_close

    Interpretation:
        - Bull Power > 0 and rising → bulls are strengthening.
        - Bear Power < 0 and rising (becoming less negative) → bears weakening.
        - Divergence between price and either power line signals reversals.

    Null-prefix: ``period − 1`` bars (inherited from the EMA).

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: EMA lookback period (default 13).

    Returns:
        DataFrame with columns ``bull_power`` and ``bear_power``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Elder Ray")

    ema_close = ema(ohlc["close"], period)

    # Distance of the bar's extreme from the consensus EMA level.
    bull_power = (ohlc["high"] - ema_close).alias("bull_power")
    bear_power = (ohlc["low"] - ema_close).alias("bear_power")

    return pl.DataFrame({"bull_power": bull_power, "bear_power": bear_power})


# ---------------------------------------------------------------------------
# Alligator
# ---------------------------------------------------------------------------


def alligator(
    ohlc: pl.DataFrame,
    jaw_period: int = 13,
    jaw_offset: int = 8,
    teeth_period: int = 8,
    teeth_offset: int = 5,
    lips_period: int = 5,
    lips_offset: int = 3,
) -> pl.DataFrame:
    """Bill Williams Alligator — three offset Wilder-smoothed median-price lines.

    The Alligator represents a sleeping, awakening, or eating market via three
    Wilder-smoothed lines of the median price ``(high + low) / 2``, each
    displaced backward by an offset (simulating the TradingView future-shift
    convention on historical data):

        jaw   = Wilder(median, jaw_period).shift(jaw_offset)
        teeth = Wilder(median, teeth_period).shift(teeth_offset)
        lips  = Wilder(median, lips_period).shift(lips_offset)

    When the lines are intertwined the market is "sleeping".  When lips > teeth
    > jaw the market is bullish; jaw > teeth > lips is bearish.

    Null-prefix: ``jaw_period + jaw_offset − 2`` bars (slowest line dominates).

    Args:
        ohlc: DataFrame with columns ``high`` and ``low``.
        jaw_period: Wilder period for the jaw / blue line (default 13).
        jaw_offset: Bars to shift the jaw backward (default 8).
        teeth_period: Wilder period for the teeth / red line (default 8).
        teeth_offset: Bars to shift the teeth backward (default 5).
        lips_period: Wilder period for the lips / green line (default 5).
        lips_offset: Bars to shift the lips backward (default 3).

    Returns:
        DataFrame with columns ``jaw``, ``teeth``, ``lips``.

    Raises:
        ValueError: If any period or offset < 1.
    """
    for val, name in [
        (jaw_period, "jaw_period"),
        (teeth_period, "teeth_period"),
        (lips_period, "lips_period"),
    ]:
        _validate_period(val, f"Alligator {name}")

    # Median price used by Bill Williams as the Alligator input.
    median = (ohlc["high"] + ohlc["low"]) / 2.0

    # shift(n) displaces the smoothed line backward n bars in the historical axis.
    jaw = wilder_smooth(median, jaw_period).shift(jaw_offset).alias("jaw")
    teeth = wilder_smooth(median, teeth_period).shift(teeth_offset).alias("teeth")
    lips = wilder_smooth(median, lips_period).shift(lips_offset).alias("lips")

    return pl.DataFrame({"jaw": jaw, "teeth": teeth, "lips": lips})


# ---------------------------------------------------------------------------
# Fractal
# ---------------------------------------------------------------------------


def fractal(ohlc: pl.DataFrame) -> pl.DataFrame:
    """Williams Fractal — 5-bar pivot high and pivot low pattern detector.

    A bearish fractal occurs when a bar's high is strictly higher than the
    two bars immediately before and after it.  A bullish fractal occurs when
    a bar's low is strictly lower.  The first two and last two bars can never
    host a fractal (they lack the required neighbours).

    Algorithm:
        bearish[t] = high[t] > high[t−1] AND high[t] > high[t−2]
                     AND high[t] > high[t+1] AND high[t] > high[t+2]
        bullish[t] = low[t]  < low[t−1]  AND low[t]  < low[t−2]
                     AND low[t]  < low[t+1]  AND low[t]  < low[t+2]

    Args:
        ohlc: DataFrame with columns ``high`` and ``low``.

    Returns:
        DataFrame with boolean columns ``fractal_bearish`` and ``fractal_bullish``.
        The first 2 and last 2 bars are always ``False``.
    """
    high = ohlc["high"]
    low = ohlc["low"]

    # A bearish fractal: current high is strictly greater than all four neighbours.
    bearish = (
        (
            (high > high.shift(2))
            & (high > high.shift(1))
            & (high > high.shift(-1))
            & (high > high.shift(-2))
        )
        .fill_null(False)
        .alias("fractal_bearish")
    )

    # A bullish fractal: current low is strictly lower than all four neighbours.
    bullish = (
        (
            (low < low.shift(2))
            & (low < low.shift(1))
            & (low < low.shift(-1))
            & (low < low.shift(-2))
        )
        .fill_null(False)
        .alias("fractal_bullish")
    )

    return pl.DataFrame({"fractal_bearish": bearish, "fractal_bullish": bullish})


# ---------------------------------------------------------------------------
# Linear Regression Channel
# ---------------------------------------------------------------------------


def linreg_channel(
    series: pl.Series,
    period: int = 100,
    num_std: float = 2.0,
) -> pl.DataFrame:
    """Rolling linear regression channel — fitted line with RMSE-based bands.

    At each bar, fits an OLS line to the last *period* bars and returns:
        - the fitted (LinReg) value at the end of the window (``lrc_mid``),
        - upper and lower bands at ``num_std`` × RMSE above/below the line.

    Algorithm (vectorised):
        slope = linreg_slope(series, period)
        mid   = rolling_mean + slope × (period − 1) / 2
        SSE   = rolling_sample_var × (period − 1) − slope² × Σ(x − mean_x)²
        RMSE  = sqrt(SSE / period)
        upper = mid + num_std × RMSE
        lower = mid − num_std × RMSE

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Input price series (e.g., close).
        period: Lookback window for each regression fit (default 100).
        num_std: Number of RMSE widths for the upper/lower bands (default 2.0).

    Returns:
        DataFrame with columns ``lrc_mid``, ``lrc_upper``, ``lrc_lower``.

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "Linear Regression Channel", min_period=2)

    n = period
    slope = linreg_slope(series, n)
    mean_y = series.rolling_mean(window_size=n, min_samples=n)

    # LinReg value at x = n-1: mean_y + slope × (n-1)/2.
    mid = (mean_y + slope * (n - 1) / 2.0).alias("lrc_mid")

    # RMSE via the algebraic decomposition of residual sum of squares.
    ss_xx = n * (n**2 - 1) / 12.0  # Σ(x_i - mean_x)² for x = 0..n-1
    sample_var = series.rolling_std(window_size=n, min_samples=n) ** 2
    # SSE = TSS - SSR = sample_var*(n-1) - slope²*ss_xx; clip to avoid sqrt of negative.
    sse = (sample_var * (n - 1) - slope**2 * ss_xx).clip(lower_bound=0.0)
    rmse = (sse / n) ** 0.5

    upper = (mid + num_std * rmse).alias("lrc_upper")
    lower = (mid - num_std * rmse).alias("lrc_lower")

    return pl.DataFrame({"lrc_mid": mid, "lrc_upper": upper, "lrc_lower": lower})


# ---------------------------------------------------------------------------
# Time Series Forecast
# ---------------------------------------------------------------------------


def tsf(series: pl.Series, period: int = 14) -> pl.Series:
    """Time Series Forecast — rolling OLS line projected one bar ahead.

    TSF extends the rolling linear regression by one step beyond the window,
    providing a one-bar-ahead projection.  It is equivalent to the LinReg
    value at the end of the window plus one slope increment:

        LinReg_value[t] = rolling_mean + slope × (period − 1) / 2
        TSF[t]          = LinReg_value[t] + slope[t]

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Input price series (e.g., close).
        period: Lookback window (default 14; must be ≥ 2).

    Returns:
        Series named ``tsf_{period}``.

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "TSF", min_period=2)

    n = period
    slope = linreg_slope(series, n)
    mean_y = series.rolling_mean(window_size=n, min_samples=n)

    # LinReg value at end of window, then projected one step further.
    lrv = mean_y + slope * (n - 1) / 2.0
    return (lrv + slope).alias(f"tsf_{period}")


# ---------------------------------------------------------------------------
# Chande Kroll Stop
# ---------------------------------------------------------------------------


def chande_kroll_stop(
    ohlc: pl.DataFrame,
    atr_period: int = 10,
    atr_mult: float = 1.5,
    stop_period: int = 9,
) -> pl.DataFrame:
    """Chande Kroll Stop — two-stage ATR-based adaptive trailing stop.

    The Chande Kroll Stop (Chande & Kroll, 1994) combines an ATR-displaced
    first-stage stop with a second rolling extreme to produce smooth long and
    short stop-loss levels.  A close above ``cks_long`` is bullish; a close
    below ``cks_short`` is bearish.

    Algorithm:
        first_high = rolling_max(high, atr_period) − atr_mult × ATR(atr_period)
        first_low  = rolling_min(low,  atr_period) + atr_mult × ATR(atr_period)
        cks_long   = rolling_max(first_low,  stop_period)
        cks_short  = rolling_min(first_high, stop_period)

    Null-prefix: ``atr_period + stop_period − 2`` bars.

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        atr_period: ATR and first-stage rolling window period (default 10).
        atr_mult: ATR multiplier for the first-stage stop (default 1.5).
        stop_period: Second-stage rolling window for the final stop (default 9).

    Returns:
        DataFrame with columns ``cks_long`` and ``cks_short``.

    Raises:
        ValueError: If ``atr_period < 1`` or ``stop_period < 1``.
    """
    _validate_period(atr_period, "Chande Kroll Stop atr_period")
    _validate_period(stop_period, "Chande Kroll Stop stop_period")

    atr_vals = atr(ohlc, atr_period)

    # First-stage: displace rolling extremes by the ATR cushion.
    highest_high = ohlc["high"].rolling_max(window_size=atr_period, min_samples=atr_period)
    lowest_low = ohlc["low"].rolling_min(window_size=atr_period, min_samples=atr_period)
    first_high_stop = highest_high - atr_mult * atr_vals
    first_low_stop = lowest_low + atr_mult * atr_vals

    # Second-stage: smooth the first-stage stops with a rolling extreme.
    cks_long = first_low_stop.rolling_max(window_size=stop_period, min_samples=stop_period).alias(
        "cks_long"
    )
    cks_short = first_high_stop.rolling_min(window_size=stop_period, min_samples=stop_period).alias(
        "cks_short"
    )

    return pl.DataFrame({"cks_long": cks_long, "cks_short": cks_short})


# ---------------------------------------------------------------------------
# Vertical Horizontal Filter (VHF)
# ---------------------------------------------------------------------------


def vhf(series: pl.Series, period: int = 28) -> pl.Series:
    """Vertical Horizontal Filter — trending vs. ranging regime quantifier.

    VHF divides the total price range (highest − lowest) over the lookback
    period by the sum of absolute bar-to-bar price changes over the same
    window.  High VHF values (typically > 0.4) indicate a trending market;
    low values (< 0.2) indicate a ranging/choppy market.  The oscillator
    oscillates between 0 and 1 (assuming a monotone run corresponds to 1).

    Algorithm:
        highest[t]  = rolling_max(close, period)
        lowest[t]   = rolling_min(close, period)
        path_sum[t] = rolling_sum(|close[t] − close[t-1]|, period)
        VHF[t]      = (highest[t] − lowest[t]) / path_sum[t]

    Null-prefix: ``period`` bars (the diff adds one extra null at bar 0).

    Args:
        series: Close price series.
        period: Lookback window (default 28).

    Returns:
        Series of VHF values named ``vhf_{period}``.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - Schwager, J. D. *Schwager on Futures: Technical Analysis* (1996),
          Chapter on the VHF.
        - Investopedia — Vertical Horizontal Filter:
          https://www.investopedia.com/terms/v/vhf.asp
    """
    _validate_period(period, "VHF")
    highest = series.rolling_max(window_size=period, min_samples=period)
    lowest = series.rolling_min(window_size=period, min_samples=period)
    # diff[0] is null → abs_diff[0] is null → rolling_sum requires period
    # non-null values → path_sum leading nulls = period.
    abs_diff = series.diff(1).abs()
    path_sum = abs_diff.rolling_sum(window_size=period, min_samples=period)
    # Avoid division by zero when all prices are identical in the window.
    safe_path = path_sum.replace(0.0, float("nan"))
    return ((highest - lowest) / safe_path).alias(f"vhf_{period}")


# ---------------------------------------------------------------------------
# Polarized Fractal Efficiency (PFE)
# ---------------------------------------------------------------------------


def pfe(series: pl.Series, period: int = 14, smooth: int = 5) -> pl.Series:
    """Polarized Fractal Efficiency — directional path-efficiency oscillator.

    PFE measures how efficiently price has moved from its position *period*
    bars ago to the current bar.  A straight-line Euclidean path (numerator)
    is compared to the actual price path (denominator); the ratio is then
    multiplied by the direction of the move (+1 or −1) and by 100.  The raw
    signal is smoothed with an EMA.

    Values near ±100 indicate highly efficient trending movement; values near
    0 indicate choppy, random-walk-like behaviour.

    Algorithm:
        abs_diff[t]    = |close[t] − close[t-1]|
        actual_path[t] = rolling_sum(abs_diff, period)          (total path length)
        net_change[t]  = close[t] − close[t-period]
        straight[t]    = sqrt(net_change² + period²)            (Euclidean distance)
        direction[t]   = sign(net_change[t])
        raw[t]         = straight / actual_path × direction × 100
        PFE[t]         = EMA(raw, smooth)

    Null-prefix: ``period + smooth − 1`` bars.

    Args:
        series: Close price series.
        period: Lookback window for the efficiency calculation (default 14).
        smooth: EMA smoothing period applied to the raw PFE (default 5).

    Returns:
        Series of PFE values named ``pfe_{period}``.

    Raises:
        ValueError: If ``period < 2`` or ``smooth < 1``.

    References:
        - Chande, T. S. *Beyond Technical Analysis* (1997), pp. 68–71.
        - Investopedia — Polarized Fractal Efficiency:
          https://www.investopedia.com/terms/p/polarized-fractal-efficiency.asp
    """
    _validate_period(period, "PFE", min_period=2)
    _validate_period(smooth, "PFE smooth")

    # Actual path: sum of absolute bar-to-bar moves over `period` bars.
    # abs_diff[0] is null (diff gives null at bar 0); rolling_sum therefore
    # requires `period` non-null values, giving leading nulls = period.
    abs_diff = series.diff(1).abs()
    actual_path = abs_diff.rolling_sum(window_size=period, min_samples=period)

    # Euclidean straight-line distance in (time, price) space.
    net_change = series - series.shift(period)
    straight = (net_change**2 + float(period) ** 2).sqrt()

    # Avoid 0/0 when there is no price movement in the window.
    safe_path = actual_path.replace(0.0, float("nan"))
    pfe_raw = (straight / safe_path).fill_nan(0.0) * net_change.sign() * 100.0

    return ema(pfe_raw, smooth).alias(f"pfe_{period}")


# ---------------------------------------------------------------------------
# Chande Forecast Oscillator (CFO)
# ---------------------------------------------------------------------------


def chande_forecast_oscillator(series: pl.Series, period: int = 14) -> pl.Series:
    """Chande Forecast Oscillator — percentage deviation of close from TSF.

    The Chande Forecast Oscillator (CFO) measures the percentage by which the
    current close exceeds (or falls below) the Time Series Forecast (TSF) —
    the one-bar-ahead projection of the rolling linear regression.  Positive
    CFO means price is above the regression forecast (momentum); negative CFO
    means price is below (mean-reversion potential).

    Algorithm:
        TSF[t] = linreg one-step-ahead projection over the last *period* bars
        CFO[t] = (close[t] − TSF[t]) / close[t] × 100

    Null-prefix: ``period − 1`` bars (same as TSF).

    Args:
        series: Close price series.
        period: Regression lookback window; must be ≥ 2 (default 14).

    Returns:
        Series of CFO values (%) named ``cfo_{period}``.

    Raises:
        ValueError: If ``period < 2``.

    References:
        - Chande, T. S. & Kroll, S. *The New Technical Trader* (1994), p. 300.
        - Investopedia — Chande Forecast Oscillator:
          https://www.investopedia.com/terms/c/chandeforecastoscillator.asp
    """
    _validate_period(period, "Chande Forecast Oscillator", min_period=2)
    tsf_vals = tsf(series, period)
    # Avoid division by zero on bars where price is zero (rare in practice).
    safe_close = series.replace(0.0, float("nan"))
    return ((series - tsf_vals) / safe_close * 100.0).alias(f"cfo_{period}")


# ---------------------------------------------------------------------------
# Linear Regression R²
# ---------------------------------------------------------------------------


def linreg_r2(series: pl.Series, period: int = 14) -> pl.Series:
    """Rolling coefficient of determination (R²) from linear regression.

    R² measures the proportion of variance in the price series explained by
    the best-fit linear model over each rolling window.  Values near 1.0
    indicate that price is moving in a nearly perfect straight line (strongly
    trending); values near 0.0 indicate random, non-directional movement.

    Uses the Pearson correlation identity:
        R² = [n·Σ(xy) − Σx·Σy]² / [(n·Σx² − (Σx)²) · (n·Σy² − (Σy)²)]

    where x = bar position (0, 1, …, n−1) and y = price.  The denominator
    for x is constant and is pre-computed analytically.

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Close price series.
        period: Rolling window length; must be ≥ 2 (default 14).

    Returns:
        Series of R² values in [0, 1] named ``linreg_r2_{period}``.

    Raises:
        ValueError: If ``period < 2``.

    References:
        - Draper, N. R. & Smith, H. *Applied Regression Analysis* (1998).
        - Investopedia — R-Squared:
          https://www.investopedia.com/terms/r/r-squared.asp
    """
    _validate_period(period, "Linear Regression R²", min_period=2)

    n = period
    # x-axis constants (bar positions 0..n-1); identical to linreg_slope.
    sum_x = n * (n - 1) / 2.0
    sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0
    # Denominator for x is constant: n·Σx² − (Σx)².
    denom_x = float(n) * sum_x2 - sum_x**2

    # Weighted cross-sum: Σ(k · price[t−n+1+k]) for k = 1…n−1.
    sum_xy: pl.Series = reduce(
        operator.add,
        (series.shift(n - 1 - k) * float(k) for k in range(1, n)),
    )
    sum_y = series.rolling_sum(window_size=n, min_samples=n)
    sum_y2 = (series**2).rolling_sum(window_size=n, min_samples=n)

    # Pearson correlation numerator and denominator for y.
    numer = float(n) * sum_xy - sum_x * sum_y
    denom_y = float(n) * sum_y2 - sum_y**2

    # R² = numer² / (denom_x × denom_y).
    # fill_nan: when all prices are equal denom_y = 0 → 0/0 = NaN → set R² = 0.
    r2 = (numer**2 / (denom_x * denom_y)).fill_nan(0.0)
    return r2.alias(f"linreg_r2_{period}")


# ---------------------------------------------------------------------------
# Trend Intensity Index (TII)
# ---------------------------------------------------------------------------


def tii(series: pl.Series, period: int = 20) -> pl.Series:
    """Trend Intensity Index — fraction of closes above or below the SMA.

    The Trend Intensity Index (TII) was introduced by M. H. Pee to quantify
    how consistently price is trending relative to its simple moving average.
    It counts the fraction of bars in the most recent *period* window that
    closed above the SMA, expressed as a percentage.  Values above 50 indicate
    that the majority of recent bars are above the SMA (uptrend); values below
    50 indicate a downtrend.  Extremes near 80–100 or 0–20 may signal
    overbought/oversold conditions.

    Algorithm:
        sma[t]     = SMA(close, period)
        above[t]   = 1 if close[t] > sma[t] else 0   (null during warm-up)
        TII[t]     = rolling_sum(above, period) / period × 100

    Null-prefix: ``2 × (period − 1)`` bars — the SMA contributes period − 1
    nulls and the second rolling sum adds another period − 1 nulls.

    Args:
        series: Close price series.
        period: Lookback window for both the SMA and the count (default 20).

    Returns:
        Series of TII values in [0, 100] named ``tii_{period}``.

    Raises:
        ValueError: If ``period < 2``.

    References:
        - Pee, M. H. "Trend Intensity Index," *Technical Analysis of Stocks
          & Commodities*, June 2002.
        - Investopedia — Trend Intensity Index:
          https://www.investopedia.com/terms/t/trendintensityindex.asp
    """
    _validate_period(period, "TII", min_period=2)
    sma_vals = series.rolling_mean(window_size=period, min_samples=period)
    # The comparison gives null wherever sma_vals is null (warm-up period).
    above = (series > sma_vals).cast(pl.Float64)
    count = above.rolling_sum(window_size=period, min_samples=period)
    return (count / float(period) * 100.0).alias(f"tii_{period}")


# ---------------------------------------------------------------------------
# Moving Average Envelope (MAE)
# ---------------------------------------------------------------------------


def ma_envelope(series: pl.Series, period: int = 20, pct: float = 0.025) -> pl.DataFrame:
    """Moving Average Envelope — MA ± percentage bands.

    Moving Average Envelopes plot a simple moving average flanked by two
    bands that are a fixed percentage above and below it.  Price touching or
    piercing the upper band may signal overbought conditions; the lower band
    may signal oversold.  The distance between the bands widens during higher-
    volatility regimes.

    Algorithm:
        middle[t] = SMA(close, period)
        upper[t]  = middle[t] × (1 + pct)
        lower[t]  = middle[t] × (1 − pct)

    Null-prefix: ``period − 1`` bars (governed by the SMA).

    Args:
        series: Close price series.
        period: SMA lookback (default 20).
        pct: Envelope width as a decimal fraction of the SMA (default 0.025 = 2.5 %).

    Returns:
        DataFrame with columns ``mae_upper_{period}``, ``mae_middle_{period}``,
        ``mae_lower_{period}``.

    Raises:
        ValueError: If ``period < 1`` or ``pct <= 0``.

    References:
        - Investopedia — Envelope Channel:
          https://www.investopedia.com/terms/e/envelope-channel.asp
        - Murphy, J. J. *Technical Analysis of the Financial Markets* (1999),
          Chapter 9.
    """
    _validate_period(period, "MA Envelope")
    if pct <= 0.0:
        raise ValueError(f"MA Envelope pct must be positive; got {pct}.")
    middle = series.rolling_mean(window_size=period, min_samples=period)
    upper = middle * (1.0 + pct)
    lower = middle * (1.0 - pct)
    return pl.DataFrame(
        {
            f"mae_upper_{period}": upper,
            f"mae_middle_{period}": middle,
            f"mae_lower_{period}": lower,
        }
    )


# ---------------------------------------------------------------------------
# Linear Regression Intercept
# ---------------------------------------------------------------------------


def linreg_intercept(series: pl.Series, period: int = 14) -> pl.Series:
    """Rolling linear regression y-intercept (OLS constant at x = 0).

    For each rolling window of length *period*, fits an OLS line
    ``y = slope·x + intercept`` where x = [0, 1, …, period−1] and returns
    the intercept (the fitted value at x = 0, the oldest bar in the window).
    Together with ``linreg_slope`` this fully specifies the regression line;
    the fitted value at the *last* bar is the Time Series Forecast (``tsf``).

    Algorithm:
        mean_x    = (period − 1) / 2
        mean_y    = rolling_mean(close, period)
        ss_xx     = sum((i − mean_x)² for i in 0..period−1)
        ss_xy     = sum((i − mean_x) × (y_i − mean_y))
        slope     = ss_xy / ss_xx
        intercept = mean_y − slope × mean_x

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Close price series.
        period: Regression window length (default 14).

    Returns:
        Series of intercept values named ``linreg_intercept_{period}``.

    Raises:
        ValueError: If ``period < 2``.

    References:
        - Investopedia — Linear Regression:
          https://www.investopedia.com/terms/l/linearrelationship.asp
        - TA-Lib LINEARREG_INTERCEPT:
          https://ta-lib.org/function.html#LINEARREG_INTERCEPT
    """
    _validate_period(period, "LinReg Intercept", min_period=2)

    def _intercept(w: pl.Series) -> float:
        """OLS intercept for a single window of length n."""
        n = len(w)
        vals: list[float] = w.to_list()
        mean_y: float = sum(vals) / n
        mean_x = (n - 1) / 2.0
        ss_xx: float = sum((i - mean_x) ** 2 for i in range(n))
        if ss_xx == 0.0:
            # Flat series: slope = 0, intercept = mean.
            return mean_y
        ss_xy: float = sum((i - mean_x) * (vals[i] - mean_y) for i in range(n))
        slope: float = ss_xy / ss_xx
        return mean_y - slope * mean_x

    return series.rolling_map(
        function=_intercept,
        window_size=period,
        min_samples=period,
    ).alias(f"linreg_intercept_{period}")


# ---------------------------------------------------------------------------
# Standard Error Bands
# ---------------------------------------------------------------------------


def standard_error_bands(series: pl.Series, period: int = 21) -> pl.DataFrame:
    """Linear regression line ± 2 × residual standard error.

    Standard Error Bands (Jon Andersen, 1996) plot the rolling OLS regression
    line flanked by bands at ±2 standard errors of the residuals.  Unlike
    Bollinger Bands (which use the standard deviation of *price*), the bands
    here reflect how tightly price adheres to its *trend line*, contracting in
    trending markets and expanding in choppy ones.

    Algorithm (per window of length n ≥ 3):
        fit predicted values  ŷ_i = intercept + slope × i
        SSR = sum((y_i − ŷ_i)²)
        SE  = sqrt(SSR / (n − 2))    (residual standard error, ddof = 2)
        upper = ŷ_{n−1} + 2 × SE
        middle= ŷ_{n−1}              (= TSF at the last bar)
        lower = ŷ_{n−1} − 2 × SE

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Close price series.
        period: Regression window length (default 21).  Must be ≥ 3 to allow
                residual ddof = 2.

    Returns:
        DataFrame with columns ``seb_upper_{period}``, ``seb_middle_{period}``,
        ``seb_lower_{period}``.

    Raises:
        ValueError: If ``period < 3``.

    References:
        - Andersen, J. "Standard Error Bands," *Technical Analysis of Stocks
          & Commodities* (1996).
        - Investopedia — Standard Error:
          https://www.investopedia.com/terms/s/standard-error.asp
    """
    _validate_period(period, "Standard Error Bands", min_period=3)

    def _seb(w: pl.Series) -> tuple[float, float, float]:
        """OLS fit returning (upper, middle, lower) for a single window."""
        n = len(w)
        vals: list[float] = w.to_list()
        mean_y: float = sum(vals) / n
        mean_x = (n - 1) / 2.0
        ss_xx: float = sum((i - mean_x) ** 2 for i in range(n))
        if ss_xx == 0.0:
            # Flat series: no slope, no spread.
            return mean_y, mean_y, mean_y
        ss_xy: float = sum((i - mean_x) * (vals[i] - mean_y) for i in range(n))
        slope: float = ss_xy / ss_xx
        intercept: float = mean_y - slope * mean_x
        # Predicted values and residual sum of squares.
        ssr: float = sum((vals[i] - (intercept + slope * i)) ** 2 for i in range(n))
        se: float = (ssr / (n - 2)) ** 0.5
        # Return fitted value at the last bar ± 2 SE.
        fitted_last: float = intercept + slope * (n - 1)
        return fitted_last + 2.0 * se, fitted_last, fitted_last - 2.0 * se

    # rolling_map returns a single Series; we need three outputs.
    # Compute each band by injecting a small adapter around _seb.
    def _upper(w: pl.Series) -> float:
        return _seb(w)[0]

    def _middle(w: pl.Series) -> float:
        return _seb(w)[1]

    def _lower(w: pl.Series) -> float:
        return _seb(w)[2]

    upper = series.rolling_map(function=_upper, window_size=period, min_samples=period)
    middle = series.rolling_map(function=_middle, window_size=period, min_samples=period)
    lower = series.rolling_map(function=_lower, window_size=period, min_samples=period)

    return pl.DataFrame(
        {
            f"seb_upper_{period}": upper,
            f"seb_middle_{period}": middle,
            f"seb_lower_{period}": lower,
        }
    )


# ---------------------------------------------------------------------------
# Centre of Gravity (COG)
# ---------------------------------------------------------------------------


def cog(series: pl.Series, period: int = 10) -> pl.Series:
    """Centre of Gravity oscillator — Ehlers' price-weighted lag estimator.

    The Centre of Gravity (COG) oscillator was introduced by John Ehlers
    (2002) as a near-zero-lag indicator that identifies turning points in
    price.  It computes a weighted average of the last *period* prices where
    the weight of each bar equals its distance from the current bar (1 for
    the most recent, *period* for the oldest).  The result is negated so that
    upward turning points produce local minima.

    Algorithm (current bar = index 0, oldest = index period−1):
        numerator[t]   = sum(close[t−i] × (i + 1) for i in 0 … period−1)
        denominator[t] = sum(close[t−i] for i in 0 … period−1)
        COG[t]         = −numerator[t] / denominator[t]

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Close price series.
        period: Lookback window (default 10).

    Returns:
        Series of COG values named ``cog_{period}``.  Values are negative
        numbers; peaks correspond to price cycle lows, troughs to highs.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - Ehlers, J. F. "Center of Gravity Indicator," *Technical Analysis of
          Stocks & Commodities* (May 2002).
        - Investopedia — Center of Gravity Indicator:
          https://www.investopedia.com/terms/c/center-of-gravity-cog-indicator.asp
    """
    _validate_period(period, "Centre of Gravity")

    # Decreasing weights: most recent bar gets weight 1, oldest gets weight `period`.
    weights = list(range(1, period + 1))

    def _cog_window(w: pl.Series) -> float:
        """Compute COG for a single window (oldest → newest order)."""
        # Polars rolling_map passes windows oldest-first; reverse for COG weights.
        vals: list[float] = w.to_list()
        numer = sum(v * wt for v, wt in zip(reversed(vals), weights, strict=True))
        denom = sum(vals)
        if denom == 0.0:
            return float("nan")
        return -numer / denom

    return series.rolling_map(
        function=_cog_window,
        window_size=period,
        min_samples=period,
    ).alias(f"cog_{period}")


# ---------------------------------------------------------------------------
# Random Walk Index (RWI)
# ---------------------------------------------------------------------------


def rwi(ohlc: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Random Walk Index — tests whether price moves exceed a random walk.

    The Random Walk Index (Michael Poulos, 1992) compares the observed price
    range over *period* bars to what would be expected from a purely random
    walk of the same length.  RWI_High > 1 signals an uptrend stronger than
    chance; RWI_Low > 1 signals a downtrend stronger than chance.  Both
    exceeding 1 simultaneously is rare and suggests a volatile, directional
    market.

    Algorithm:
        ATR[t]     = Average True Range(ohlc, period)
        RWI_High[t] = (H[t] − L[t−period]) / (ATR[t] × sqrt(period))
        RWI_Low[t]  = (H[t−period] − L[t]) / (ATR[t] × sqrt(period))

    Null-prefix: ``period`` bars (ATR and the *period*-bar shifts both
    require *period* prior bars).

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: Lookback window (default 14).

    Returns:
        DataFrame with columns ``rwi_high_{period}`` and ``rwi_low_{period}``.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - Poulos, M. "Of Trends and Random Walks," *Technical Analysis of
          Stocks & Commodities* (1992).
        - Investopedia — Random Walk Index:
          https://www.investopedia.com/terms/r/random-walk-index.asp
    """
    _validate_period(period, "RWI")
    h = ohlc["high"]
    lo = ohlc["low"]

    atr_vals = atr(ohlc, period)
    denom = atr_vals * (period**0.5)
    safe_denom = denom.replace(0.0, float("nan"))

    # Compare current high to the low `period` bars ago (upward sweep).
    rwi_high = (h - lo.shift(period)) / safe_denom
    # Compare the high `period` bars ago to current low (downward sweep).
    rwi_low = (h.shift(period) - lo) / safe_denom

    return pl.DataFrame(
        {
            f"rwi_high_{period}": rwi_high,
            f"rwi_low_{period}": rwi_low,
        }
    )
