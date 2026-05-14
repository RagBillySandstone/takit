"""
Moving average indicators.

All functions accept a ``pl.Series`` and return a ``pl.Series``.
The first ``(period - 1)`` values of any windowed output are ``null``
rather than zero, preserving null-propagation semantics in Polars.

Functions
---------
sma               Simple Moving Average
ema               Exponential Moving Average  (α = 2 / (n + 1))
wma               Weighted Moving Average     (linearly weighted)
wilder_smooth     Wilder's Smoothing / RMA    (α = 1 / n)
dema              Double EMA                  (2·EMA - EMA(EMA))
tema              Triple EMA                  (3·EMA - 3·EMA(EMA) + EMA(EMA(EMA)))
hma               Hull Moving Average         (WMA of 2·WMA(n/2) - WMA(n))
vwma              Volume-Weighted Moving Average
mcginley_dynamic  McGinley Dynamic            (self-adjusting, tracks price closely)
kama              Kaufman Adaptive Moving Average (ER-based adaptive smoothing)
trix              Triple-smoothed EMA oscillator with signal line
zlema             Zero Lag EMA                (lag-corrected via error-correction term)
t3                Tillson T3                  (triple GD expansion; low-lag, smooth)
alma              Arnaud Legoux Moving Average (Gaussian-weighted rolling mean)
"""

from __future__ import annotations

import math
import operator
from functools import reduce

import polars as pl

from polarticks._validate import _validate_period

# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


def sma(series: pl.Series, period: int) -> pl.Series:
    """Simple Moving Average over a rolling window.

    Computes the arithmetic mean of the last *period* values.  The first
    ``period - 1`` values are ``null``.

    Args:
        series: Input price series.
        period: Lookback window (number of bars).

    Returns:
        Series of SMA values aligned with the input.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "SMA")
    return series.rolling_mean(window_size=period, min_samples=period).alias(f"sma_{period}")


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def ema(series: pl.Series, period: int) -> pl.Series:
    """Exponential Moving Average using the standard smoothing factor.

    Smoothing factor: α = 2 / (period + 1).  The series is seeded with
    the SMA of the first *period* bars (``adjust=False, min_periods=period``).
    Values before the seed point are ``null``.

    Args:
        series: Input price series.
        period: Lookback period — larger values produce more smoothing.

    Returns:
        Series of EMA values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "EMA")
    alpha = 2.0 / (period + 1)
    return series.ewm_mean(
        alpha=alpha,
        adjust=False,
        min_samples=period,
    ).alias(f"ema_{period}")


# ---------------------------------------------------------------------------
# WMA
# ---------------------------------------------------------------------------


def wma(series: pl.Series, period: int) -> pl.Series:
    """Weighted Moving Average with linearly increasing weights.

    Each bar within the window receives a weight proportional to its
    position: the most recent bar has weight *period*, the oldest has
    weight 1.  The sum of weights for a window of size *n* is
    ``n * (n + 1) / 2``.

    The first ``period - 1`` values are ``null``.

    Args:
        series: Input price series.
        period: Lookback window.

    Returns:
        Series of WMA values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "WMA")

    weights = list(range(1, period + 1))
    weight_sum = sum(weights)

    # Sum shifted copies of the series, each multiplied by its linear weight.
    # shift(k) moves the series forward k bars, so shift(0) = most recent,
    # shift(period-1) = oldest bar in the window.  Null propagates naturally
    # through the summation — no explicit warm-up mask is needed.
    weighted: pl.Series = reduce(
        operator.add,
        (series.shift(period - 1 - i) * weights[i] for i in range(period)),
    )
    return (weighted / weight_sum).alias(f"wma_{period}")


# ---------------------------------------------------------------------------
# Wilder's Smoothing (RMA)
# ---------------------------------------------------------------------------


def wilder_smooth(series: pl.Series, period: int) -> pl.Series:
    """Wilder's Smoothing, also known as the Running Moving Average (RMA).

    Uses a smoothing factor of α = 1 / period, which is more conservative
    than standard EMA (α = 2 / (n + 1)).  Used internally by RSI and ATR
    to match Wilder's original formulas.

    Args:
        series: Input series.
        period: Lookback period.

    Returns:
        Series of Wilder-smoothed values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Wilder smooth")
    alpha = 1.0 / period
    return series.ewm_mean(
        alpha=alpha,
        adjust=False,
        min_samples=period,
    ).alias(f"wilder_{period}")


# ---------------------------------------------------------------------------
# DEMA
# ---------------------------------------------------------------------------


def dema(series: pl.Series, period: int) -> pl.Series:
    """Double Exponential Moving Average.

    DEMA reduces the lag of a standard EMA by applying a correction:

        DEMA = 2 · EMA(series, n) − EMA(EMA(series, n), n)

    The warm-up period is ``2 * (period - 1)`` bars, after which both the
    outer EMA and the EMA-of-EMA are defined.

    Args:
        series: Input price series.
        period: EMA period.

    Returns:
        Series of DEMA values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "DEMA")
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    return (2.0 * ema1 - ema2).alias(f"dema_{period}")


# ---------------------------------------------------------------------------
# TEMA
# ---------------------------------------------------------------------------


def tema(series: pl.Series, period: int) -> pl.Series:
    """Triple Exponential Moving Average.

    Further reduces lag compared to DEMA:

        TEMA = 3 · EMA(series, n)
             − 3 · EMA(EMA(series, n), n)
             +     EMA(EMA(EMA(series, n), n), n)

    The warm-up period is ``3 * (period - 1)`` bars.

    Args:
        series: Input price series.
        period: EMA period.

    Returns:
        Series of TEMA values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "TEMA")
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return (3.0 * ema1 - 3.0 * ema2 + ema3).alias(f"tema_{period}")


# ---------------------------------------------------------------------------
# HMA
# ---------------------------------------------------------------------------


def hma(series: pl.Series, period: int) -> pl.Series:
    """Hull Moving Average — reduces lag while remaining smooth.

    HMA applies a WMA over a derived series that halves conventional lag:

        raw      = 2 · WMA(series, n/2) − WMA(series, n)
        HMA(n)   = WMA(raw, √n)

    The ``n/2`` half-period uses integer division; the smoothing window
    uses ``round(√n)``.  The warm-up is ``(period - 1) + (round(√period) - 1)``
    bars, after which both underlying WMAs and the final WMA are defined.

    Args:
        series: Input price series.
        period: Primary lookback window.

    Returns:
        Series of HMA values.

    Raises:
        ValueError: If ``period < 2`` (need at least two bars for a half-period).
    """
    _validate_period(period, "HMA", min_period=2)

    half_period = period // 2
    sqrt_period = round(period**0.5)

    # The intermediate series amplifies the short-term WMA to reduce lag.
    raw = 2.0 * wma(series, half_period) - wma(series, period)

    return wma(raw, sqrt_period).alias(f"hma_{period}")


# ---------------------------------------------------------------------------
# VWMA
# ---------------------------------------------------------------------------


def vwma(price: pl.Series, volume: pl.Series, period: int) -> pl.Series:
    """Volume-Weighted Moving Average.

    Each bar within the window is weighted by its volume rather than by
    its position (as in WMA) or equally (as in SMA).  VWMA tracks price
    more closely during high-volume bars and gives less weight to
    low-volume noise.

        VWMA(n) = Σ(price[i] × volume[i]) / Σ(volume[i])   over the last n bars

    The first ``period - 1`` values are ``null``.

    Args:
        price: Price series (e.g. close).
        volume: Volume series aligned bar-for-bar with *price*.
        period: Lookback window.

    Returns:
        Series of VWMA values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "VWMA")

    pv_sum = (price * volume.cast(pl.Float64)).rolling_sum(window_size=period, min_samples=period)
    vol_sum = volume.cast(pl.Float64).rolling_sum(window_size=period, min_samples=period)

    # fill_nan handles the edge case of a window with zero total volume.
    return (pv_sum / vol_sum).fill_nan(None).alias(f"vwma_{period}")


# ---------------------------------------------------------------------------
# McGinley Dynamic
# ---------------------------------------------------------------------------


def mcginley_dynamic(series: pl.Series, period: int) -> pl.Series:
    """McGinley Dynamic — self-adjusting moving average.

    The McGinley Dynamic adapts its speed to market velocity so that it
    stays closer to price during fast moves and is smoother during slow
    ones.  Unlike EMA or WMA it does not require the trader to choose a
    period that matches the current market rhythm — the formula
    self-corrects:

        MD[t] = MD[t-1] + (price - MD[t-1]) / (N × (price / MD[t-1])⁴)

    The indicator is seeded with the SMA of the first *period* bars.
    Values before the seed point are ``null``.

    Args:
        series: Input price series.
        period: Nominal period (controls the denominator constant N).

    Returns:
        Series of McGinley Dynamic values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "McGinley Dynamic")

    raw_values: list[float | None] = series.to_list()
    output: list[float | None] = [None] * len(raw_values)

    # Seed with the SMA of the first period bars; skip if any input is null.
    seed_window = [v for v in raw_values[:period] if v is not None]
    if len(seed_window) < period:
        # Not enough non-null data to seed — return all-null series.
        return pl.Series(f"mcginley_{period}", output, dtype=pl.Float64)

    md = sum(seed_window) / period
    output[period - 1] = md

    for idx in range(period, len(raw_values)):
        price = raw_values[idx]
        if price is None or md == 0.0:
            # Propagate null on missing data or degenerate seed.
            output[idx] = None
            continue
        # The (price/MD)^4 term makes the denominator larger (slower) when
        # price is far above MD and smaller (faster) when price is far below.
        md = md + (price - md) / (period * (price / md) ** 4)
        output[idx] = md

    return pl.Series(f"mcginley_{period}", output, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# KAMA
# ---------------------------------------------------------------------------


def kama(
    series: pl.Series,
    period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> pl.Series:
    """Kaufman Adaptive Moving Average.

    KAMA self-adjusts its smoothing speed based on the *Efficiency Ratio* (ER),
    which compares net price movement (signal) against total path length (noise).
    In trending markets the ER is high and KAMA tracks price quickly; in choppy
    markets the ER is low and KAMA barely moves.

    Algorithm:
        abs_diffs[t]    = |price[t] − price[t-1]|
        direction[t]    = |price[t] − price[t-period]|   (net move over period)
        volatility[t]   = Σ abs_diffs over the last period bars  (total path)
        ER[t]           = direction[t] / volatility[t]
        fast_sc         = 2 / (fast_period + 1)
        slow_sc         = 2 / (slow_period + 1)
        SC[t]           = (ER[t] × (fast_sc − slow_sc) + slow_sc)²
        KAMA[t]         = KAMA[t-1] + SC[t] × (price[t] − KAMA[t-1])

    The indicator is seeded at bar ``period - 1`` with the raw price; the first
    ``period - 1`` output values are ``null``.

    Args:
        series: Input price series (e.g., close).
        period: Lookback window for the Efficiency Ratio (default 10).
        fast_period: Fastest EMA period used in the smoothing constant (default 2).
        slow_period: Slowest EMA period used in the smoothing constant (default 30).

    Returns:
        Series of KAMA values.  The first ``period - 1`` values are ``null``.

    Raises:
        ValueError: If any period is below its minimum or ``fast_period >= slow_period``.
    """
    _validate_period(period, "KAMA")
    _validate_period(fast_period, "KAMA fast_period")
    _validate_period(slow_period, "KAMA slow_period")
    if fast_period >= slow_period:
        raise ValueError(
            f"KAMA fast_period ({fast_period}) must be less than slow_period ({slow_period})."
        )

    # Smoothing constants for the fastest and slowest EMA bounds.
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)

    raw: list[float | None] = series.to_list()
    n = len(raw)
    output: list[float | None] = [None] * n

    # Pre-build absolute bar-to-bar differences; index 0 is a sentinel 0.0
    # because bar 0 has no predecessor.
    abs_diffs: list[float] = [0.0] + [
        abs(raw[i] - raw[i - 1])  # type: ignore[operator]
        if (raw[i] is not None and raw[i - 1] is not None)
        else 0.0
        for i in range(1, n)
    ]

    seed_idx = period - 1
    if raw[seed_idx] is None:
        return pl.Series(f"kama_{period}", output, dtype=pl.Float64)

    kama_val: float = raw[seed_idx]  # type: ignore[assignment]
    output[seed_idx] = kama_val

    # Initialise the sliding-window sum with the period-1 diffs that precede
    # the first computation bar (idx=period).  The loop then extends the window
    # by one diff before computing and shrinks it by one diff after, keeping
    # the window size constant at exactly period bars.
    window_vol: float = sum(abs_diffs[1:period])

    for idx in range(period, n):
        price = raw[idx]

        # Extend the window with the newest diff before computing or skipping.
        window_vol += abs_diffs[idx]

        if price is not None and raw[idx - period] is not None:
            direction = abs(price - raw[idx - period])  # type: ignore[operator]
            # ER = 0 when the market is perfectly choppy (no net movement).
            er = direction / window_vol if window_vol != 0.0 else 0.0

            # Squaring the SC compresses the range and ensures non-negative values.
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

            kama_val = kama_val + sc * (price - kama_val)
            output[idx] = kama_val

        # Evict the oldest diff so the window stays at exactly period bars.
        window_vol -= abs_diffs[idx - period + 1]

    return pl.Series(f"kama_{period}", output, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# TRIX
# ---------------------------------------------------------------------------


def trix(
    series: pl.Series,
    period: int = 14,
    signal: int = 9,
) -> pl.DataFrame:
    """TRIX — triple-smoothed EMA rate-of-change oscillator.

    TRIX applies three successive EMA passes to filter out short cycles and
    market noise, then computes the percentage rate of change of the result.
    It oscillates around zero like MACD but is more resistant to whipsaws.
    A cross of the TRIX line above the signal line is a buy signal.

    Algorithm:
        ema1      = EMA(series, period)
        ema2      = EMA(ema1,   period)
        ema3      = EMA(ema2,   period)
        trix_line = 100 × (ema3[t] − ema3[t-1]) / ema3[t-1]
        trix_signal    = EMA(trix_line, signal)
        trix_histogram = trix_line − trix_signal

    Null-prefix for ``trix_line``:      ``3 × (period - 1) + 1`` bars.
    Null-prefix for ``trix_signal``:    ``3 × (period - 1) + signal`` bars.
    Null-prefix for ``trix_histogram``: same as ``trix_signal``.

    Args:
        series: Input price series (e.g., close).
        period: EMA period applied three times (default 14).
        signal: EMA period for the signal line (default 9).

    Returns:
        DataFrame with columns ``trix_line``, ``trix_signal``, ``trix_histogram``.

    Raises:
        ValueError: If ``period < 1`` or ``signal < 1``.
    """
    _validate_period(period, "TRIX")
    _validate_period(signal, "TRIX signal")

    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)

    # Percentage change of the triple-smoothed EMA; fill_nan converts any
    # division-by-zero (flat or zero-valued ema3) to null.
    trix_line = (100.0 * (ema3 - ema3.shift(1)) / ema3.shift(1)).fill_nan(None).alias("trix_line")

    signal_line = ema(trix_line, signal).alias("trix_signal")
    histogram = (trix_line - signal_line).alias("trix_histogram")

    return pl.DataFrame(
        {"trix_line": trix_line, "trix_signal": signal_line, "trix_histogram": histogram}
    )


# ---------------------------------------------------------------------------
# ZLEMA
# ---------------------------------------------------------------------------


def zlema(series: pl.Series, period: int) -> pl.Series:
    """Zero Lag Exponential Moving Average.

    ZLEMA reduces EMA lag by adjusting the input with an error-correction term
    that compensates for the half-period delay inherent in a standard EMA.

    Algorithm:
        lag      = (period − 1) // 2
        adjusted = 2 × price[t] − price[t − lag]
        ZLEMA    = EMA(adjusted, period)

    The de-lagged input introduces ``lag`` leading nulls before the EMA
    warm-up, giving a total null-prefix of ``lag + (period − 1)`` bars.

    Args:
        series: Input price series (e.g., close).
        period: EMA period.

    Returns:
        Series of ZLEMA values.  First ``(period − 1) // 2 + period − 1`` values
        are ``null``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "ZLEMA")

    lag = (period - 1) // 2
    # De-lagged input: emphasise current price direction by doubling and
    # subtracting the lagged copy; shift(lag) introduces lag leading nulls.
    adjusted = (2.0 * series - series.shift(lag)).alias("zlema_adj")
    return ema(adjusted, period).alias(f"zlema_{period}")


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------


def t3(series: pl.Series, period: int = 5, vfactor: float = 0.7) -> pl.Series:
    """Tillson T3 Moving Average — low-lag triple-smoothed EMA.

    T3 applies six successive EMA passes and combines them with weights
    derived from the *volume factor* (``vfactor``), which is mathematically
    equivalent to applying the Generalised Double EMA (GD) three times:

        GD(x, v)  = (1 + v) × EMA(x, n) − v × EMA(EMA(x, n), n)
        T3        = GD(GD(GD(price, v), v), v)

    Expanding the three GD applications yields:
        T3 = c3 × e3 + c4 × e4 + c5 × e5 + c6 × e6

    where e1…e6 are successive EMA passes and:
        c3 =  (1 + v)³
        c4 = −3v(1 + v)²
        c5 =  3v²(1 + v)
        c6 = −v³

    With ``vfactor = 0``, T3 degenerates to the plain triple EMA (e3).

    Null-prefix: ``6 × (period − 1)`` bars.

    Args:
        series: Input price series (e.g., close).
        period: EMA period applied at each of the six passes (default 5).
        vfactor: Volume factor in [0, 1] trading smoothness for responsiveness
                 (default 0.7).

    Returns:
        Series of T3 values.  The first ``6 × (period − 1)`` values are ``null``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "T3")

    # Six successive EMA passes; each adds (period − 1) leading nulls.
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    e4 = ema(e3, period)
    e5 = ema(e4, period)
    e6 = ema(e5, period)

    # Binomial expansion coefficients for GD applied three times.
    v = vfactor
    c3 = (1.0 + v) ** 3
    c4 = -3.0 * v * (1.0 + v) ** 2
    c5 = 3.0 * v**2 * (1.0 + v)
    c6 = -(v**3)

    return (c3 * e3 + c4 * e4 + c5 * e5 + c6 * e6).alias(f"t3_{period}")


# ---------------------------------------------------------------------------
# ALMA
# ---------------------------------------------------------------------------


def alma(
    series: pl.Series,
    period: int = 9,
    offset: float = 0.85,
    sigma: float = 6.0,
) -> pl.Series:
    """Arnaud Legoux Moving Average — Gaussian-weighted rolling mean.

    ALMA applies a Gaussian window centred at ``offset`` within the lookback
    period.  Placing the peak near the recent end (``offset = 0.85``) reduces
    lag; widening the bell (lower ``sigma``) increases smoothness at the cost
    of responsiveness.

    Algorithm:
        mu      = offset × (period − 1)
        s       = period / sigma
        w_k     = exp(−(k − mu)² / (2 × s²)),  k = 0 … period − 1
        ALMA[t] = Σ_k (w_k / Σ w) × price[t − (period − 1 − k)]

    Null-prefix: ``period − 1`` bars (same as SMA/WMA).

    Args:
        series: Input price series (e.g., close).
        period: Lookback window (default 9).
        offset: Gaussian peak position within the window as a fraction in
                [0, 1]; 0 = oldest bar, 1 = newest bar (default 0.85).
        sigma: Gaussian width divisor; higher values narrow the bell curve
               and reduce smoothing (default 6.0).

    Returns:
        Series of ALMA values.  The first ``period − 1`` values are ``null``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "ALMA")

    # Build normalised Gaussian weights over the lookback window.
    mu = offset * (period - 1)
    s = period / sigma
    raw_weights = [math.exp(-((k - mu) ** 2) / (2.0 * s**2)) for k in range(period)]
    weight_sum = sum(raw_weights)
    norm_weights = [w / weight_sum for w in raw_weights]

    # Weighted sum of shifted copies: shift(period-1-k) aligns the bar at
    # position k within the window (k=0 is oldest, k=period-1 is current bar).
    weighted: pl.Series = reduce(
        operator.add,
        (series.shift(period - 1 - i) * norm_weights[i] for i in range(period)),
    )
    return weighted.alias(f"alma_{period}")
