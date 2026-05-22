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
var_mov_avg       Variable Moving Average     (signal/noise ER-adaptive smoothing)
frama             Fractal Adaptive Moving Average (fractal-dimension adaptive smoothing)
laguerre          Laguerre Filter             (four-element low-lag smoother, Ehlers)
trima             Triangular Moving Average   (double-smoothed SMA; triangular weights)
vidya             Variable Index Dynamic Average (CMO-adaptive EMA, Chande 1994)
ehma              Exponential Hull Moving Average (EMA variant of HMA; low-lag, smooth)
pwma              Pascal's Weighted Moving Average (binomial-coefficient weighting)
mama              MESA Adaptive Moving Average + Following Adaptive MA  (Ehlers 2004)
dominant_cycle_period  Hilbert Transform Dominant Cycle Period         (Ehlers 2004)
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


# ---------------------------------------------------------------------------
# VarMovAvg
# ---------------------------------------------------------------------------


def var_mov_avg(
    series: pl.Series,
    period: int = 50,
    nfast: int = 15,
    nslow: int = 10,
    g: float = 1.0,
) -> pl.Series:
    """Variable Moving Average — signal/noise Efficiency Ratio adaptive MA.

    VarMovAvg self-adjusts its smoothing speed by computing an Efficiency Ratio
    (ER) that compares the net price movement (signal) against the total path
    length (noise) over ``period`` bars.  The resulting smoothing constant (SSC)
    is raised to the power ``g`` before being applied, allowing the user to
    amplify or dampen the adaptive effect.

    Based on Var_Mov_Avg3.mq4 by GOODMAN & Mstera & AF, as published by
    EarnForex (https://github.com/EarnForex/VarMovAvg).

    Algorithm:
        abs_diff[t]  = |price[t] − price[t−1]|
        noise[t]     = Σ abs_diff over [t−period+1, t]  (+ ε to prevent /0)
        signal[t]    = |price[t] − price[t−period]|
        ER[t]        = signal[t] / noise[t]
        slow_sc      = 2 / (nslow + 1)
        fast_sc      = 2 / (nfast + 1)
        SSC[t]       = ER[t] × (fast_sc − slow_sc) + slow_sc
        VMA[t]       = VMA[t−1] + SSC[t]^G × (price[t] − VMA[t−1])

    Seeding: VMA is seeded at bar ``period`` with the raw price; the first
    output appears at bar ``period + 1``.  The first ``period + 1`` values
    are ``null``.

    Note on ``nfast`` / ``nslow`` naming: the original MT4/MT5 indicator uses
    ``nslow=10`` and ``nfast=15`` as defaults.  With these values
    ``slow_sc > fast_sc`` and ``fast_sc − slow_sc`` is negative, so ER=1
    (trending) gives the *smaller* SSC while ER=0 (choppy) gives the larger
    one.  The difference is mild with these defaults; increasing the gap or
    ``g`` sharpens the adaptive response.

    Args:
        series: Input price series (e.g., close).
        period: Efficiency Ratio lookback window (default 50).
        nfast: Fast smoothing period used in the SSC floor/ceiling (default 15).
        nslow: Slow smoothing period used in the SSC floor/ceiling (default 10).
        g: Power exponent applied to SSC before updating VMA (default 1.0).

    Returns:
        Series of VarMovAvg values.  The first ``period + 1`` values are
        ``null``.

    Raises:
        ValueError: If ``period < 1``, ``nfast < 1``, or ``nslow < 1``.
    """
    _validate_period(period, "VarMovAvg")
    _validate_period(nfast, "VarMovAvg nfast")
    _validate_period(nslow, "VarMovAvg nslow")

    slow_sc = 2.0 / (nslow + 1)
    fast_sc = 2.0 / (nfast + 1)
    dsc = fast_sc - slow_sc

    # Vectorised ER: noise is the rolling sum of bar-to-bar absolute changes;
    # the epsilon prevents division by zero on perfectly flat price series.
    abs_diff = (series - series.shift(1)).abs()
    noise = abs_diff.rolling_sum(window_size=period, min_samples=period) + 1e-9
    signal = (series - series.shift(period)).abs()
    ssc_series = signal / noise * dsc + slow_sc

    raw: list[float | None] = series.to_list()
    ssc_list: list[float | None] = ssc_series.to_list()
    n = len(raw)
    output: list[float | None] = [None] * n

    # Seed AMA0 with the raw price at index ``period``; first output is at
    # index ``period + 1`` (matching the MQL5 reference PlotIndexGetInteger
    # draw-begin of periodAMA + 2, i.e., first non-null at periodAMA + 1).
    seed_idx = period + 1
    if seed_idx >= n or raw[period] is None:
        return pl.Series(f"var_mov_avg_{period}", output, dtype=pl.Float64)

    ama: float = raw[period]  # type: ignore[assignment]

    for idx in range(seed_idx, n):
        price = raw[idx]
        ssc = ssc_list[idx]
        if price is None or ssc is None:
            output[idx] = None
            continue
        # SSC^g scales the adaptive step; with g=1 this reduces to a standard
        # SC-weighted update identical to KAMA.
        ddk = ssc**g * (price - ama)
        ama = ama + ddk
        output[idx] = ama

    return pl.Series(f"var_mov_avg_{period}", output, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# FRAMA
# ---------------------------------------------------------------------------


def frama(series: pl.Series, period: int = 16) -> pl.Series:
    """Fractal Adaptive Moving Average — fractal-dimension adaptive EMA.

    FRAMA (Ehlers, 2004) adapts its smoothing factor based on the fractal
    dimension of the price series.  In trending markets the dimension
    approaches 1 (straight line) and FRAMA tracks price quickly; in choppy
    markets the dimension approaches 2 (Brownian motion) and FRAMA barely
    moves, filtering out noise.

    Algorithm:
        half = period // 2
        n1   = (max(first_half) − min(first_half)) / half
        n2   = (max(second_half) − min(second_half)) / half
        n3   = (max(full_window) − min(full_window)) / period
        D    = (log(n1 + n2) − log(n3)) / log(2)
        alpha = exp(−4.6 × (D − 1)),  clamped to [0.01, 1]
        FRAMA[t] = alpha × price[t] + (1 − alpha) × FRAMA[t−1]

    ``period`` must be even (odd values are rounded down to the nearest even).
    Null-prefix: ``period − 1`` bars (seeded at bar ``period − 1``).

    Args:
        series: Input price series (e.g., close).
        period: Lookback window; must be even and ≥ 4 (default 16).

    Returns:
        Series of FRAMA values. The first ``period − 1`` values are ``null``.

    Raises:
        ValueError: If ``period < 4``.
    """
    _validate_period(period, "FRAMA", min_period=4)
    # Silently enforce even period — fractal dimension requires equal sub-windows.
    if period % 2 != 0:
        period = period - 1

    half = period // 2
    raw: list[float | None] = series.to_list()
    n = len(raw)
    output: list[float | None] = [None] * n

    # Seed FRAMA at bar period-1 with the raw price.
    seed_idx = period - 1
    if seed_idx >= n or raw[seed_idx] is None:
        return pl.Series(f"frama_{period}", output, dtype=pl.Float64)

    frama_val: float = raw[seed_idx]  # type: ignore[assignment]
    output[seed_idx] = frama_val

    for idx in range(period, n):
        price = raw[idx]
        if price is None:
            output[idx] = None
            continue

        # Extract the three windows: first half, second half, and full period.
        full = [v for v in raw[idx - period + 1 : idx + 1] if v is not None]
        first = [v for v in raw[idx - period + 1 : idx - half + 1] if v is not None]
        second = [v for v in raw[idx - half + 1 : idx + 1] if v is not None]

        if len(full) < period or len(first) < half or len(second) < half:
            output[idx] = None
            continue

        # Fractal dimension via the log-ratio of range estimates.
        n3 = (max(full) - min(full)) / period
        n1 = (max(first) - min(first)) / half
        n2 = (max(second) - min(second)) / half

        if n3 > 0.0 and (n1 + n2) > 0.0:
            d = (math.log(n1 + n2) - math.log(n3)) / math.log(2.0)
            d = max(1.0, min(2.0, d))  # clamp to valid fractal-dimension range
            alpha = math.exp(-4.6 * (d - 1.0))
            alpha = max(0.01, min(1.0, alpha))
        else:
            # Flat sub-window: fall back to minimum (slow) smoothing.
            alpha = 0.01

        frama_val = alpha * price + (1.0 - alpha) * frama_val
        output[idx] = frama_val

    return pl.Series(f"frama_{period}", output, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# Laguerre Filter
# ---------------------------------------------------------------------------


def laguerre(series: pl.Series, gamma: float = 0.8) -> pl.Series:
    """Laguerre Filter — four-element state-space low-lag smoother.

    The Laguerre Filter (Ehlers, 2004) achieves low lag by maintaining a
    four-element state machine with a single ``gamma`` parameter that
    controls the lag/smoothness trade-off.  Lower ``gamma`` → less lag but
    noisier; higher ``gamma`` → smoother but more lagged.

    Algorithm:
        L0[t] = (1 − γ) × price[t] + γ × L0[t−1]
        L1[t] = −γ × L0[t] + L0[t−1] + γ × L1[t−1]
        L2[t] = −γ × L1[t] + L1[t−1] + γ × L2[t−1]
        L3[t] = −γ × L2[t] + L2[t−1] + γ × L3[t−1]
        filter[t] = (L0[t] + 2×L1[t] + 2×L2[t] + L3[t]) / 6

    No formal null-prefix: values are valid from bar 0, but the filter
    requires a handful of bars to converge from its zero initial state.

    Args:
        series: Input price series (e.g., close).
        gamma: Smoothing factor in the open interval (0, 1) (default 0.8).

    Returns:
        Series of Laguerre Filter values; all bars produce a value.

    Raises:
        ValueError: If ``gamma`` is not strictly in (0, 1).
    """
    if not (0.0 < gamma < 1.0):
        raise ValueError(f"Laguerre gamma must be in (0, 1), got {gamma}.")

    raw: list[float | None] = series.to_list()
    output: list[float | None] = [None] * len(raw)

    # Four-element state; initialised to zero (cold start).
    l0_prev = l1_prev = l2_prev = l3_prev = 0.0
    one_minus_g = 1.0 - gamma

    for idx, price in enumerate(raw):
        if price is None:
            # Reset state on missing data so the filter re-warms cleanly.
            l0_prev = l1_prev = l2_prev = l3_prev = 0.0
            continue

        # Each stage feeds into the next using the previous state.
        l0 = one_minus_g * price + gamma * l0_prev
        l1 = -gamma * l0 + l0_prev + gamma * l1_prev
        l2 = -gamma * l1 + l1_prev + gamma * l2_prev
        l3 = -gamma * l2 + l2_prev + gamma * l3_prev

        # Triangular weighted average; outer stages count double.
        output[idx] = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0
        l0_prev, l1_prev, l2_prev, l3_prev = l0, l1, l2, l3

    return pl.Series(f"laguerre_{gamma:.4g}", output, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# TRIMA
# ---------------------------------------------------------------------------


def trima(series: pl.Series, period: int) -> pl.Series:
    """Triangular Moving Average — double-smoothed SMA with triangular weights.

    TRIMA applies two sequential SMAs whose lengths sum to ``period + 1``,
    so the combined warm-up is exactly ``period − 1`` bars (the standard
    library null-prefix contract).  The triangular weighting approximates a
    Gaussian smoothing kernel: the centre of the window receives the highest
    effective weight, making TRIMA smoother than a single SMA of the same
    length but slower to react to price changes.

    Pass lengths (TA-Lib convention):
        first_len  = period // 2 + 1
        second_len = period − first_len + 1

    Algorithm:
        s1    = SMA(series, first_len)
        TRIMA = SMA(s1,     second_len)

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Input price series.
        period: Nominal lookback length (number of bars).

    Returns:
        Series of TRIMA values named ``trima_{period}``.
        The first ``period − 1`` values are ``null``.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - TA-Lib TRIMA: https://ta-lib.org/function.html#TRIMA
        - Investopedia — Triangular Moving Average:
          https://www.investopedia.com/terms/t/triangular-moving-average.asp
    """
    _validate_period(period, "TRIMA")
    # Two-pass window lengths that satisfy first_len + second_len = period + 1,
    # guaranteeing total warm-up of exactly period − 1 bars.
    first_len = period // 2 + 1
    second_len = period - first_len + 1
    s1 = series.rolling_mean(window_size=first_len, min_samples=first_len)
    return s1.rolling_mean(window_size=second_len, min_samples=second_len).alias(f"trima_{period}")


# ---------------------------------------------------------------------------
# VIDYA
# ---------------------------------------------------------------------------


def vidya(series: pl.Series, cmo_period: int = 9, alpha: float = 0.2) -> pl.Series:
    """Variable Index Dynamic Average (VIDYA).

    VIDYA is an adaptive EMA whose per-bar smoothing constant is scaled by a
    *Volatility Index* (VI) derived from the absolute Chande Momentum
    Oscillator (CMO).  In trending markets the absolute CMO is large and
    VIDYA tracks price quickly; in choppy markets the CMO is near zero and
    VIDYA barely moves.

    Algorithm:
        up[t]    = max(close[t] − close[t-1], 0)
        dn[t]    = max(close[t-1] − close[t], 0)
        sum_up   = rolling_sum(up, cmo_period)
        sum_dn   = rolling_sum(dn, cmo_period)
        VI[t]    = |sum_up − sum_dn| / (sum_up + sum_dn)   (range 0 to 1)
        sc[t]    = alpha × VI[t]
        VIDYA[t] = sc[t] × close[t] + (1 − sc[t]) × VIDYA[t-1]

    The indicator is seeded at the first bar where VI is defined
    (bar ``cmo_period``); the first ``cmo_period`` output values are ``null``.

    Args:
        series: Input price series (e.g., close).
        cmo_period: Lookback for the CMO-based volatility index (default 9).
        alpha: Base EMA smoothing factor multiplied by VI; must be in (0, 1]
               (default 0.2).

    Returns:
        Series of VIDYA values named ``vidya_{cmo_period}``.
        The first ``cmo_period`` values are ``null``.

    Raises:
        ValueError: If ``cmo_period < 1`` or ``alpha`` is outside (0, 1].

    References:
        - Chande, T. S. & Kroll, S. *The New Technical Trader* (1994), Chapter 3.
        - Investopedia — VIDYA:
          https://www.investopedia.com/terms/v/vidya.asp
    """
    _validate_period(cmo_period, "VIDYA")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"VIDYA alpha must be in (0, 1]; got {alpha}.")

    # Compute the Volatility Index (VI) via a CMO-derived ratio.
    # diff[0] is null (no predecessor), so up/dn retain that null, which
    # prevents the rolling sum from starting until cmo_period real changes exist.
    diff = series.diff(1)
    up = diff.clip(lower_bound=0.0)
    dn = (-diff).clip(lower_bound=0.0)
    sum_up = up.rolling_sum(window_size=cmo_period, min_samples=cmo_period)
    sum_dn = dn.rolling_sum(window_size=cmo_period, min_samples=cmo_period)
    total = sum_up + sum_dn
    # When total == 0 both sums are zero → 0/0 = NaN → fill to 0 (no movement, freeze).
    vi = ((sum_up - sum_dn).abs() / total).fill_nan(0.0)

    raw: list[float | None] = series.to_list()
    vi_list: list[float | None] = vi.to_list()
    n = len(raw)
    output: list[float | None] = [None] * n
    prev: float | None = None

    for i in range(n):
        if vi_list[i] is None:
            # Warm-up: VI not yet defined; leave output as null.
            continue
        sc = alpha * vi_list[i]  # type: ignore[operator]
        # Seed at first valid bar; thereafter apply the adaptive smoothing.
        prev = raw[i] if prev is None else sc * raw[i] + (1.0 - sc) * prev  # type: ignore[operator]
        output[i] = prev

    return pl.Series(f"vidya_{cmo_period}", output, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# Exponential Hull Moving Average (EHMA)
# ---------------------------------------------------------------------------


def ehma(series: pl.Series, period: int) -> pl.Series:
    """Exponential Hull Moving Average — EMA-based variant of the Hull MA.

    The EHMA substitutes Exponential Moving Averages for the Weighted Moving
    Averages used in the standard HMA.  Like HMA it applies the
    "2·fast − slow" de-lagging trick then smooths the result with a
    sqrt(period)-span EMA, yielding a low-lag, smooth curve.

    Algorithm:
        fast_period = period // 2
        sqrt_period = max(2, round(sqrt(period)))
        fast  = EMA(series, fast_period)
        slow  = EMA(series, period)
        raw   = 2 × fast − slow
        EHMA  = EMA(raw, sqrt_period)

    Null-prefix: ``period + sqrt_period − 2`` bars.
    - ``slow`` contributes ``period − 1`` nulls.
    - ``EMA(raw, sqrt_period)`` adds another ``sqrt_period − 1`` nulls.

    Args:
        series: Input price series (e.g., close).
        period: Lookback window.  Must be ≥ 2.

    Returns:
        Series of EHMA values named ``ehma_{period}``.

    Raises:
        ValueError: If ``period < 2``.

    References:
        - Based on Hull, A. "Hull Moving Average."
          https://alanhull.com/hull-moving-average
        - Investopedia — Hull Moving Average:
          https://www.investopedia.com/terms/h/hull-moving-average.asp
    """
    _validate_period(period, "EHMA", min_period=2)
    fast_period = period // 2
    sqrt_period = max(2, round(math.sqrt(period)))

    # Two EMA passes at different spans; then denoise with a third.
    fast_ema = ema(series, fast_period)
    slow_ema = ema(series, period)
    raw = 2.0 * fast_ema - slow_ema
    return ema(raw, sqrt_period).alias(f"ehma_{period}")


# ---------------------------------------------------------------------------
# Pascal's Weighted Moving Average (PWMA)
# ---------------------------------------------------------------------------


def pwma(series: pl.Series, period: int) -> pl.Series:
    """Pascal's Weighted Moving Average — binomial-coefficient weighting.

    PWMA weights each bar in the rolling window by the corresponding entry in
    row ``period − 1`` of Pascal's triangle (i.e. binomial coefficients
    C(period−1, i) for i = 0 … period−1, oldest to most recent).  The
    bell-shaped weight profile gives heavy emphasis to the centre of the
    window, producing a smooth output similar to a Gaussian filter.

    Algorithm:
        weights[i]  = C(period − 1, i)  for i = 0, 1, …, period − 1
        PWMA[t]     = sum(close[t−period+1+i] × weights[i]) / sum(weights)

    Null-prefix: ``period − 1`` bars.

    Args:
        series: Input price series (e.g., close).
        period: Lookback window (also the number of binomial weights).
                Must be ≥ 1.

    Returns:
        Series of PWMA values named ``pwma_{period}``.

    Raises:
        ValueError: If ``period < 1``.

    References:
        - Kaufman, P. J. *Trading Systems and Methods*, 5th ed. (2013),
          Chapter 2 (weighted averages overview).
        - Investopedia — Weighted Moving Average:
          https://www.investopedia.com/terms/w/weighted.asp
    """
    _validate_period(period, "PWMA")

    # Pre-compute binomial weights from row (period-1) of Pascal's triangle.
    weights = [math.comb(period - 1, i) for i in range(period)]
    total_w = float(sum(weights))

    def _pwma_window(w: pl.Series) -> float:
        """Apply binomial weights to a single window."""
        vals: list[float] = w.to_list()
        return sum(v * wt for v, wt in zip(vals, weights, strict=True)) / total_w

    return series.rolling_map(
        function=_pwma_window,
        window_size=period,
        min_samples=period,
    ).alias(f"pwma_{period}")


# ---------------------------------------------------------------------------
# MAMA / FAMA
# ---------------------------------------------------------------------------


def mama(
    series: pl.Series,
    fast_limit: float = 0.5,
    slow_limit: float = 0.05,
) -> pl.DataFrame:
    """MESA Adaptive Moving Average (MAMA) and Following Adaptive MA (FAMA).

    MAMA (Ehlers, 2004) adapts its smoothing constant in real time using the
    phase rate-of-change derived from the Hilbert Transform Homodyne
    Discriminator.  Fast phase rotation (trending markets) yields a large alpha
    (price tracking); slow phase rotation (choppy markets) yields a small alpha
    (noise filtering).  FAMA is a half-speed follower of MAMA that provides a
    signal/confirmation line; crossovers between MAMA and FAMA generate signals.

    Algorithm (Homodyne Discriminator pipeline):
        1. 4-bar weighted price smoothing  (weights 4,3,2,1 → divisor 10).
        2. Hilbert Transform detrender via a 6-tap kernel with adaptive gain
           coefficient ``(0.075 × prev_period + 0.54)``.
        3. In-phase component  I1[t] = detrender[t−3];
           Quadrature component Q1[t] from HT of detrender.
        4. 90-degree phase advance: jI, jQ via second HT pass on I1, Q1.
        5. Phasor addition: I2 = I1 − jQ; Q2 = Q1 + jI.
        6. EWM smoothing of I2, Q2 (α = 0.2).
        7. Homodyne discriminator:
               Re = I2[t]×I2[t−1] + Q2[t]×Q2[t−1]
               Im = I2[t]×Q2[t−1] − Q2[t]×I2[t−1]
           both further EWM-smoothed (α = 0.2).
        8. Dominant cycle: period = 2π / atan(Im/Re), clamped to ±50% change
           per bar and hard-clipped to [6, 50]; then EWM-smoothed (α = 0.2).
        9. Instantaneous phase = atan(Q1/I1) in degrees;
           delta_phase = max(1, phase[t−1] − phase[t]).
       10. Adaptive alpha = clamp(fast_limit / delta_phase, slow_limit, fast_limit).
       11. MAMA[t] = alpha × price[t] + (1 − alpha) × MAMA[t−1].
           FAMA[t] = 0.5×alpha × MAMA[t] + (1 − 0.5×alpha) × FAMA[t−1].

    Null-prefix: 6 bars (minimum for the HT kernel to have non-trivial inputs).
    Statistical convergence typically requires ~32 bars.

    Args:
        series: Input price series (typically ``close`` or HL2).
        fast_limit: Maximum adaptive smoothing constant (upper bound on alpha).
                    Default 0.5.
        slow_limit: Minimum adaptive smoothing constant (lower bound on alpha).
                    Default 0.05.

    Returns:
        DataFrame with columns ``mama`` and ``fama``.
        The first 6 rows are ``null``.

    Raises:
        ValueError: If the relationship ``0 < slow_limit < fast_limit ≤ 1``
                    is not satisfied.

    References:
        - Ehlers, J. F. *Cybernetic Analysis for Stocks and Futures* (2004),
          Chapter 3.
        - Ehlers, J. F. "MESA and Trading Market Cycles" (1992).
    """
    if not (0.0 < slow_limit < fast_limit <= 1.0):
        raise ValueError(
            f"mama: requires 0 < slow_limit ({slow_limit}) < fast_limit ({fast_limit}) ≤ 1."
        )

    raw: list[float | None] = series.to_list()
    n = len(raw)

    # State arrays — all zero-initialised; warm-up values have negligible effect
    # once the algorithm converges after ~32 bars.
    smooth: list[float] = [0.0] * n
    detrender: list[float] = [0.0] * n
    q1: list[float] = [0.0] * n
    i1: list[float] = [0.0] * n
    ji: list[float] = [0.0] * n
    jq: list[float] = [0.0] * n
    i2: list[float] = [0.0] * n
    q2: list[float] = [0.0] * n
    re: list[float] = [0.0] * n
    im: list[float] = [0.0] * n
    period: list[float] = [0.0] * n
    phase: list[float] = [0.0] * n
    mama_arr: list[float] = [0.0] * n
    fama_arr: list[float] = [0.0] * n

    mama_out: list[float | None] = [None] * n
    fama_out: list[float | None] = [None] * n

    for t in range(n):
        price = raw[t]
        if price is None:
            continue

        # 4-bar weighted price smoothing; missing prior bars fall back to price.
        p1 = raw[t - 1] if t >= 1 and raw[t - 1] is not None else price
        p2 = raw[t - 2] if t >= 2 and raw[t - 2] is not None else price
        p3 = raw[t - 3] if t >= 3 and raw[t - 3] is not None else price
        smooth[t] = (4.0 * price + 3.0 * p1 + 2.0 * p2 + p3) / 10.0

        # Seed MAMA/FAMA with the raw price during the HT warm-up window.
        if t < 6:
            mama_arr[t] = price
            fama_arr[t] = price
            continue

        # Adaptive gain coefficient tied to the prior bar's cycle estimate.
        prev_period = period[t - 1] if period[t - 1] > 0.0 else 6.0
        coeff = 0.075 * prev_period + 0.54

        # Hilbert Transform detrender (6-tap kernel with adaptive gain).
        detrender[t] = (
            0.0962 * smooth[t]
            + 0.5769 * smooth[t - 2]
            - 0.5769 * smooth[t - 4]
            - 0.0962 * smooth[t - 6]
        ) * coeff

        # Quadrature (Q1) via HT of detrender; in-phase (I1) delayed 3 bars.
        q1[t] = (
            0.0962 * detrender[t]
            + 0.5769 * detrender[t - 2]
            - 0.5769 * detrender[t - 4]
            - 0.0962 * detrender[t - 6]
        ) * coeff
        i1[t] = detrender[t - 3]

        # 90-degree phase advance: apply HT to I1 and Q1 independently.
        ji[t] = (
            0.0962 * i1[t] + 0.5769 * i1[t - 2] - 0.5769 * i1[t - 4] - 0.0962 * i1[t - 6]
        ) * coeff
        jq[t] = (
            0.0962 * q1[t] + 0.5769 * q1[t - 2] - 0.5769 * q1[t - 4] - 0.0962 * q1[t - 6]
        ) * coeff

        # Phasor addition achieves 3-bar averaging of the cycle components.
        i2_raw = i1[t] - jq[t]
        q2_raw = q1[t] + ji[t]
        # EWM smoothing (α=0.2) stabilises the phasor before discriminator.
        i2[t] = 0.2 * i2_raw + 0.8 * i2[t - 1]
        q2[t] = 0.2 * q2_raw + 0.8 * q2[t - 1]

        # Homodyne discriminator: cross-multiply current with prior phasor.
        re_raw = i2[t] * i2[t - 1] + q2[t] * q2[t - 1]
        im_raw = i2[t] * q2[t - 1] - q2[t] * i2[t - 1]
        re[t] = 0.2 * re_raw + 0.8 * re[t - 1]
        im[t] = 0.2 * im_raw + 0.8 * im[t - 1]

        # Dominant cycle period from the discriminator phase angle.
        if im[t] != 0.0 and re[t] != 0.0:
            raw_period = 2.0 * math.pi / math.atan(im[t] / re[t])
        else:
            raw_period = prev_period
        # Rate-of-change clamp: period may only change by ±50% per bar.
        raw_period = max(0.67 * prev_period, min(1.5 * prev_period, raw_period))
        # Hard clip to the physically meaningful range [6, 50] bars.
        raw_period = max(6.0, min(50.0, raw_period))
        period[t] = 0.2 * raw_period + 0.8 * prev_period

        # Instantaneous phase angle in degrees (arctangent of Q/I ratio).
        if i1[t] != 0.0:
            phase[t] = math.degrees(math.atan(q1[t] / i1[t]))
        else:
            phase[t] = phase[t - 1]
        # Phase change drives alpha; clamp to ≥1 to avoid division by zero.
        delta_phase = max(1.0, phase[t - 1] - phase[t])

        # Adaptive smoothing constant bounded between the user limits.
        alpha = max(slow_limit, min(fast_limit, fast_limit / delta_phase))

        mama_arr[t] = alpha * price + (1.0 - alpha) * mama_arr[t - 1]
        fama_arr[t] = 0.5 * alpha * mama_arr[t] + (1.0 - 0.5 * alpha) * fama_arr[t - 1]

        mama_out[t] = mama_arr[t]
        fama_out[t] = fama_arr[t]

    return pl.DataFrame(
        {
            "mama": pl.Series("mama", mama_out, dtype=pl.Float64),
            "fama": pl.Series("fama", fama_out, dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Hilbert Transform Dominant Cycle Period
# ---------------------------------------------------------------------------


def dominant_cycle_period(series: pl.Series) -> pl.Series:
    """Hilbert Transform Dominant Cycle Period.

    Estimates the instantaneous dominant cycle period using Ehlers' Homodyne
    Discriminator — the same mechanism that drives MAMA's adaptive smoothing.
    The output is useful for dynamically sizing other indicators (e.g., setting
    an RSI or stochastic period to half the dominant cycle).

    Algorithm: identical Hilbert Transform pipeline as ``mama`` (smooth →
    detrender → I1/Q1 → jI/jQ → I2/Q2 → Re/Im → period).  Returns the
    EWM-smoothed dominant cycle period in bars.

    Null-prefix: 6 bars (minimum for the HT kernel).  Convergence ≈ 32 bars;
    values in bars 6–31 reflect a warming-up state.

    Args:
        series: Input price series (typically ``close`` or HL2).

    Returns:
        Series named ``dominant_cycle`` with period estimates in bars
        (range approximately 6–50 after convergence).
        The first 6 values are ``null``.

    References:
        - Ehlers, J. F. *Cybernetic Analysis for Stocks and Futures* (2004),
          Chapters 2–3.
    """
    raw: list[float | None] = series.to_list()
    n = len(raw)

    smooth: list[float] = [0.0] * n
    detrender: list[float] = [0.0] * n
    q1: list[float] = [0.0] * n
    i1: list[float] = [0.0] * n
    ji: list[float] = [0.0] * n
    jq: list[float] = [0.0] * n
    i2: list[float] = [0.0] * n
    q2: list[float] = [0.0] * n
    re: list[float] = [0.0] * n
    im: list[float] = [0.0] * n
    period: list[float] = [0.0] * n

    output: list[float | None] = [None] * n

    for t in range(n):
        price = raw[t]
        if price is None:
            continue

        # 4-bar weighted smoothing with fallback for bars near the start.
        p1 = raw[t - 1] if t >= 1 and raw[t - 1] is not None else price
        p2 = raw[t - 2] if t >= 2 and raw[t - 2] is not None else price
        p3 = raw[t - 3] if t >= 3 and raw[t - 3] is not None else price
        smooth[t] = (4.0 * price + 3.0 * p1 + 2.0 * p2 + p3) / 10.0

        if t < 6:
            continue  # HT kernel requires at least 6 prior smooth values

        prev_period = period[t - 1] if period[t - 1] > 0.0 else 6.0
        coeff = 0.075 * prev_period + 0.54

        detrender[t] = (
            0.0962 * smooth[t]
            + 0.5769 * smooth[t - 2]
            - 0.5769 * smooth[t - 4]
            - 0.0962 * smooth[t - 6]
        ) * coeff

        q1[t] = (
            0.0962 * detrender[t]
            + 0.5769 * detrender[t - 2]
            - 0.5769 * detrender[t - 4]
            - 0.0962 * detrender[t - 6]
        ) * coeff
        i1[t] = detrender[t - 3]

        ji[t] = (
            0.0962 * i1[t] + 0.5769 * i1[t - 2] - 0.5769 * i1[t - 4] - 0.0962 * i1[t - 6]
        ) * coeff
        jq[t] = (
            0.0962 * q1[t] + 0.5769 * q1[t - 2] - 0.5769 * q1[t - 4] - 0.0962 * q1[t - 6]
        ) * coeff

        i2_raw = i1[t] - jq[t]
        q2_raw = q1[t] + ji[t]
        i2[t] = 0.2 * i2_raw + 0.8 * i2[t - 1]
        q2[t] = 0.2 * q2_raw + 0.8 * q2[t - 1]

        re_raw = i2[t] * i2[t - 1] + q2[t] * q2[t - 1]
        im_raw = i2[t] * q2[t - 1] - q2[t] * i2[t - 1]
        re[t] = 0.2 * re_raw + 0.8 * re[t - 1]
        im[t] = 0.2 * im_raw + 0.8 * im[t - 1]

        if im[t] != 0.0 and re[t] != 0.0:
            raw_period = 2.0 * math.pi / math.atan(im[t] / re[t])
        else:
            raw_period = prev_period
        raw_period = max(0.67 * prev_period, min(1.5 * prev_period, raw_period))
        raw_period = max(6.0, min(50.0, raw_period))
        period[t] = 0.2 * raw_period + 0.8 * prev_period

        output[t] = period[t]

    return pl.Series("dominant_cycle", output, dtype=pl.Float64)
