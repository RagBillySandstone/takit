"""
Volatility indicators.

All multi-input functions accept a ``pl.DataFrame`` with generic OHLC
column names (``open``, ``high``, ``low``, ``close``).  Single-series
functions accept a ``pl.Series``.

Functions
---------
true_range            Single-bar True Range (prerequisite for ATR)
atr                   Average True Range (Wilder smoothing, default period 14)
natr                  Normalised ATR — ATR as a percentage of close
bollinger_bands       Bollinger Bands: middle, upper, lower, %B, bandwidth
keltner_channels      Keltner Channels: middle (EMA), upper, lower
chaikin_volatility    Rate of change of EMA(H-L range)
historical_volatility Rolling annualised standard deviation of log returns
ulcer_index           Drawdown-based volatility measure
chandelier_exit       ATR-based dynamic trailing-stop levels
mass_index            Reversal detector via High-Low range expansion (Dorsey)
parkinson             High-Low range-based volatility estimator (Parkinson 1980)
garman_klass          OHLC volatility estimator accounting for open-close drift
yang_zhang            Overnight-adjusted OHLC volatility estimator (Yang-Zhang 2000)
williams_vix_fix      Synthetic fear gauge: rolling-high distance from low
choppiness_index      Choppiness Index — trending vs. choppy regime quantifier
squeeze_momentum      TTM Squeeze — BB/KC compression detector with momentum histogram
volatility_ratio      Volatility Ratio — current True Range vs. its n-period maximum
"""

from __future__ import annotations

import math

import polars as pl

from polarticks._validate import _validate_period
from polarticks.moving_averages import ema, wilder_smooth

# ---------------------------------------------------------------------------
# True Range
# ---------------------------------------------------------------------------


def true_range(ohlc: pl.DataFrame) -> pl.Series:
    """Compute the True Range for each bar.

    True Range accounts for overnight gaps and is defined as the greatest of:
        - high − low
        - |high − previous close|
        - |low  − previous close|

    The first bar has no previous close; the gap-aware terms are set to zero
    there, so True Range collapses to ``high − low`` on bar 0 — matching
    Wilder's convention.

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.

    Returns:
        Series of True Range values (all non-negative, first value is ``high - low``).
    """
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]
    prev_close = close.shift(1)

    return (
        pl.DataFrame(
            {
                "hl": (high - low).abs(),
                "hc": (high - prev_close).abs().fill_null(0.0),
                "lc": (low - prev_close).abs().fill_null(0.0),
            }
        )
        .select(pl.max_horizontal("hl", "hc", "lc").alias("true_range"))
        .to_series()
    )


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def atr(ohlc: pl.DataFrame, period: int = 14) -> pl.Series:
    """Average True Range using Wilder's smoothing.

    ATR is the Wilder-smoothed average of the True Range.  It measures
    market volatility without regard to price direction.  The first
    *period* values are ``null``.

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: Wilder smoothing period (default 14).

    Returns:
        Series of ATR values.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "ATR")
    tr = true_range(ohlc)
    return wilder_smooth(tr, period).alias(f"atr_{period}")


# ---------------------------------------------------------------------------
# NATR
# ---------------------------------------------------------------------------


def natr(ohlc: pl.DataFrame, period: int = 14) -> pl.Series:
    """Normalised Average True Range — ATR expressed as a percentage of close.

    NATR scales ATR by the current closing price, making volatility directly
    comparable across instruments at different price levels.  A NATR of 2.0
    means the average true range is 2% of the close.

        NATR = 100 × ATR(period) / close

    The first ``period − 1`` values are ``null`` (inherited from ATR).

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: ATR lookback period (default 14).

    Returns:
        Series of NATR values (percentage, non-negative).

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "NATR")

    atr_values = atr(ohlc, period)
    close = ohlc["close"]

    # fill_nan guards against the theoretical zero-close edge case.
    return (100.0 * atr_values / close).fill_nan(None).alias(f"natr_{period}")


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def bollinger_bands(
    series: pl.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pl.DataFrame:
    """Bollinger Bands: middle band (SMA), upper and lower bands, %B, bandwidth.

    Bands widen during volatile markets and contract during quiet periods.
    %B indicates where price sits within the bands (0 = lower, 1 = upper).
    Bandwidth is a normalised measure of volatility: (upper − lower) / middle.

    Args:
        series: Close price series.
        period: SMA period and rolling std window (default 20).
        num_std: Number of standard deviations for band width (default 2.0).

    Returns:
        DataFrame with columns:
            ``bb_middle_{period}``, ``bb_upper_{period}``, ``bb_lower_{period}``,
            ``bb_pct_b_{period}``, ``bb_width_{period}``.

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "Bollinger Bands", min_period=2)

    middle = series.rolling_mean(window_size=period, min_samples=period)
    std = series.rolling_std(window_size=period, min_samples=period)

    upper = middle + num_std * std
    lower = middle - num_std * std

    band_range = upper - lower
    # pct_b = (price − lower) / (upper − lower); default to 0.5 in a flat market.
    pct_b = ((series - lower) / band_range).fill_nan(0.5)
    # bandwidth is safe to divide: middle > 0 for all real price data.
    bandwidth = band_range / middle

    return pl.DataFrame(
        {
            f"bb_middle_{period}": middle,
            f"bb_upper_{period}": upper,
            f"bb_lower_{period}": lower,
            f"bb_pct_b_{period}": pct_b,
            f"bb_width_{period}": bandwidth,
        }
    )


# ---------------------------------------------------------------------------
# Keltner Channels
# ---------------------------------------------------------------------------


def keltner_channels(
    ohlc: pl.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pl.DataFrame:
    """Keltner Channels: EMA-based bands using ATR for width.

    Unlike Bollinger Bands (which use standard deviation), Keltner Channels
    use ATR for band width, making them less reactive to single large moves.
    When price breaks outside the Keltner Channels it signals a strong trend.

        middle = EMA(close, ema_period)
        upper  = middle + multiplier × ATR(atr_period)
        lower  = middle − multiplier × ATR(atr_period)

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        ema_period: EMA period for the middle band (default 20).
        atr_period: ATR period for channel width (default 10).
        multiplier: ATR multiplier for upper/lower bands (default 2.0).

    Returns:
        DataFrame with columns ``kc_middle``, ``kc_upper``, ``kc_lower``.

    Raises:
        ValueError: If ``ema_period < 1`` or ``atr_period < 1``.
    """
    _validate_period(ema_period, "Keltner EMA")
    _validate_period(atr_period, "Keltner ATR")

    middle = ema(ohlc["close"], ema_period).alias("kc_middle")
    atr_values = atr(ohlc, atr_period)

    upper = (middle + multiplier * atr_values).alias("kc_upper")
    lower = (middle - multiplier * atr_values).alias("kc_lower")

    return pl.DataFrame({"kc_middle": middle, "kc_upper": upper, "kc_lower": lower})


# ---------------------------------------------------------------------------
# Chaikin Volatility
# ---------------------------------------------------------------------------


def chaikin_volatility(
    ohlc: pl.DataFrame,
    ema_period: int = 10,
    roc_period: int = 10,
) -> pl.Series:
    """Chaikin Volatility — rate of change of the EMA of the high-low range.

    Measures the rate at which the smoothed trading range expands or contracts.
    Rising values indicate increasing volatility; falling values suggest
    diminishing volatility and potential consolidation.

        ema_range = EMA(high − low, ema_period)
        CV        = 100 × (ema_range[t] − ema_range[t − roc_period]) / ema_range[t − roc_period]

    Args:
        ohlc: DataFrame with columns ``high`` and ``low``.
        ema_period: EMA smoothing period for the H-L range (default 10).
        roc_period: Look-back for the rate-of-change calculation (default 10).

    Returns:
        Series of Chaikin Volatility values (percentage).

    Raises:
        ValueError: If ``ema_period < 1`` or ``roc_period < 1``.
    """
    _validate_period(ema_period, "Chaikin Volatility EMA")
    _validate_period(roc_period, "Chaikin Volatility ROC")

    hl_ema = ema(ohlc["high"] - ohlc["low"], ema_period)
    past_ema = hl_ema.shift(roc_period)

    # fill_nan converts inf/nan (zero past EMA, e.g. flat zero-range window)
    # to null rather than silently propagating an undefined value.
    return (
        (100.0 * (hl_ema - past_ema) / past_ema)
        .fill_nan(None)
        .alias(f"chaikin_vol_{ema_period}_{roc_period}")
    )


# ---------------------------------------------------------------------------
# Historical Volatility
# ---------------------------------------------------------------------------


def historical_volatility(
    series: pl.Series,
    period: int = 20,
    annualise: bool = True,
    trading_days: int = 252,
) -> pl.Series:
    """Historical Volatility — rolling annualised standard deviation of log returns.

    Measures the realised volatility of a price series over a rolling window.
    The result is expressed as an annualised percentage when ``annualise=True``.

        log_return[t] = ln(price[t] / price[t-1])
        HV            = std(log_return, period) × √trading_days   [if annualise]

    Args:
        series: Close price series (or any price series).
        period: Rolling window length (default 20).
        annualise: If ``True``, scale by ``√trading_days`` (default ``True``).
        trading_days: Annualisation factor (default 252 for equities; use 365
                      for crypto or 260 for FX).

    Returns:
        Series of Historical Volatility values.  Annualised values are
        typically in the range [0, 2] (0%–200%).

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "Historical Volatility", min_period=2)

    log_ret = (series / series.shift(1)).log(base=math.e)
    hv = log_ret.rolling_std(window_size=period, min_samples=period)

    if annualise:
        hv = hv * math.sqrt(trading_days)

    return hv.alias(f"hv_{period}")


# ---------------------------------------------------------------------------
# Ulcer Index
# ---------------------------------------------------------------------------


def ulcer_index(series: pl.Series, period: int = 14) -> pl.Series:
    """Ulcer Index — drawdown-based volatility measure.

    Unlike standard deviation, the Ulcer Index only penalises downside
    moves (drawdowns from the rolling maximum).  It is useful for
    risk-adjusted metrics such as the Ulcer Performance Index (UPI).

        rolling_max[t] = max(price, period)
        pct_drawdown   = 100 × (price − rolling_max) / rolling_max
        UI             = √(mean(pct_drawdown², period))

    Low UI values indicate smooth, upward price action; high values
    indicate deep or prolonged drawdowns.

    Args:
        series: Close price series.
        period: Lookback window for both rolling max and the squaring average
                (default 14).

    Returns:
        Series of Ulcer Index values (non-negative).

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Ulcer Index")

    rolling_max = series.rolling_max(window_size=period, min_samples=period)

    # Drawdown is zero or negative; squaring removes the sign.
    pct_drawdown = 100.0 * (series - rolling_max) / rolling_max
    squared_drawdown = pct_drawdown**2

    return (squared_drawdown.rolling_mean(window_size=period, min_samples=period) ** 0.5).alias(
        f"ulcer_{period}"
    )


# ---------------------------------------------------------------------------
# Chandelier Exit
# ---------------------------------------------------------------------------


def chandelier_exit(
    ohlc: pl.DataFrame,
    period: int = 22,
    multiplier: float = 3.0,
) -> pl.DataFrame:
    """Chandelier Exit — ATR-based dynamic trailing-stop levels.

    The Chandelier Exit defines two trailing-stop lines: one for long
    positions and one for short positions.  When price falls below the long
    exit (or rises above the short exit) it signals a potential trend
    reversal.  Developed by Charles Le Beau.

    Algorithm:
        long_exit[t]  = highest_high(period)[t] − multiplier × ATR(period)[t]
        short_exit[t] = lowest_low(period)[t]  + multiplier × ATR(period)[t]

    The first ``period - 1`` values are ``null`` (ATR warm-up).

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: Lookback window for the highest-high, lowest-low, and ATR
            (default 22).
        multiplier: ATR multiplier that controls how tight the stops are
            (default 3.0).

    Returns:
        DataFrame with columns:
            ``ce_long_{period}``  — long-position trailing stop level,
            ``ce_short_{period}`` — short-position trailing stop level.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Chandelier Exit")

    highest_high = ohlc["high"].rolling_max(window_size=period, min_samples=period)
    lowest_low = ohlc["low"].rolling_min(window_size=period, min_samples=period)
    atr_values = atr(ohlc, period)

    long_exit = highest_high - multiplier * atr_values
    short_exit = lowest_low + multiplier * atr_values

    return pl.DataFrame(
        {
            f"ce_long_{period}": long_exit,
            f"ce_short_{period}": short_exit,
        }
    )


# ---------------------------------------------------------------------------
# Mass Index
# ---------------------------------------------------------------------------


def mass_index(
    ohlc: pl.DataFrame,
    ema_period: int = 9,
    sum_period: int = 25,
) -> pl.Series:
    """Mass Index — detects trend reversals via High-Low range expansion.

    The Mass Index (Donald Dorsey, 1992) compares a double-smoothed EMA of
    the daily price range to its single-smoothed counterpart.  When the 25-bar
    rolling sum of the ratio forms a "reversal bulge" (rises above 27 then
    falls back below 26.5), a trend reversal is signalled regardless of
    direction.

    Algorithm:
        single_ema = EMA(high − low, ema_period)
        double_ema = EMA(single_ema, ema_period)
        mass       = Σ(single_ema / double_ema, sum_period)

    Null-prefix: ``2 × (ema_period − 1) + (sum_period − 1)`` bars.

    Args:
        ohlc: DataFrame with columns ``high`` and ``low``.
        ema_period: EMA period applied to the range for both passes (default 9).
        sum_period: Rolling sum accumulation period (default 25).

    Returns:
        Series named ``mass_index``.

    Raises:
        ValueError: If any period < 1.
    """
    _validate_period(ema_period, "Mass Index ema_period")
    _validate_period(sum_period, "Mass Index sum_period")

    hl_range = ohlc["high"] - ohlc["low"]

    # Two successive EMA passes; double_ema has 2×(ema_period−1) leading nulls.
    single_ema = ema(hl_range, ema_period)
    double_ema = ema(single_ema, ema_period)

    # fill_nan handles the rare edge case of a perfectly flat price range.
    ratio = (single_ema / double_ema).fill_nan(None)

    return ratio.rolling_sum(window_size=sum_period, min_samples=sum_period).alias("mass_index")


# ---------------------------------------------------------------------------
# Parkinson Volatility
# ---------------------------------------------------------------------------


def parkinson(
    ohlc: pl.DataFrame,
    period: int = 20,
    annualise: bool = True,
    trading_days: int = 252,
) -> pl.Series:
    """Parkinson Volatility — high-low range-based volatility estimator.

    The Parkinson (1980) estimator is more efficient than the close-to-close
    historical volatility because it uses intrabar high and low information.
    It does not account for overnight gaps or drift.

    Algorithm:
        log_hl[t]  = ln(high[t] / low[t])
        PV         = sqrt(mean(log_hl², period) / (4 · ln 2)) × √trading_days

    Null-prefix: ``period − 1`` bars.

    Args:
        ohlc: DataFrame with columns ``high`` and ``low``.
        period: Rolling window length (default 20).
        annualise: If ``True``, scale by ``√trading_days`` (default ``True``).
        trading_days: Annualisation factor (default 252).

    Returns:
        Series named ``parkinson_{period}``.

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "Parkinson", min_period=2)

    log_hl = (ohlc["high"] / ohlc["low"]).log(base=math.e)

    # 1 / (4·ln2) ≈ 0.3607 — the normalisation factor from Parkinson (1980).
    factor = 1.0 / (4.0 * math.log(2.0))
    variance = factor * (log_hl**2).rolling_mean(window_size=period, min_samples=period)

    hv = variance.clip(lower_bound=0.0) ** 0.5
    if annualise:
        hv = hv * math.sqrt(trading_days)

    return hv.alias(f"parkinson_{period}")


# ---------------------------------------------------------------------------
# Garman-Klass Volatility
# ---------------------------------------------------------------------------


def garman_klass(
    ohlc: pl.DataFrame,
    period: int = 20,
    annualise: bool = True,
    trading_days: int = 252,
) -> pl.Series:
    """Garman-Klass Volatility — OHLC estimator accounting for open-close drift.

    The Garman-Klass (1980) estimator improves on Parkinson by incorporating
    the open-to-close return as a correction for drift.  It assumes no
    overnight gaps (open = prior close in the original derivation).

    Algorithm:
        GK[t] = 0.5 · ln(H/L)² − (2·ln2 − 1) · ln(C/O)²
        GKV   = sqrt(mean(GK, period)) × √trading_days

    Null-prefix: ``period − 1`` bars.

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        period: Rolling window length (default 20).
        annualise: If ``True``, scale by ``√trading_days`` (default ``True``).
        trading_days: Annualisation factor (default 252).

    Returns:
        Series named ``garman_klass_{period}``.

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "Garman-Klass", min_period=2)

    log_hl = (ohlc["high"] / ohlc["low"]).log(base=math.e)
    log_co = (ohlc["close"] / ohlc["open"]).log(base=math.e)

    # GK per-bar term: range component minus drift correction.
    gk_term = 0.5 * log_hl**2 - (2.0 * math.log(2.0) - 1.0) * log_co**2

    variance = gk_term.rolling_mean(window_size=period, min_samples=period)

    hv = variance.clip(lower_bound=0.0) ** 0.5
    if annualise:
        hv = hv * math.sqrt(trading_days)

    return hv.alias(f"garman_klass_{period}")


# ---------------------------------------------------------------------------
# Yang-Zhang Volatility
# ---------------------------------------------------------------------------


def yang_zhang(
    ohlc: pl.DataFrame,
    period: int = 20,
    annualise: bool = True,
    trading_days: int = 252,
) -> pl.Series:
    """Yang-Zhang Volatility — overnight-adjusted OHLC volatility estimator.

    The Yang-Zhang (2000) estimator is the most efficient unbiased estimator
    for close-to-close volatility that accounts for overnight gaps, open
    jumps, and intrabar drift.  It blends three components:

        σ²_overnight — sample variance of ln(Open / PrevClose) over the window
        σ²_oc        — sample variance of ln(Close / Open) over the window
        σ²_RS        — Rogers-Satchell rolling mean: Σ(ln(H/O)·ln(H/C) + ln(L/O)·ln(L/C))
        k            = 0.34 / (1.34 + (n+1) / (n−1))
        σ²_YZ        = σ²_overnight + k·σ²_oc + (1−k)·σ²_RS

    Null-prefix: ``period − 1`` bars (plus 1 extra on the overnight term from the
    shift; the rolling_std absorbs this within its own warm-up).

    Args:
        ohlc: DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        period: Rolling window length (default 20).
        annualise: If ``True``, scale by ``√trading_days`` (default ``True``).
        trading_days: Annualisation factor (default 252).

    Returns:
        Series named ``yang_zhang_{period}``.

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "Yang-Zhang", min_period=2)

    open_ = ohlc["open"]
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]
    prev_close = close.shift(1)

    # Overnight return: open vs prior close.
    log_overnight = (open_ / prev_close).log(base=math.e)
    # Open-to-close return: intraday component.
    log_oc = (close / open_).log(base=math.e)

    # Rogers-Satchell per-bar term (zero-mean estimator of intrabar variance).
    rs = (high / open_).log(base=math.e) * (high / close).log(base=math.e) + (low / open_).log(
        base=math.e
    ) * (low / close).log(base=math.e)

    # Rolling sample variances (ddof=1 via rolling_std).
    var_overnight = log_overnight.rolling_std(window_size=period, min_samples=period) ** 2
    var_oc = log_oc.rolling_std(window_size=period, min_samples=period) ** 2
    # Rogers-Satchell is already zero-mean, so its rolling mean = population variance.
    var_rs = rs.rolling_mean(window_size=period, min_samples=period)

    # Optimal blend weight derived in Yang-Zhang (2000).
    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    variance = var_overnight + k * var_oc + (1.0 - k) * var_rs

    hv = variance.clip(lower_bound=0.0) ** 0.5
    if annualise:
        hv = hv * math.sqrt(trading_days)

    return hv.alias(f"yang_zhang_{period}")


# ---------------------------------------------------------------------------
# Williams VIX Fix
# ---------------------------------------------------------------------------


def williams_vix_fix(ohlc: pl.DataFrame, period: int = 22) -> pl.Series:
    """Williams VIX Fix — synthetic fear gauge based on close distance from rolling high.

    The Williams VIX Fix (Larry Williams) mimics the shape of the CBOE VIX
    using price data alone.  It spikes when the current low is far below the
    recent rolling maximum of close prices, signalling fear or panic.

    Algorithm:
        highest_close[t] = max(close, period)
        WVF[t]           = 100 × (highest_close[t] − low[t]) / highest_close[t]

    Null-prefix: ``period − 1`` bars.

    Args:
        ohlc: DataFrame with columns ``low`` and ``close``.
        period: Rolling window for the highest close (default 22).

    Returns:
        Series named ``wvf_{period}``, values in [0, 100].

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Williams VIX Fix")

    close = ohlc["close"]
    low = ohlc["low"]

    highest_close = close.rolling_max(window_size=period, min_samples=period)

    # fill_nan guards against a theoretical zero highest-close edge case.
    return (100.0 * (highest_close - low) / highest_close).fill_nan(None).alias(f"wvf_{period}")


# ---------------------------------------------------------------------------
# Choppiness Index
# ---------------------------------------------------------------------------


def choppiness_index(ohlc: pl.DataFrame, period: int = 14) -> pl.Series:
    """Choppiness Index — quantifies trending versus choppy market conditions.

    CHOP (Dreiss) measures whether the market is trending or oscillating.
    Values approaching 100 indicate maximum choppiness (sideways/consolidating);
    values approaching the theoretical lower bound indicate a strong trend.
    The conventional thresholds are: above 61.8 (choppy), below 38.2 (trending).

    Algorithm:
        CHOP = 100 × log10(Σ TR(1), period) / (highest_high − lowest_low))
                   / log10(period)

    Null-prefix: ``period − 1`` bars.

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: Lookback window (default 14; must be ≥ 2).

    Returns:
        Series named ``chop_{period}``.

    Raises:
        ValueError: If ``period < 2``.
    """
    _validate_period(period, "Choppiness Index", min_period=2)

    # True range of each individual bar (ATR period = 1).
    tr = true_range(ohlc)
    atr_sum = tr.rolling_sum(window_size=period, min_samples=period)

    highest_high = ohlc["high"].rolling_max(window_size=period, min_samples=period)
    lowest_low = ohlc["low"].rolling_min(window_size=period, min_samples=period)
    hl_range = highest_high - lowest_low

    # fill_nan handles perfectly flat markets where the range is zero.
    ratio = (atr_sum / hl_range).fill_nan(1.0)
    log_period = math.log10(period)

    # log10(ratio) via change-of-base: log(x)/log(10).
    return (100.0 * ratio.log(base=math.e) / (log_period * math.log(10))).alias(f"chop_{period}")


# ---------------------------------------------------------------------------
# Squeeze Momentum
# ---------------------------------------------------------------------------


def squeeze_momentum(
    ohlc: pl.DataFrame,
    length: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
) -> pl.DataFrame:
    """TTM Squeeze — Bollinger/Keltner compression detector with momentum histogram.

    The Squeeze Momentum Indicator (Carter / Lazybear TTM Squeeze) detects
    periods of market consolidation ("squeeze") by checking when Bollinger
    Bands are entirely inside Keltner Channels.  The subsequent breakout
    direction is forecast by a linear-regression momentum histogram.

    Algorithm:
        BB:     ±bb_mult × std around SMA(close, length)
        KC:     ±kc_mult × ATR(length) around EMA(close, length)
        sqz_on  = (bb_upper < kc_upper) AND (bb_lower > kc_lower)
        sqz_off = (bb_upper ≥ kc_upper) AND (bb_lower ≤ kc_lower)
        mid     = (rolling_max(high, length) + rolling_min(low, length)) / 2
        delta   = close − (mid + SMA(close, length)) / 2
        momentum = linreg_value(delta, length)  [value at end of rolling window]

    Null-prefix: ``length − 1`` bars.

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        length: Lookback period for BB, KC, and linreg (default 20).
        bb_mult: BB standard-deviation multiplier (default 2.0).
        kc_mult: KC ATR multiplier (default 1.5).

    Returns:
        DataFrame with columns ``sqz_on`` (bool), ``sqz_off`` (bool),
        ``sqz_momentum`` (float).

    Raises:
        ValueError: If ``length < 2``.
    """
    _validate_period(length, "Squeeze Momentum", min_period=2)

    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]

    # Bollinger Bands around SMA.
    mid_sma = close.rolling_mean(window_size=length, min_samples=length)
    std = close.rolling_std(window_size=length, min_samples=length)
    bb_upper = mid_sma + bb_mult * std
    bb_lower = mid_sma - bb_mult * std

    # Keltner Channels around EMA using ATR for width.
    kc_mid = ema(close, length)
    kc_range = atr(ohlc, length) * kc_mult
    kc_upper = kc_mid + kc_range
    kc_lower = kc_mid - kc_range

    # Squeeze flags.
    sqz_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    sqz_off = (bb_upper >= kc_upper) & (bb_lower <= kc_lower)

    # Momentum: linreg value of delta from bar midpoint.
    highest = high.rolling_max(window_size=length, min_samples=length)
    lowest = low.rolling_min(window_size=length, min_samples=length)
    delta = close - (highest + lowest) / 2.0 - mid_sma

    # Linear regression value at the end of each window via rolling_map.
    def _lrv(w: pl.Series) -> float:
        """Fit OLS to the window and return the predicted value at the last bar."""
        n = len(w)
        vals = w.to_list()
        mean_y = sum(vals) / n
        mean_x = (n - 1) / 2.0
        ss_xx = sum((i - mean_x) ** 2 for i in range(n))
        if ss_xx == 0.0:
            return mean_y
        slope = sum((i - mean_x) * (vals[i] - mean_y) for i in range(n)) / ss_xx
        # Predicted value at x = n-1 = mean_y + slope*(n-1)/2.
        return mean_y + slope * (n - 1 - mean_x)

    momentum = delta.rolling_map(function=_lrv, window_size=length, min_samples=length)

    return pl.DataFrame(
        {
            "sqz_on": sqz_on.fill_null(False),
            "sqz_off": sqz_off.fill_null(False),
            "sqz_momentum": momentum,
        }
    )


# ---------------------------------------------------------------------------
# Volatility Ratio
# ---------------------------------------------------------------------------


def volatility_ratio(ohlc: pl.DataFrame, period: int = 14) -> pl.Series:
    """Volatility Ratio — current True Range as a fraction of its n-period maximum.

    Measures how the current bar's True Range compares to its maximum over
    the lookback period.  Values close to 1 indicate unusually wide-range bars
    (potential breakout bars); values close to 0 indicate narrow-range bars
    (low volatility / inside-bar environment).

        VR = true_range / rolling_max(true_range, period)

    Null-prefix: ``period − 1`` bars.

    Args:
        ohlc: DataFrame with columns ``high``, ``low``, ``close``.
        period: Lookback window (default 14).

    Returns:
        Series named ``vol_ratio_{period}``.

    Raises:
        ValueError: If ``period < 1``.
    """
    _validate_period(period, "Volatility Ratio")

    tr = true_range(ohlc)
    max_tr = tr.rolling_max(window_size=period, min_samples=period)

    # fill_nan handles the rare case where the rolling maximum is zero (flat market).
    return (tr / max_tr).fill_nan(1.0).alias(f"vol_ratio_{period}")
