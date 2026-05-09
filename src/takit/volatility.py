"""
Volatility indicators.

All multi-input functions accept a ``pl.DataFrame`` with generic OHLC
column names (``open``, ``high``, ``low``, ``close``).  Single-series
functions accept a ``pl.Series``.

Functions
---------
true_range      Single-bar True Range (prerequisite for ATR)
atr             Average True Range (Wilder smoothing, default period 14)
bollinger_bands Bollinger Bands: middle, upper, lower, %B, bandwidth
keltner_channels Keltner Channels: middle (EMA), upper, lower
"""

from __future__ import annotations

import polars as pl

from takit.moving_averages import _validate_period, ema, wilder_smooth

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
