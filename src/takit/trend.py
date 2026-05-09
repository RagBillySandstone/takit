"""
Trend-following indicators.

Functions
---------
donchian_channels   Highest high / lowest low channel over a rolling window
"""

from __future__ import annotations

import polars as pl

from takit.moving_averages import _validate_period


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
