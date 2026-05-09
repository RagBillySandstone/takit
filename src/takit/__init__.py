"""
takit — Polars-native technical analysis indicator library.

Import from the top-level package for convenience:

    from takit import sma, ema, rsi, atr, bollinger_bands
    from takit import crossover, is_bullish_engulfing, vwap

Or import from the specific module for clarity:

    from takit.momentum import rsi
    from takit.volatility import atr, keltner_channels
"""

from takit.levels import pivot_points_camarilla, pivot_points_floor
from takit.momentum import cci, macd, roc, rsi, stochastic, williams_r
from takit.moving_averages import dema, ema, sma, tema, wilder_smooth, wma
from takit.patterns import (
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_doji,
    is_inside_bar,
    is_pin_bar_bearish,
    is_pin_bar_bullish,
)
from takit.trend import donchian_channels
from takit.utils import crossover, crossunder, log_returns, simple_returns
from takit.volatility import atr, bollinger_bands, keltner_channels, true_range
from takit.volume import vwap

__all__ = [
    # Moving averages
    "sma",
    "ema",
    "wma",
    "wilder_smooth",
    "dema",
    "tema",
    # Momentum
    "rsi",
    "macd",
    "stochastic",
    "williams_r",
    "cci",
    "roc",
    # Volatility
    "true_range",
    "atr",
    "bollinger_bands",
    "keltner_channels",
    # Trend
    "donchian_channels",
    # Volume
    "vwap",
    # Levels
    "pivot_points_floor",
    "pivot_points_camarilla",
    # Patterns
    "is_bullish_engulfing",
    "is_bearish_engulfing",
    "is_pin_bar_bullish",
    "is_pin_bar_bearish",
    "is_inside_bar",
    "is_doji",
    # Utils
    "crossover",
    "crossunder",
    "log_returns",
    "simple_returns",
]
