"""
takit — Polars-native technical analysis indicator library.

Import from the top-level package for convenience:

    from takit import sma, ema, rsi, atr, bollinger_bands
    from takit import crossover, is_bullish_engulfing, vwap

Or import from the specific module for clarity:

    from polarticks.momentum import rsi
    from polarticks.volatility import atr, keltner_channels
"""

from polarticks.levels import (
    pivot_points_camarilla,
    pivot_points_demark,
    pivot_points_fibonacci,
    pivot_points_floor,
    pivot_points_woodie,
)
from polarticks.momentum import (
    cci,
    cmf,
    macd,
    mfi,
    roc,
    rsi,
    stochastic,
    tsi,
    ultimate_oscillator,
    williams_r,
)
from polarticks.moving_averages import (
    dema,
    ema,
    hma,
    mcginley_dynamic,
    sma,
    tema,
    vwma,
    wilder_smooth,
    wma,
)
from polarticks.patterns import (
    is_bearish_engulfing,
    is_bearish_harami,
    is_bullish_engulfing,
    is_bullish_harami,
    is_doji,
    is_evening_star,
    is_inside_bar,
    is_morning_star,
    is_pin_bar_bearish,
    is_pin_bar_bullish,
    is_three_black_crows,
    is_three_white_soldiers,
)
from polarticks.trend import adx, donchian_channels, parabolic_sar, supertrend
from polarticks.utils import crossover, crossunder, log_returns, simple_returns
from polarticks.volatility import (
    atr,
    bollinger_bands,
    chaikin_volatility,
    historical_volatility,
    keltner_channels,
    true_range,
    ulcer_index,
)
from polarticks.volume import obv, vwap, vwap_bands

__all__ = [
    # Moving averages
    "sma",
    "ema",
    "wma",
    "wilder_smooth",
    "dema",
    "tema",
    "hma",
    "vwma",
    "mcginley_dynamic",
    # Momentum
    "rsi",
    "macd",
    "stochastic",
    "williams_r",
    "cci",
    "roc",
    "mfi",
    "cmf",
    "tsi",
    "ultimate_oscillator",
    # Volatility
    "true_range",
    "atr",
    "bollinger_bands",
    "keltner_channels",
    "chaikin_volatility",
    "historical_volatility",
    "ulcer_index",
    # Trend
    "donchian_channels",
    "adx",
    "supertrend",
    "parabolic_sar",
    # Volume
    "vwap",
    "vwap_bands",
    "obv",
    # Levels
    "pivot_points_floor",
    "pivot_points_camarilla",
    "pivot_points_fibonacci",
    "pivot_points_woodie",
    "pivot_points_demark",
    # Patterns
    "is_bullish_engulfing",
    "is_bearish_engulfing",
    "is_pin_bar_bullish",
    "is_pin_bar_bearish",
    "is_inside_bar",
    "is_doji",
    "is_three_white_soldiers",
    "is_three_black_crows",
    "is_morning_star",
    "is_evening_star",
    "is_bullish_harami",
    "is_bearish_harami",
    # Utils
    "crossover",
    "crossunder",
    "log_returns",
    "simple_returns",
]
