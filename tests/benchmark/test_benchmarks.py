"""Performance benchmarks for all takit indicators.

Run with::

    uv run pytest tests/benchmark/ --benchmark-only

Each benchmark exercises the indicator on a 100 000-bar OHLCV series to
produce timings that are representative of real-world intraday datasets.
Results can be compared across versions with::

    uv run pytest tests/benchmark/ --benchmark-compare

Fixtures
--------
ohlcv_100k : pl.DataFrame
    100 000 rows of synthetic OHLCV data generated with deterministic
    pseudo-random values so results are reproducible.
close_100k : pl.Series
    The ``close`` column extracted for single-series indicators.
volume_100k : pl.Series
    The ``volume`` column for volume-weighted indicators.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl
import pytest

import takit
from takit.moving_averages import mcginley_dynamic, wilder_smooth
from takit.volatility import true_range

# ---------------------------------------------------------------------------
# 100 000-bar synthetic OHLCV fixture
# ---------------------------------------------------------------------------

_N = 100_000


def _build_ohlcv(n: int) -> pl.DataFrame:
    """Generate *n* bars of deterministic synthetic OHLCV data.

    Uses a simple sine-wave price model to produce realistic H > O/C > L
    relationships without randomness.

    Args:
        n: Number of bars to generate.

    Returns:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``, ``volume``.
    """
    closes = [100.0 + math.sin(i * 0.003) * 20.0 + i * 0.001 for i in range(n)]
    highs = [c + abs(math.cos(i * 0.005)) * 1.5 + 0.2 for i, c in enumerate(closes)]
    lows = [c - abs(math.cos(i * 0.005)) * 1.5 - 0.2 for i, c in enumerate(closes)]
    opens = [c + math.sin(i * 0.007) * 0.5 for i, c in enumerate(closes)]
    volumes = [10_000.0 + math.sin(i * 0.01) * 3_000.0 for i in range(n)]
    return pl.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


# Build once at module level so fixture overhead is not counted in each timing.
_OHLCV = _build_ohlcv(_N)
_CLOSE = _OHLCV["close"]
_VOLUME = _OHLCV["volume"]


@pytest.fixture(scope="module")
def ohlcv_100k() -> pl.DataFrame:
    """Return the shared 100 000-bar OHLCV DataFrame."""
    return _OHLCV


@pytest.fixture(scope="module")
def close_100k() -> pl.Series:
    """Return the ``close`` column of the 100 000-bar fixture."""
    return _CLOSE


@pytest.fixture(scope="module")
def volume_100k() -> pl.Series:
    """Return the ``volume`` column of the 100 000-bar fixture."""
    return _VOLUME


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


def test_sma(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: SMA with period 20."""
    benchmark(takit.sma, close_100k, 20)


def test_ema(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: EMA with period 20."""
    benchmark(takit.ema, close_100k, 20)


def test_wma(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: WMA with period 20."""
    benchmark(takit.wma, close_100k, 20)


def test_wilder_smooth(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: Wilder's smoothing (RMA) with period 14."""
    benchmark(wilder_smooth, close_100k, 14)


def test_dema(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: DEMA with period 20."""
    benchmark(takit.dema, close_100k, 20)


def test_tema(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: TEMA with period 20."""
    benchmark(takit.tema, close_100k, 20)


def test_hma(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: HMA with period 20."""
    benchmark(takit.hma, close_100k, 20)


def test_vwma(benchmark: Any, close_100k: pl.Series, volume_100k: pl.Series) -> None:
    """Benchmark: VWMA with period 20."""
    benchmark(takit.vwma, close_100k, volume_100k, 20)


def test_mcginley_dynamic(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: McGinley Dynamic with period 14 (Python loop — expected slower)."""
    benchmark(mcginley_dynamic, close_100k, 14)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def test_rsi(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: RSI with period 14."""
    benchmark(takit.rsi, close_100k, 14)


def test_macd(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: MACD with default periods (12, 26, 9)."""
    benchmark(takit.macd, close_100k)


def test_stochastic(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Stochastic Oscillator (14, 3)."""
    benchmark(takit.stochastic, ohlcv_100k)


def test_williams_r(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Williams %R with period 14."""
    benchmark(takit.williams_r, ohlcv_100k, 14)


def test_cci(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: CCI with period 20."""
    benchmark(takit.cci, ohlcv_100k, 20)


def test_roc(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: ROC with period 10."""
    benchmark(takit.roc, close_100k, 10)


def test_mfi(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: MFI with period 14."""
    benchmark(takit.mfi, ohlcv_100k, 14)


def test_cmf(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: CMF with period 20."""
    benchmark(takit.cmf, ohlcv_100k, 20)


def test_tsi(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: TSI with default periods (25, 13)."""
    benchmark(takit.tsi, close_100k)


def test_ultimate_oscillator(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Ultimate Oscillator with default periods (7, 14, 28)."""
    benchmark(takit.ultimate_oscillator, ohlcv_100k)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def test_true_range(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: True Range (single-pass, no period)."""
    benchmark(true_range, ohlcv_100k)


def test_atr(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: ATR with period 14."""
    benchmark(takit.atr, ohlcv_100k, 14)


def test_bollinger_bands(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: Bollinger Bands with period 20, 2 std."""
    benchmark(takit.bollinger_bands, close_100k, 20)


def test_keltner_channels(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Keltner Channels with default periods."""
    benchmark(takit.keltner_channels, ohlcv_100k)


def test_chaikin_volatility(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Chaikin Volatility with default periods (10, 10)."""
    benchmark(takit.chaikin_volatility, ohlcv_100k)


def test_historical_volatility(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: Historical Volatility with period 20."""
    benchmark(takit.historical_volatility, close_100k, 20)


def test_ulcer_index(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: Ulcer Index with period 14."""
    benchmark(takit.ulcer_index, close_100k, 14)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def test_donchian_channels(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Donchian Channels with period 20."""
    benchmark(takit.donchian_channels, ohlcv_100k, 20)


def test_adx(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: ADX with period 14."""
    benchmark(takit.adx, ohlcv_100k, 14)


def test_supertrend(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Supertrend with period 7 (Python loop — expected slower)."""
    benchmark(takit.supertrend, ohlcv_100k, 7)


def test_parabolic_sar(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Parabolic SAR with default acceleration factors."""
    benchmark(takit.parabolic_sar, ohlcv_100k)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def test_obv(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: On-Balance Volume (single-pass cumulative sum)."""
    benchmark(takit.obv, ohlcv_100k)


def test_vwap(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Session-anchored VWAP (Python loop — expected slower)."""
    benchmark(takit.vwap, ohlcv_100k)


def test_vwap_bands(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: VWAP with ±1σ / ±2σ bands."""
    benchmark(takit.vwap_bands, ohlcv_100k)


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------


def test_pivot_floor(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Floor pivot points."""
    benchmark(
        takit.pivot_points_floor,
        ohlcv_100k["high"],
        ohlcv_100k["low"],
        ohlcv_100k["close"],
    )


def test_pivot_camarilla(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Camarilla pivot points."""
    benchmark(
        takit.pivot_points_camarilla,
        ohlcv_100k["high"],
        ohlcv_100k["low"],
        ohlcv_100k["close"],
    )


def test_pivot_fibonacci(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Fibonacci pivot points."""
    benchmark(
        takit.pivot_points_fibonacci,
        ohlcv_100k["high"],
        ohlcv_100k["low"],
        ohlcv_100k["close"],
    )


def test_pivot_woodie(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Woodie pivot points."""
    benchmark(
        takit.pivot_points_woodie,
        ohlcv_100k["high"],
        ohlcv_100k["low"],
        ohlcv_100k["close"],
    )


def test_pivot_demark(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: DeMark pivot points (prev_open is the first positional arg)."""
    benchmark(
        takit.pivot_points_demark,
        ohlcv_100k["open"],
        ohlcv_100k["high"],
        ohlcv_100k["low"],
        ohlcv_100k["close"],
    )


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


def test_is_bullish_engulfing(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Bullish engulfing pattern detection."""
    benchmark(takit.is_bullish_engulfing, ohlcv_100k)


def test_is_bearish_engulfing(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Bearish engulfing pattern detection."""
    benchmark(takit.is_bearish_engulfing, ohlcv_100k)


def test_is_pin_bar_bullish(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Bullish pin bar detection."""
    benchmark(takit.is_pin_bar_bullish, ohlcv_100k)


def test_is_pin_bar_bearish(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Bearish pin bar detection."""
    benchmark(takit.is_pin_bar_bearish, ohlcv_100k)


def test_is_inside_bar(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Inside bar pattern detection."""
    benchmark(takit.is_inside_bar, ohlcv_100k)


def test_is_doji(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Doji pattern detection."""
    benchmark(takit.is_doji, ohlcv_100k)


def test_is_three_white_soldiers(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Three white soldiers pattern detection."""
    benchmark(takit.is_three_white_soldiers, ohlcv_100k)


def test_is_three_black_crows(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Three black crows pattern detection."""
    benchmark(takit.is_three_black_crows, ohlcv_100k)


def test_is_morning_star(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Morning star pattern detection."""
    benchmark(takit.is_morning_star, ohlcv_100k)


def test_is_evening_star(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Evening star pattern detection."""
    benchmark(takit.is_evening_star, ohlcv_100k)


def test_is_bullish_harami(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Bullish harami pattern detection."""
    benchmark(takit.is_bullish_harami, ohlcv_100k)


def test_is_bearish_harami(benchmark: Any, ohlcv_100k: pl.DataFrame) -> None:
    """Benchmark: Bearish harami pattern detection."""
    benchmark(takit.is_bearish_harami, ohlcv_100k)


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def test_crossover(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: crossover() between close and a lagged version of itself."""
    fast = takit.ema(close_100k, 12)
    slow = takit.ema(close_100k, 26)
    benchmark(takit.crossover, fast, slow)


def test_crossunder(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: crossunder() between two EMA series."""
    fast = takit.ema(close_100k, 12)
    slow = takit.ema(close_100k, 26)
    benchmark(takit.crossunder, fast, slow)


def test_log_returns(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: log_returns() on 100 000 prices."""
    benchmark(takit.log_returns, close_100k)


def test_simple_returns(benchmark: Any, close_100k: pl.Series) -> None:
    """Benchmark: simple_returns() on 100 000 prices."""
    benchmark(takit.simple_returns, close_100k)
