# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all unit tests
uv run pytest tests/unit/

# Run a single test file
uv run pytest tests/unit/test_momentum.py

# Run a single test by name
uv run pytest tests/unit/test_momentum.py::TestRSI::test_basic_calculation

# Run benchmarks (generates baseline)
uv run pytest tests/benchmark/ --benchmark-save=baseline

# Compare benchmarks against saved baseline
uv run pytest tests/benchmark/ --benchmark-compare=baseline

# Lint and format
uv run ruff check --fix && uv run ruff format

# Type check
uv run mypy src/
```

## Architecture

**polarticks** is a Polars-native technical analysis library. The only runtime dependency is `polars>=1.0.0`. The package follows a flat module structure under `src/polarticks/`, with one file per indicator category:

| Module | Contents |
|---|---|
| `moving_averages.py` | SMA, EMA, WMA, Wilder, DEMA, TEMA, HMA, VWMA, McGinley Dynamic, KAMA, TRIX, ZLEMA, T3, ALMA, FRAMA, Laguerre, TRIMA, VIDYA, EHMA, PWMA |
| `momentum.py` | RSI, MACD, Stochastic, Williams %R, CCI, ROC, ROCR, MFI, CMF, TSI, Ultimate Oscillator, PPO, CMO, DPO, KST, Coppock, Fisher Transform, SMI, RVI, BOP, QQE, Awesome Oscillator, Accelerator Oscillator, Connors RSI, Q-Stick, Psychological Line, Disparity Index, APO, ASI, PMO, Chande Trend Score |
| `volatility.py` | True Range, ATR, Bollinger Bands, BBW, BBP, Keltner, Chaikin Volatility, Historical Volatility, Realized Variance, Ulcer Index, NATR, Chandelier Exit, Mass Index, Parkinson, Garman-Klass, Yang-Zhang, Williams VIX Fix, Choppiness Index, Squeeze Momentum, Volatility Ratio, CV, Efficiency Ratio, Standard Error |
| `trend.py` | Donchian Channels, ADX, Supertrend, Parabolic SAR, Aroon, Vortex, Ichimoku, LinReg Slope, LinReg R², STC, Elder Ray, Alligator, Fractal, LinReg Channel, TSF, Chande Kroll Stop, VHF, PFE, Chande Forecast Oscillator, TII, MA Envelope, LinReg Intercept, Standard Error Bands, COG, RWI |
| `volume.py` | OBV, OBV Oscillator, VWAP, VWAP Bands, KVO, EOM, PVT, Force Index, NVI, PVI, AD Line, Chaikin Oscillator, Volume Oscillator, RVOL, Volume ROC, TWAP, VZO, MFI_BW, Volume Delta |
| `levels.py` | Pivot Points (Floor, Camarilla, Fibonacci, Woodie, Demark), Fibonacci Retracement |
| `patterns.py` | 27 candlestick pattern detectors (single-, two-, three-, and five-bar) incl. Dragonfly Doji, Gravestone Doji, Spinning Top, Marubozu Bullish/Bearish |
| `utils.py` | crossover, crossunder, log_returns, simple_returns, rolling_highest, rolling_lowest, rolling_std, percent_rank, rolling_zscore, rolling_beta, hurst_exponent |

All 75+ public functions are re-exported from `src/polarticks/__init__.py` via `__all__`.

## Indicator Implementation Patterns

There are five distinct implementation patterns. Identify which applies before writing or reviewing indicator code.

**1. Single-series vectorised** — most indicators (SMA, EMA, RSI, etc.)
```python
def sma(series: pl.Series, period: int) -> pl.Series:
    _validate_period(period, "SMA")
    return series.rolling_mean(window_size=period, min_samples=period).alias(f"sma_{period}")
```

**2. Composed indicators** — DEMA, TEMA, HMA reuse simpler functions:
```python
def dema(series: pl.Series, period: int) -> pl.Series:
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    return (2.0 * ema1 - ema2).alias(f"dema_{period}")
```
Null-prefix compounds: DEMA = `2*(period-1)`, TEMA = `3*(period-1)`.

**3. Multi-output** — MACD, Bollinger Bands, Keltner, etc. return `pl.DataFrame`:
```python
return pl.DataFrame({"macd_line": macd_line, "macd_signal": signal_line, "macd_histogram": histogram})
```

**4. Candlestick patterns** — accept `pl.DataFrame` with OHLC columns, return `pl.Series[bool]`. Always `.fill_null(False)` on the first bar where no prior exists.

**5. Stateful/iterative** — `mcginley_dynamic`, `frama`, `laguerre`, and `qqe` use Python loops (seed from a rolling calculation, then recursively update state). All other indicators are fully vectorised.

## Null-Prefix Contract

Every indicator produces exactly `period - 1` leading nulls (or the documented equivalent for multi-pass indicators). This is a core invariant tested exhaustively in `tests/unit/test_null_prefix.py` (121 tests). When adding indicators:
- Use `min_samples=period` on Polars rolling methods to enforce this.
- Never fill warm-up nulls with zeros — they must remain null.
- Diff-based indicators (ROC, RSI) produce `period` nulls (one extra from the diff).

## Input Conventions

- **Single-series indicators:** accept `pl.Series` (e.g., `close`).
- **Multi-column indicators:** accept `pl.DataFrame` with lowercase columns `open`, `high`, `low`, `close`, `volume`.
- **Pivot points:** accept individual `pl.Series` per OHLC component to support daily broadcast.

## Testing Conventions

- Unit tests use `pytest` with class-based grouping, e.g., `class TestRSI`.
- Fixtures use synthetic price series (constant, sinusoidal OHLCV) — no external data.
- Float assertions use `pytest.approx()`.
- Each indicator test class checks: output length, leading-null count, first valid value, and at least one edge case (e.g., `period=1`, invalid period raises `ValueError`).
- Benchmarks in `tests/benchmark/test_benchmarks.py` run all indicators on 100k-bar synthetic series.

## Ruff Configuration

Line length is 100. Rules enabled: `E`, `F`, `I`, `UP`, `B`, `SIM`. E501 (line-too-long) is ignored. Always run `uv run ruff check --fix && uv run ruff format` before committing.
