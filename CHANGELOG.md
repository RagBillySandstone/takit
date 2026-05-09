# Changelog

All notable changes to takit are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-05-08

Initial public release of takit, a Polars-native technical analysis library.

### Moving Averages
- `sma` — Simple Moving Average
- `ema` — Exponential Moving Average (α = 2 / (n + 1))
- `wma` — Weighted Moving Average (linearly weighted)
- `wilder_smooth` — Wilder's Smoothing / RMA (α = 1 / n)
- `dema` — Double EMA (2·EMA − EMA(EMA))
- `tema` — Triple EMA (3·EMA − 3·EMA² + EMA³)
- `hma` — Hull Moving Average (WMA of 2·WMA(n/2) − WMA(n), window √n)
- `vwma` — Volume-Weighted Moving Average
- `mcginley_dynamic` — McGinley Dynamic self-adjusting MA

### Momentum & Oscillators
- `rsi` — Relative Strength Index (Wilder smoothing, default 14)
- `macd` — MACD line, signal line, and histogram
- `stochastic` — Stochastic Oscillator (%K and %D)
- `williams_r` — Williams Percent Range
- `cci` — Commodity Channel Index
- `roc` — Rate of Change (percentage)
- `mfi` — Money Flow Index (volume-weighted RSI)
- `cmf` — Chaikin Money Flow
- `tsi` — True Strength Index (double-smoothed momentum)
- `ultimate_oscillator` — Weighted blend of three time-frame oscillators

### Volatility
- `true_range` — Single-bar True Range
- `atr` — Average True Range (Wilder smoothing)
- `bollinger_bands` — Bollinger Bands (middle, upper, lower, %B, bandwidth)
- `keltner_channels` — Keltner Channels (EMA ± ATR multiplier)
- `chaikin_volatility` — Rate of change of EMA(H−L range)
- `historical_volatility` — Rolling annualised standard deviation of log returns
- `ulcer_index` — Drawdown-based volatility measure

### Trend
- `donchian_channels` — Donchian Channels (rolling highest high / lowest low)
- `adx` — Average Directional Index with +DI and −DI components
- `supertrend` — ATR-based trailing stop and trend direction
- `parabolic_sar` — Parabolic SAR acceleration-factor dot plot

### Volume
- `obv` — On-Balance Volume (running signed cumulative volume)
- `vwap` — Session-anchored Volume Weighted Average Price
- `vwap_bands` — VWAP with ±1σ and ±2σ standard-deviation bands

### Levels
- `pivot_points_floor` — Classic floor-trader pivot points (PP, S1–S3, R1–R3)
- `pivot_points_camarilla` — Camarilla pivot points (S1–S4, R1–R4)
- `pivot_points_fibonacci` — Fibonacci pivot points (PP ± 0.382/0.618/1.0 × range)
- `pivot_points_woodie` — Woodie pivot points (double-weights close)
- `pivot_points_demark` — DeMark pivot points (conditional on open vs. close)

### Candlestick Patterns
- `is_bullish_engulfing` / `is_bearish_engulfing`
- `is_pin_bar_bullish` / `is_pin_bar_bearish` (vectorised with `pl.min/max_horizontal`)
- `is_inside_bar`
- `is_doji`
- `is_three_white_soldiers` / `is_three_black_crows`
- `is_morning_star` / `is_evening_star`
- `is_bullish_harami` / `is_bearish_harami`

### Utilities
- `crossover` / `crossunder` — Series cross detection with optional `atol` tolerance
- `log_returns` / `simple_returns` — Bar-to-bar return series

### Quality
- All indicators produce `null` (not `0.0`) during their warm-up period
- Null-prefix counts documented and enforced by 44 dedicated audit tests
- `py.typed` marker included for downstream mypy compatibility
- Full `--strict` mypy pass with no errors
- 54-benchmark suite (`tests/benchmark/`) via `pytest-benchmark` on 100 000-bar series

### Bug Fixes (included in initial release)
- `is_doji` now returns `True` for zero-range bars (`high == low`) — previously returned `False`
  due to NaN propagating through the boolean comparison
- `mfi` now treats equal-TP bars as neutral; previously they were allocated to the negative
  money-flow bucket, causing slightly understated MFI readings
- `cci` now returns `0.0` for perfectly flat windows instead of propagating silent `NaN`
