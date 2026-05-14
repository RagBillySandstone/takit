# Changelog

All notable changes to polarticks are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] — 2026-05-14

### Added

#### Momentum
- `fisher_transform(ohlc, period=9)` — arctanh normalisation of the HL midpoint; returns `fisher` and `fisher_signal` columns

#### Volatility
- `parkinson(ohlc, period=20, annualise=True, trading_days=252)` — high-low range-based estimator (more efficient than close-to-close HV)
- `garman_klass(ohlc, period=20, annualise=True, trading_days=252)` — OHLC estimator accounting for open-close drift
- `yang_zhang(ohlc, period=20, annualise=True, trading_days=252)` — overnight-adjusted OHLC estimator (combines Rogers-Satchell with overnight variance); `period` leading nulls
- `williams_vix_fix(ohlc, period=22)` — synthetic fear gauge: `100 × (rolling_max(close) − low) / rolling_max(close)`

#### Trend
- `elder_ray(ohlc, period=13)` — Bull Power (`high − EMA`) and Bear Power (`low − EMA`); returns both columns in a `pl.DataFrame`

#### Volume
- `force_index(ohlcv, period=13)` — Elder's Force Index: EMA of `(close − prev_close) × volume`
- `nvi(ohlcv)` — Negative Volume Index; accumulates price change only on falling-volume bars (starts at 1000; no leading nulls)
- `pvi(ohlcv)` — Positive Volume Index; accumulates price change only on rising-volume bars (starts at 1000; no leading nulls)

#### Levels
- `fibonacci_retracement(high, low)` — seven standard Fibonacci retracement levels (0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%) from a high-low range

#### Utilities
- `rolling_highest(series, period)` — rolling n-period maximum; `period − 1` leading nulls
- `rolling_lowest(series, period)` — rolling n-period minimum; `period − 1` leading nulls
- `rolling_std(series, period)` — rolling n-period sample standard deviation (ddof=1); `period − 1` leading nulls
- `percent_rank(series, period)` — rolling percentile rank: fraction of the last n bars ≤ current value, scaled to [0, 100]

### Tests
- 110 new unit tests across `test_v03_indicators.py` and `test_null_prefix.py` (577 total)

---

## [0.1.0] — 2026-05-08

Initial public release of polarticks, a Polars-native technical analysis library.

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
