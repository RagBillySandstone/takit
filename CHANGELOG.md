# Changelog

All notable changes to polarticks are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.6.0] — 2026-05-20

### Added

#### Moving Averages
- `ehma(series, period)` — Exponential Hull Moving Average; uses EMA instead of WMA for hull computation; `period − 1` leading nulls
- `pwma(series, period)` — Pascal's Triangle Weighted Moving Average; coefficients from Pascal's triangle row

#### Momentum
- `disparity_index(series, period=14)` — percentage deviation of close from its SMA
- `apo(series, fast=12, slow=26)` — Absolute Price Oscillator; difference of two EMAs
- `asi(ohlc, limit_move=1.0)` — Accumulation Swing Index; Wilder's directional swing metric
- `pmo(series, first_period=35, second_period=20, signal_period=10)` — Price Momentum Oscillator; double-smoothed ROC
- `chande_trend_score(series, periods=(7, 14, 21, 28, 35, 42, 49, 56, 63, 70))` — count of positive ROC periods

#### Trend
- `ma_envelope(series, period=20, pct=0.025)` — Moving Average Envelope; upper/lower bands offset by a percentage
- `linreg_intercept(series, period=14)` — Rolling linear regression intercept
- `standard_error_bands(series, period=14, num_std=2.0)` — LinReg line ± standard error bands; returns 3-column DataFrame
- `cog(series, period=10)` — Center of Gravity oscillator; weighted mean lag
- `rwi(ohlc, period=14)` — Random Walk Index; measures deviation from a random walk

#### Volatility
- `coefficient_of_variation(series, period=14)` — rolling CV; std / mean × 100
- `efficiency_ratio(series, period=10)` — Kaufman's Efficiency Ratio; directional movement / total path
- `standard_error(series, period=14)` — rolling standard error of the regression line

#### Volume
- `vzo(ohlc_vol, period=14)` — Volume Zone Oscillator; EMA-smoothed signed volume ratio
- `mfi_bw(ohlc_vol, period=14)` — Bandwidth-adjusted Money Flow Index variant
- `volume_delta(ohlc_vol)` — Per-bar directional volume proxy (buy − sell estimate)

#### Candlestick Patterns
- `is_marubozu_bullish(ohlc, ...)` — Bullish Marubozu; full-body candle with minimal or no wicks
- `is_marubozu_bearish(ohlc, ...)` — Bearish Marubozu; full-body bearish candle with minimal wicks

### Fixed
- **Supertrend direction** — direction is now sticky and only flips when close crosses the opposite band; the previous implementation recomputed direction stateless on each bar, causing immediate reversals in trending markets
- **VWAP / vwap_bands session detection** — session reset now fires only on the first bar entering `session_start_hour` (hour transition), not on every bar within that hour; M1/M5 data was resetting VWAP up to 60 times per session
- **`crossover` / `crossunder` bar-0** — bar 0 is now always `False`; a crossing requires a prior bar to cross *from*, and the previous sentinel `fill_null(-1.0)` was manufacturing a spurious signal on the first bar

### Tests
- 147 new unit tests in `test_v06_indicators.py`
- 4 regression tests for the three bugs fixed above (1099 total)

---

## [0.5.0] — 2026-05-20

### Added

#### Moving Averages
- `trima(series, period)` — Triangular Moving Average; double-smoothed SMA with triangular weight profile; `period − 1` leading nulls
- `vidya(series, cmo_period=9, alpha=0.2)` — Variable Index Dynamic Average; CMO-adaptive EMA; `cmo_period` leading nulls

#### Momentum
- `crsi(series, rsi_period=3, streak_period=2, rank_period=100)` — Connors RSI; composite of RSI, streak RSI, and rolling percent rank
- `qstick(ohlc, period=8)` — Q-Stick; EMA of close minus open; intraday directional bias
- `psy_line(series, period=14)` — Psychological Line; percentage of rising bars in a rolling window
- `rocr(series, period=10)` — Rate of Change Ratio; `close / close[n]`

#### Trend
- `vhf(series, period=28)` — Vertical Horizontal Filter; quantifies trending vs. ranging regime
- `pfe(series, period=14, smooth=5)` — Polarized Fractal Efficiency; directional path-efficiency oscillator
- `chande_forecast_oscillator(series, period=14)` — % deviation of close from the Time Series Forecast
- `linreg_r2(series, period=14)` — Rolling R² coefficient of determination
- `tii(series, period=20)` — Trend Intensity Index; fraction of closes above/below the SMA

#### Volatility
- `bbw(series, period=20, num_std=2.0)` — Bollinger Band Width; `(upper − lower) / middle`; normalised spread
- `bbp(series, period=20, num_std=2.0)` — Bollinger %B; price position within Bollinger Bands
- `realized_variance(series, period=20, annualize=True)` — Rolling annualised sum of squared log-returns

#### Volume
- `rvol(ohlc_vol, period=20)` — Relative Volume; current volume relative to its rolling average
- `obv_osc(ohlc_vol, fast=5, slow=10)` — OBV Oscillator; EMA spread of On-Balance Volume
- `volume_roc(ohlc_vol, period=14)` — Volume Rate of Change; percentage change in trading volume

#### Candlestick Patterns
- `is_dragonfly_doji(ohlc, ...)` — Doji with long lower shadow and minimal upper shadow
- `is_gravestone_doji(ohlc, ...)` — Doji with long upper shadow and minimal lower shadow
- `is_spinning_top(ohlc, ...)` — Small non-doji body with roughly equal upper and lower wicks

### Tests
- 145 new unit tests across `test_v05_indicators.py` and `test_null_prefix.py` (948 total)

### Fixed
- mypy strict-mode errors in CRSI streak loop (`momentum.py`) and squeeze momentum LRV helper (`volatility.py`)

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
