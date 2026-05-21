# polarticks — TODO

## Indicators to add

### Moving averages
- [x] **Hull MA** (`hma`) — `WMA(2*WMA(n/2) - WMA(n), sqrt(n))`. Reduces lag while staying smooth.
- [x] **VWMA** (`vwma`) — volume-weighted moving average; requires `volume` column alongside price.
- [x] **McGinley Dynamic** — self-adjusting MA that tracks price more closely during fast moves.

### Trend / directional
- [x] **ADX** (`adx`) — Average Directional Index with +DI / -DI components. Was explicitly deferred from the initial build. Returns a 3-column DataFrame (`adx`, `plus_di`, `minus_di`).
- [x] **Supertrend** (`supertrend`) — ATR-based trailing stop/trend indicator. Returns direction (`1` / `-1`) and the band level.
- [x] **Parabolic SAR** (`psar`) — acceleration-factor dot plot. Useful for trailing stop placement.

### Oscillators / momentum
- [x] **MFI** (`mfi`) — Money Flow Index (volume-weighted RSI). Requires OHLCV.
- [x] **CMF** (`cmf`) — Chaikin Money Flow. Requires OHLCV.
- [x] **TSI** (`tsi`) — True Strength Index (double-smoothed momentum oscillator).
- [x] **Ultimate Oscillator** (`ultimate_oscillator`) — weighted blend of 3 time-frame oscillators.

### Volatility
- [x] **Chaikin Volatility** (`chaikin_volatility`) — rate of change of EMA of H-L range.
- [x] **Historical Volatility** (`historical_volatility`) — rolling annualised standard deviation of log returns.
- [x] **Ulcer Index** (`ulcer_index`) — drawdown-based volatility measure; useful for risk-adjusted metrics.

### Levels / structure
- [x] **Fibonacci pivot points** (`pivot_points_fibonacci`) — PP ± (0.382, 0.618, 1.0) × range.
- [x] **Woodie pivot points** (`pivot_points_woodie`) — weights close more heavily than floor pivots.
- [x] **DeMark pivot points** (`pivot_points_demark`) — conditional on prior open vs. close.

### Patterns
- [x] **Three white soldiers / three black crows** — three consecutive strong candles in one direction.
- [x] **Morning star / evening star** — three-candle reversal pattern with an inside doji.
- [x] **Harami** (bullish + bearish) — inside candle within a larger opposing candle.

### Volume
- [x] **OBV** (`obv`) — On-Balance Volume; running total of signed volume.
- [x] **VWAP with standard deviation bands** — extend `vwap()` with ±1σ / ±2σ bands computed from intraday deviation.

---

## New indicators (v0.2.0 roadmap)

### Moving averages
- [x] **ALMA** (`alma`) — Arnaud Legoux Moving Average; Gaussian-weighted, good at noise reduction.
- [x] **KAMA** (`kama`) — Kaufman Adaptive Moving Average; self-adjusts speed based on market noise. ⭐
- [x] **ZLEMA** (`zlema`) — Zero Lag EMA; removes EMA lag via error correction.
- [x] **T3** (`t3`) — Tillson Triple EMA; smoother than TEMA, distinct algorithm.

### Momentum / oscillators
- [x] **StochRSI** (`stoch_rsi`) — Stochastic applied to RSI; very popular in crypto/algo trading. ⭐
- [x] **Aroon** (`aroon`) — Aroon Up/Down/Oscillator; measures time since highest high / lowest low.
- [x] **PPO** (`ppo`) — Percentage Price Oscillator; MACD normalised as a percentage.
- [x] **CMO** (`cmo`) — Chande Momentum Oscillator; RSI variant using positive/negative momentum ratio.
- [x] **DPO** (`dpo`) — Detrended Price Oscillator; removes trend to isolate price cycles.
- [x] **KST** (`kst`) — Know Sure Thing; weighted sum of ROC at four timeframes.
- [x] **Coppock Curve** (`coppock`) — long-term momentum oscillator.

### Trend
- [x] **Ichimoku Cloud** (`ichimoku`) — 5-component system (Tenkan, Kijun, Senkou A/B, Chikou); multi-output. ⭐
- [x] **Linear Regression Slope** (`linreg_slope`) — slope of regression line over rolling window.
- [x] **Vortex Indicator** (`vortex`) — VI+ and VI− measure directional movement intensity.
- [x] **TRIX** (`trix`) — triple-smoothed EMA, 1-period ROC; great noise filter. ⭐
- [x] **Schaff Trend Cycle** (`stc`) — stochastic of MACD; faster cycle detection.

### Volatility
- [x] **Chandelier Exit** (`chandelier_exit`) — ATR-based trailing stop; widely used for position exits. ⭐
- [x] **Mass Index** (`mass_index`) — detects reversals by measuring range expansion.
- [x] **NATR** (`natr`) — Normalised ATR; ATR as % of close, comparable across instruments.

### Volume
- [x] **A/D Line** (`ad_line`) — Accumulation/Distribution; OBV variant weighted by position in bar range.
- [x] **Klinger Volume Oscillator** (`kvo`) — short vs long EMA of signed volume flow.
- [x] **Ease of Movement** (`eom`) — relates price change to volume; low EOM = easy movement.
- [x] **PVT** (`pvt`) — Price Volume Trend; cumulative volume scaled by % price change.

### Candlestick patterns (multi-candle)
- [x] **Abandoned Baby** (bullish + bearish) — gap + doji gap reversal sequence.

> ⭐ = high-value picks to implement first

---

## API / quality improvements
- [x] **Vectorise pin bar wicks** — replace Python `zip` loop in `is_pin_bar_bullish/bearish` with `pl.min_horizontal` / `pl.max_horizontal` for speed on large datasets.
- [x] **`crossover` / `crossunder` with tolerance** — add optional `atol` parameter to handle floating-point equality at cross.
- [x] **Null-prefix consistency audit** — verify every indicator returns exactly `period - 1` leading nulls (no accidental zeroes or extra nulls).

---

## New indicators (v0.3.0 roadmap)

### Momentum / oscillators
- [x] **Fisher Transform** (`fisher_transform`) — normalizes HL midpoint to a Gaussian distribution via arctanh; highlights turning points.

### Volatility
- [x] **Parkinson Volatility** (`parkinson`) — high-low range-based estimator; more efficient than close-to-close HV.
- [x] **Garman-Klass Volatility** (`garman_klass`) — OHLC estimator that accounts for open-close drift.
- [x] **Yang-Zhang Volatility** (`yang_zhang`) — accounts for overnight gaps; combines Garman-Klass with Rogers-Satchell.
- [x] **Williams VIX Fix** (`williams_vix_fix`) — synthetic fear gauge: `(rolling_max(close) − low) / rolling_max(close)`.

### Trend
- [x] **Elder Ray Index** (`elder_ray`) — Bull Power (`high − EMA`) and Bear Power (`low − EMA`).

### Volume
- [x] **Force Index** (`force_index`) — Elder's force index: EMA of `(close − prev_close) × volume`.
- [x] **NVI** (`nvi`) — Negative Volume Index; cumulates price-change only on days when volume falls.
- [x] **PVI** (`pvi`) — Positive Volume Index; cumulates price-change only on days when volume rises.

### Utilities
- [x] **Rolling Highest** (`rolling_highest`) — rolling n-period maximum; building block for many indicators.
- [x] **Rolling Lowest** (`rolling_lowest`) — rolling n-period minimum.
- [x] **Rolling Std** (`rolling_std`) — rolling n-period sample standard deviation.
- [x] **Percent Rank** (`percent_rank`) — rolling percentile rank of current value within the last n bars.

### Levels / structure
- [x] **Fibonacci Retracement** (`fibonacci_retracement`) — levels at 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100% of a given high-low range.

---

## New indicators (v0.5.0 roadmap)

### Moving averages
- [x] **TRIMA** (`trima`) — Triangular MA; double-smoothed SMA with triangular weight profile.
- [x] **VIDYA** (`vidya`) — Variable Index Dynamic Average; CMO-adaptive EMA.

### Momentum / oscillators
- [x] **Connors RSI** (`crsi`) — Composite of RSI(3), streak RSI, and rolling percent rank.
- [x] **Q-Stick** (`qstick`) — EMA of close-minus-open; intraday directional bias.
- [x] **Psychological Line** (`psy_line`) — Percentage of rising bars in rolling window.
- [x] **ROCR** (`rocr`) — Rate of Change Ratio (close / close[n]); ratio form of ROC.

### Trend
- [x] **VHF** (`vhf`) — Vertical Horizontal Filter; quantifies trending vs. ranging regime.
- [x] **PFE** (`pfe`) — Polarized Fractal Efficiency; directional path-efficiency oscillator.
- [x] **Chande Forecast Oscillator** (`chande_forecast_oscillator`) — % deviation of close from TSF.
- [x] **Linear Regression R²** (`linreg_r2`) — Rolling R² coefficient of determination.
- [x] **Trend Intensity Index** (`tii`) — Fraction of closes above/below the SMA.

### Volatility
- [x] **Bollinger Band Width** (`bbw`) — (upper − lower) / middle; normalised spread.
- [x] **Bollinger %B** (`bbp`) — Price position within Bollinger Bands.
- [x] **Realized Variance** (`realized_variance`) — Rolling annualised sum of squared log-returns.

### Volume
- [x] **Relative Volume** (`rvol`) — Current volume relative to its rolling average.
- [x] **OBV Oscillator** (`obv_osc`) — EMA spread of On-Balance Volume.
- [x] **Volume Rate of Change** (`volume_roc`) — Percentage change in trading volume.

### Candlestick patterns
- [x] **Dragonfly Doji** (`is_dragonfly_doji`) — Doji with long lower shadow.
- [x] **Gravestone Doji** (`is_gravestone_doji`) — Doji with long upper shadow.
- [x] **Spinning Top** (`is_spinning_top`) — Small body with equal upper and lower wicks.

---

## New indicators (v0.6.0 roadmap)

### Moving averages
- [x] **EHMA** (`ehma`) — Exponential Hull Moving Average; EMA-based hull variant.
- [x] **PWMA** (`pwma`) — Pascal's Triangle Weighted Moving Average.

### Momentum / oscillators
- [x] **Disparity Index** (`disparity_index`) — % deviation of close from SMA.
- [x] **APO** (`apo`) — Absolute Price Oscillator; difference of two EMAs.
- [x] **ASI** (`asi`) — Accumulation Swing Index; Wilder's directional swing metric.
- [x] **PMO** (`pmo`) — Price Momentum Oscillator; double-smoothed ROC with signal.
- [x] **Chande Trend Score** (`chande_trend_score`) — count of positive ROC across multiple periods.

### Trend
- [x] **MA Envelope** (`ma_envelope`) — SMA ± percentage offset bands.
- [x] **LinReg Intercept** (`linreg_intercept`) — Rolling linear regression intercept.
- [x] **Standard Error Bands** (`standard_error_bands`) — LinReg ± standard error channel.
- [x] **COG** (`cog`) — Center of Gravity oscillator.
- [x] **RWI** (`rwi`) — Random Walk Index; trend vs. random walk test.

### Volatility
- [x] **Coefficient of Variation** (`coefficient_of_variation`) — rolling std / mean × 100.
- [x] **Efficiency Ratio** (`efficiency_ratio`) — Kaufman directional efficiency ratio.
- [x] **Standard Error** (`standard_error`) — rolling regression standard error.

### Volume
- [x] **VZO** (`vzo`) — Volume Zone Oscillator.
- [x] **MFI_BW** (`mfi_bw`) — Bandwidth-adjusted MFI variant.
- [x] **Volume Delta** (`volume_delta`) — per-bar directional volume proxy.

### Candlestick patterns
- [x] **Marubozu Bullish** (`is_marubozu_bullish`) — full-body bullish candle with minimal wicks.
- [x] **Marubozu Bearish** (`is_marubozu_bearish`) — full-body bearish candle with minimal wicks.

---

## Packaging
- [x] **PyPI release** — bump to `0.1.0`, write `CHANGELOG.md`, publish to PyPI.
- [x] **Benchmark suite** — `tests/benchmark/` using `pytest-benchmark` to track performance on a 100k-bar series across all indicators.
- [x] **`py.typed` marker** — already present; verify downstream mypy usage works end-to-end.
- [x] **Bump to v0.5.0** — update `pyproject.toml` version, add CHANGELOG entry, publish to PyPI.
- [x] **Bump to v0.6.0** — update `pyproject.toml` version, add CHANGELOG entry, publish to PyPI. Fixed Supertrend direction, VWAP session detection, and crossover/crossunder bar-0 before publishing.
