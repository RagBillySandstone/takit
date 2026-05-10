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
- [ ] **ALMA** (`alma`) — Arnaud Legoux Moving Average; Gaussian-weighted, good at noise reduction.
- [ ] **KAMA** (`kama`) — Kaufman Adaptive Moving Average; self-adjusts speed based on market noise. ⭐
- [ ] **ZLEMA** (`zlema`) — Zero Lag EMA; removes EMA lag via error correction.
- [ ] **T3** (`t3`) — Tillson Triple EMA; smoother than TEMA, distinct algorithm.

### Momentum / oscillators
- [ ] **StochRSI** (`stoch_rsi`) — Stochastic applied to RSI; very popular in crypto/algo trading. ⭐
- [ ] **Aroon** (`aroon`) — Aroon Up/Down/Oscillator; measures time since highest high / lowest low.
- [ ] **PPO** (`ppo`) — Percentage Price Oscillator; MACD normalised as a percentage.
- [ ] **CMO** (`cmo`) — Chande Momentum Oscillator; RSI variant using positive/negative momentum ratio.
- [ ] **DPO** (`dpo`) — Detrended Price Oscillator; removes trend to isolate price cycles.
- [ ] **KST** (`kst`) — Know Sure Thing; weighted sum of ROC at four timeframes.
- [ ] **Coppock Curve** (`coppock`) — long-term momentum oscillator.

### Trend
- [ ] **Ichimoku Cloud** (`ichimoku`) — 5-component system (Tenkan, Kijun, Senkou A/B, Chikou); multi-output. ⭐
- [ ] **Linear Regression Slope** (`linreg_slope`) — slope of regression line over rolling window.
- [ ] **Vortex Indicator** (`vortex`) — VI+ and VI− measure directional movement intensity.
- [ ] **TRIX** (`trix`) — triple-smoothed EMA, 1-period ROC; great noise filter. ⭐
- [ ] **Schaff Trend Cycle** (`stc`) — stochastic of MACD; faster cycle detection.

### Volatility
- [ ] **Chandelier Exit** (`chandelier_exit`) — ATR-based trailing stop; widely used for position exits. ⭐
- [ ] **Mass Index** (`mass_index`) — detects reversals by measuring range expansion.
- [ ] **NATR** (`natr`) — Normalised ATR; ATR as % of close, comparable across instruments.

### Volume
- [ ] **A/D Line** (`ad_line`) — Accumulation/Distribution; OBV variant weighted by position in bar range.
- [ ] **Klinger Volume Oscillator** (`kvo`) — short vs long EMA of signed volume flow.
- [ ] **Ease of Movement** (`eom`) — relates price change to volume; low EOM = easy movement.
- [ ] **PVT** (`pvt`) — Price Volume Trend; cumulative volume scaled by % price change.

### Candlestick patterns (multi-candle)
- [ ] **Abandoned Baby** (bullish + bearish) — gap + doji gap reversal sequence.

> ⭐ = high-value picks to implement first

---

## API / quality improvements
- [x] **Vectorise pin bar wicks** — replace Python `zip` loop in `is_pin_bar_bullish/bearish` with `pl.min_horizontal` / `pl.max_horizontal` for speed on large datasets.
- [x] **`crossover` / `crossunder` with tolerance** — add optional `atol` parameter to handle floating-point equality at cross.
- [x] **Null-prefix consistency audit** — verify every indicator returns exactly `period - 1` leading nulls (no accidental zeroes or extra nulls).

---

## Packaging
- [x] **PyPI release** — bump to `0.1.0`, write `CHANGELOG.md`, publish to PyPI.
- [x] **Benchmark suite** — `tests/benchmark/` using `pytest-benchmark` to track performance on a 100k-bar series across all indicators.
- [x] **`py.typed` marker** — already present; verify downstream mypy usage works end-to-end.
