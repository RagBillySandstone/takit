# takit — TODO

## Indicators to add

### Moving averages
- [ ] **Hull MA** (`hma`) — `WMA(2*WMA(n/2) - WMA(n), sqrt(n))`. Reduces lag while staying smooth.
- [ ] **VWMA** (`vwma`) — volume-weighted moving average; requires `volume` column alongside price.
- [ ] **McGinley Dynamic** — self-adjusting MA that tracks price more closely during fast moves.

### Trend / directional
- [ ] **ADX** (`adx`) — Average Directional Index with +DI / -DI components. Was explicitly deferred from the initial build. Returns a 3-column DataFrame (`adx`, `plus_di`, `minus_di`).
- [ ] **Supertrend** (`supertrend`) — ATR-based trailing stop/trend indicator. Returns direction (`1` / `-1`) and the band level.
- [ ] **Parabolic SAR** (`psar`) — acceleration-factor dot plot. Useful for trailing stop placement.

### Oscillators / momentum
- [ ] **MFI** (`mfi`) — Money Flow Index (volume-weighted RSI). Requires OHLCV.
- [ ] **CMF** (`cmf`) — Chaikin Money Flow. Requires OHLCV.
- [ ] **TSI** (`tsi`) — True Strength Index (double-smoothed momentum oscillator).
- [ ] **Ultimate Oscillator** (`ultimate_oscillator`) — weighted blend of 3 time-frame oscillators.

### Volatility
- [ ] **Chaikin Volatility** (`chaikin_volatility`) — rate of change of EMA of H-L range.
- [ ] **Historical Volatility** (`historical_volatility`) — rolling annualised standard deviation of log returns.
- [ ] **Ulcer Index** (`ulcer_index`) — drawdown-based volatility measure; useful for risk-adjusted metrics.

### Levels / structure
- [ ] **Fibonacci pivot points** (`pivot_points_fibonacci`) — PP ± (0.382, 0.618, 1.0) × range.
- [ ] **Woodie pivot points** (`pivot_points_woodie`) — weights close more heavily than floor pivots.
- [ ] **DeMark pivot points** (`pivot_points_demark`) — conditional on prior open vs. close.

### Patterns
- [ ] **Three white soldiers / three black crows** — three consecutive strong candles in one direction.
- [ ] **Morning star / evening star** — three-candle reversal pattern with an inside doji.
- [ ] **Harami** (bullish + bearish) — inside candle within a larger opposing candle.

### Volume
- [ ] **OBV** (`obv`) — On-Balance Volume; running total of signed volume.
- [ ] **VWAP with standard deviation bands** — extend `vwap()` with ±1σ / ±2σ bands computed from intraday deviation.

---

## API / quality improvements
- [ ] **Vectorise pin bar wicks** — replace Python `zip` loop in `is_pin_bar_bullish/bearish` with `pl.min_horizontal` / `pl.max_horizontal` for speed on large datasets.
- [ ] **`crossover` / `crossunder` with tolerance** — add optional `atol` parameter to handle floating-point equality at cross.
- [ ] **Null-prefix consistency audit** — verify every indicator returns exactly `period - 1` leading nulls (no accidental zeroes or extra nulls).

---

## Packaging
- [ ] **PyPI release** — bump to `0.1.0`, write `CHANGELOG.md`, publish to PyPI.
- [ ] **Benchmark suite** — `tests/benchmark/` using `pytest-benchmark` to track performance on a 100k-bar series across all indicators.
- [ ] **`py.typed` marker** — already present; verify downstream mypy usage works end-to-end.
