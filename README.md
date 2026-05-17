# PolarTicks

A Polars-native technical analysis library for Python.

Every indicator is implemented directly against the Polars API — no pandas
conversions, no NumPy loops where a vectorised expression will do.  Inputs and
outputs are plain `pl.Series` or `pl.DataFrame` objects, so the results drop
straight into your existing Polars pipeline.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Input conventions](#input-conventions)
- [Null-prefix semantics](#null-prefix-semantics)
- [API reference](#api-reference)
  - [Moving averages](#moving-averages)
  - [Momentum & oscillators](#momentum--oscillators)
  - [Volatility](#volatility)
  - [Trend](#trend)
  - [Volume](#volume)
  - [Levels (pivot points)](#levels-pivot-points)
  - [Candlestick patterns](#candlestick-patterns)
  - [Utilities](#utilities)
- [Running tests](#running-tests)
- [Running benchmarks](#running-benchmarks)

---

## Installation

```bash
# with uv (recommended)
uv add polarticks

# with pip
pip install polarticks
```

Requires Python ≥ 3.11 and Polars ≥ 1.0.

---

## Quick start

```python
import polars as pl
import polarticks

# Load your OHLCV data however you like
df = pl.read_csv("prices.csv")

# Single-series indicators
close   = df["close"]
rsi_14  = polarticks.rsi(close, period=14)          # pl.Series
ema_20  = polarticks.ema(close, period=20)           # pl.Series
macd_df = polarticks.macd(close)                     # pl.DataFrame (3 columns)

# Multi-column indicators
bb  = polarticks.bollinger_bands(close, period=20)   # pl.DataFrame (5 columns)
atr = polarticks.atr(df, period=14)                  # pl.Series

# Pattern detection
signals = polarticks.is_bullish_engulfing(df)        # pl.Series[bool]

# Attach results to your DataFrame
df = df.with_columns([
    rsi_14.alias("rsi_14"),
    ema_20.alias("ema_20"),
    *bb.get_columns(),                          # spread all band columns
    signals.alias("bull_engulf"),
])
```

### A complete signal pipeline

```python
import polars as pl
import polarticks

df = pl.read_csv("eurusd_h1.csv")
close = df["close"]

# Compute a fast/slow EMA crossover and RSI filter
fast = polarticks.ema(close, 9)
slow = polarticks.ema(close, 21)

df = df.with_columns([
    fast.alias("ema_9"),
    slow.alias("ema_21"),
    polarticks.rsi(close, 14).alias("rsi"),
    polarticks.crossover(fast, slow).alias("cross_up"),
    polarticks.crossunder(fast, slow).alias("cross_dn"),
])

# Long entry: crossover AND RSI not overbought
df = df.with_columns(
    (pl.col("cross_up") & (pl.col("rsi") < 70)).alias("long_entry")
)
```

---

## Input conventions

### Single-series indicators

Functions like `sma`, `ema`, `rsi`, `roc`, `hma` accept a `pl.Series`:

```python
result = polarticks.sma(df["close"], period=20)
```

### OHLC / OHLCV indicators

Functions that need multiple price columns accept a `pl.DataFrame`.  The
expected column names are always lowercase: `open`, `high`, `low`, `close`,
`volume`.

```python
result = polarticks.atr(df, period=14)         # needs high, low, close
result = polarticks.mfi(df, period=14)         # needs high, low, close, volume
result = polarticks.stochastic(df)             # needs high, low, close
```

### Pivot points

Pivot point functions accept individual `pl.Series` (one per OHLC component)
rather than a DataFrame.  This lets you broadcast yesterday's session values
across today's intraday bars however your data model requires.

```python
# Scalar broadcast: yesterday's values repeated across all bars
n = len(df)
levels = polarticks.pivot_points_floor(
    prev_high  = pl.Series([prev_high]  * n),
    prev_low   = pl.Series([prev_low]   * n),
    prev_close = pl.Series([prev_close] * n),
)

# Rolling: shift the daily OHLC so each bar sees the prior day
daily = df.group_by_dynamic("date", every="1d").agg(...)
levels = polarticks.pivot_points_floor(
    prev_high  = daily["high"].shift(1),
    prev_low   = daily["low"].shift(1),
    prev_close = daily["close"].shift(1),
)
```

---

## Null-prefix semantics

Every indicator returns exactly as many leading `null` values as its algorithm
requires before it can produce a valid output.  **No zeros are substituted in
the warm-up region.**  This means:

- `sma(n)` → `n − 1` leading nulls
- `ema(n)` → `n − 1` leading nulls
- `rsi(n)` → `n` leading nulls (one extra from the initial `diff`)
- `dema(n)` → `2 × (n − 1)` leading nulls (two EMA passes)

When you attach indicator columns to a DataFrame, Polars propagates nulls
through any downstream arithmetic exactly as you would expect — no silent
zeroes contaminating your signals.

```python
# Safe: Polars propagates nulls through arithmetic
signal = pl.col("rsi") < 30   # null where rsi is null, False otherwise (after fill)

# If you need to fill before a join or export:
rsi = polarticks.rsi(close, 14).fill_null(strategy="forward")
```

---

## API reference

### Moving averages

All moving averages accept `(series: pl.Series, period: int)` unless noted
and return a `pl.Series`.

| Function | Description | Leading nulls |
|---|---|---|
| `sma(series, period)` | Simple Moving Average | `period − 1` |
| `ema(series, period)` | Exponential MA (α = 2/(n+1)) | `period − 1` |
| `wma(series, period)` | Linearly Weighted MA | `period − 1` |
| `wilder_smooth(series, period)` | Wilder's RMA (α = 1/n) | `period − 1` |
| `dema(series, period)` | Double EMA — `2·EMA − EMA(EMA)` | `2·(period−1)` |
| `tema(series, period)` | Triple EMA | `3·(period−1)` |
| `hma(series, period)` | Hull MA — reduces lag | `(period−1) + (√period−1)` |
| `vwma(price, volume, period)` | Volume-Weighted MA | `period − 1` |
| `mcginley_dynamic(series, period)` | Self-adjusting MA | `period − 1` |
| `kama(series, period=10, fast_period=2, slow_period=30)` | Kaufman Adaptive MA | `period − 1` |
| `trix(series, period=14, signal=9)` | Triple EMA oscillator + signal | `3·(period−1)+1` / `3·(period−1)+signal` |
| `zlema(series, period)` | Zero Lag EMA — lag-corrected via shifted series | `lag + (period−1)` where `lag = (period−1)//2` |
| `t3(series, period=5, vfactor=0.7)` | Tillson T3 — 6-pass EMA with binomial blend | `6·(period−1)` |
| `alma(series, period=9, offset=0.85, sigma=6.0)` | Arnaud Legoux MA — Gaussian-weighted | `period − 1` |
| `frama(series, period=16)` | Fractal Adaptive MA — dimension-driven alpha | `period − 1` |
| `laguerre(series, gamma=0.8)` | Laguerre Filter — 4-state low-lag smoother | 0 |

```python
close  = df["close"]
volume = df["volume"]

sma20  = polarticks.sma(close, 20)
ema20  = polarticks.ema(close, 20)
hma20  = polarticks.hma(close, 20)
vwma20 = polarticks.vwma(close, volume, 20)
```

**HMA** is particularly useful when you need low lag without excessive noise.
It runs `WMA(2·WMA(n/2) − WMA(n), √n)` and has a warm-up of
`(n−1) + (⌈√n⌉−1)` bars.

**McGinley Dynamic** self-corrects its smoothing speed based on how fast price
is moving relative to the current indicator value.  Seed is the SMA of the
first `period` bars.

**KAMA** uses the *Efficiency Ratio* — net price change divided by total path
length — to adapt between a fast and slow EMA smoothing constant.  In a strong
trend ER → 1 and KAMA tracks price closely; in chop ER → 0 and KAMA barely
moves, filtering noise.

```python
kama20 = polarticks.kama(close, period=10, fast_period=2, slow_period=30)
```

**TRIX** triple-smoothes price with EMA then computes the 1-period percentage
rate of change, returning a `pl.DataFrame` with `trix_line`, `trix_signal`,
and `trix_histogram` — the same shape as `macd()`.  The three EMA passes filter
out short cycles so TRIX is far less prone to whipsaws than raw ROC.

```python
tx = polarticks.trix(close, period=14, signal=9)
# tx["trix_line"]      — triple-smoothed momentum
# tx["trix_signal"]    — EMA of the TRIX line
# tx["trix_histogram"] — line minus signal
```

**ZLEMA** eliminates most of EMA's lag by feeding an error-corrected input
(`2·close − close.shift(lag)`, where `lag = (period−1)//2`) into a standard
EMA.

**T3** applies six EMA passes and blends the results with configurable binomial
coefficients.  With the default `vfactor=0.7` it produces a line that is
smoother than TEMA while tracking price faster than a plain triple EMA.

**ALMA** weights each bar in the window using a Gaussian bell curve positioned
by `offset` (0 = oldest, 1 = newest) and shaped by `sigma`.  The default
`offset=0.85` keeps the bell close to the current bar for low lag.

---

### Momentum & oscillators

#### `rsi(series, period=14)` → `pl.Series`

Relative Strength Index via Wilder's smoothing.  Values in [0, 100].

```python
rsi = polarticks.rsi(df["close"], 14)
overbought = rsi > 70
oversold   = rsi < 30
```

#### `macd(series, fast=12, slow=26, signal=9)` → `pl.DataFrame`

Returns a DataFrame with three columns: `macd_line`, `macd_signal`,
`macd_histogram`.

```python
m = polarticks.macd(df["close"])
# m["macd_line"]      — fast EMA minus slow EMA
# m["macd_signal"]    — EMA of the MACD line
# m["macd_histogram"] — line minus signal
```

#### `stochastic(ohlc, k_period=14, d_period=3)` → `pl.DataFrame`

Returns `stoch_k` and `stoch_d`.  Both range from 0 to 100.

```python
st = polarticks.stochastic(df)
cross_up = polarticks.crossover(st["stoch_k"], st["stoch_d"])
```

#### `ppo(series, fast=12, slow=26, signal=9)` → `pl.DataFrame`

Percentage Price Oscillator — MACD expressed as a percentage of the slow EMA,
making values comparable across instruments at different price levels.  Returns
`ppo_line`, `ppo_signal`, `ppo_histogram` with the same null-prefix structure
as `macd()`.

```python
p = polarticks.ppo(df["close"])
# p["ppo_line"] == +1.5 means fast EMA is 1.5% above slow EMA
```

#### `stoch_rsi(series, rsi_period=14, stoch_period=14, k_period=3, d_period=3)` → `pl.DataFrame`

Stochastic oscillator applied to RSI values rather than price.  Generates
overbought/oversold signals more frequently than raw RSI.  Returns
`stoch_rsi_k` and `stoch_rsi_d`, both in [0, 100].

Leading nulls: `rsi_period + stoch_period + k_period − 2` for `%K`;
add `d_period − 1` more for `%D`.

```python
srsi = polarticks.stoch_rsi(df["close"])
oversold = srsi["stoch_rsi_k"] < 20
```

#### `williams_r(ohlc, period=14)` → `pl.Series`

Williams %R in the range [−100, 0].  Readings near 0 are overbought; near
−100 are oversold.

#### `cci(ohlc, period=20)` → `pl.Series`

Commodity Channel Index.  Readings above +100 suggest overbought; below −100
suggest oversold.

#### `roc(series, period=10)` → `pl.Series`

Rate of Change as a percentage: `100 × (close − close[n]) / close[n]`.
Has `period` leading nulls (one more than most indicators) because it uses
`shift(period)`.

#### `mfi(ohlcv, period=14)` → `pl.Series`

Money Flow Index — the volume-weighted version of RSI.  Requires a `volume`
column.  Values in [0, 100].

#### `cmf(ohlcv, period=20)` → `pl.Series`

Chaikin Money Flow.  Measures buying vs. selling pressure in the range [−1, 1].
Positive values indicate accumulation.

#### `tsi(series, slow=25, fast=13)` → `pl.Series`

True Strength Index — double-smoothed momentum oscillator.  Values in
(−100, +100).  Signal line: apply `ema(tsi, 7)` to the output.

```python
tsi_vals = polarticks.tsi(df["close"])
signal   = polarticks.ema(tsi_vals.fill_null(0.0), 7)  # fill before second EMA
```

#### `ultimate_oscillator(ohlc, period1=7, period2=14, period3=28)` → `pl.Series`

Weighted blend of three time-frame buying-pressure ratios.  Values in [0, 100].
Warm-up is `period3 − 1` bars.

#### `cmo(series, period=14)` → `pl.Series`

Chande Momentum Oscillator — like RSI but uses the sum of positive changes minus
the sum of negative changes divided by their total.  Bounded in [−100, +100].
`period` leading nulls.

#### `dpo(series, period=20)` → `pl.Series`

Detrended Price Oscillator — subtracts a displaced SMA to remove the dominant
trend, isolating shorter price cycles.  `(period − 1) + (period // 2 + 1)`
leading nulls.

#### `kst(series, roc1=10, roc2=13, roc3=14, roc4=24, sma1=10, sma2=13, sma3=14, sma4=24, signal=9)` → `pl.DataFrame`

Know Sure Thing — weighted sum of four smoothed ROC values across different
time frames.  Returns `kst_line` and `kst_signal`.  Leading nulls for
`kst_line`: `roc4 + sma4 − 1`.

```python
k = polarticks.kst(df["close"])
crossup = polarticks.crossover(k["kst_line"], k["kst_signal"])
```

#### `coppock(series, long_roc=14, short_roc=11, wma_period=10)` → `pl.Series`

Coppock Curve — WMA of the sum of two ROC values, originally designed as a
long-term buy signal for equity indices.  Leading nulls: `long_roc + wma_period − 1`.

#### `awesome_oscillator(ohlc, fast=5, slow=34)` → `pl.Series`

Bill Williams Awesome Oscillator — `SMA(midpoint, fast) − SMA(midpoint, slow)` where
`midpoint = (high + low) / 2`.  Values above zero are bullish.  `slow − 1` leading nulls.

#### `accelerator_oscillator(ohlc, fast=5, slow=34, signal=5)` → `pl.Series`

Accelerator Oscillator — `AO − SMA(AO, signal)`.  Changes direction before AO,
providing an earlier signal.  `slow + signal − 2` leading nulls.

#### `smi(ohlc, period=14, smooth1=3, smooth2=3, signal=9)` → `pl.DataFrame`

Stochastic Momentum Index — double-EMA smoothed stochastic oscillator bounded in [−100, +100].
Returns `smi` and `smi_signal`.

#### `rvi(ohlc, period=10)` → `pl.DataFrame`

Relative Vigor Index — symmetric 4-bar triangular weighted ratio of close-open to high-low,
smoothed over `period` bars.  Returns `rvi` and `rvi_signal`.
Leading nulls: `period + 2` (rvi) and `period + 5` (signal).

#### `bop(ohlc, period=14)` → `pl.Series`

Balance of Power — `(close − open) / (high − low)` optionally SMA-smoothed.
Values near +1 indicate strong buying; near −1 indicate strong selling.
`period − 1` leading nulls (0 when `period=1`).

#### `qqe(series, rsi_period=14, sf=5, qqe_factor=4.236)` → `pl.DataFrame`

Quantitative Qualitative Estimation — RSI-derived adaptive trailing trend line via
double Wilder-smoothed ATR bands.  Returns `qqe_line` (trailing stop) and `qqe_fast`
(smoothed RSI).

---

### Volatility

#### `true_range(ohlc)` → `pl.Series`

Single-bar True Range — the greatest of `H−L`, `|H−prev_C|`, `|L−prev_C|`.
The first bar has no prior close; its TR collapses to `H−L` (Wilder's
convention).  Zero leading nulls.

#### `atr(ohlc, period=14)` → `pl.Series`

Average True Range via Wilder's smoothing.  `period − 1` leading nulls.

#### `bollinger_bands(series, period=20, num_std=2.0)` → `pl.DataFrame`

Returns five columns:

| Column | Description |
|---|---|
| `bb_middle_{period}` | SMA of close |
| `bb_upper_{period}` | Middle + `num_std` × rolling std |
| `bb_lower_{period}` | Middle − `num_std` × rolling std |
| `bb_pct_b_{period}` | Position within the band (0 = lower, 1 = upper) |
| `bb_width_{period}` | (Upper − Lower) / Middle |

```python
bb = polarticks.bollinger_bands(df["close"], 20)
squeeze = bb["bb_width_20"] < bb["bb_width_20"].rolling_mean(20)
```

#### `keltner_channels(ohlc, ema_period=20, atr_period=10, multiplier=2.0)` → `pl.DataFrame`

Returns `kc_middle`, `kc_upper`, `kc_lower`.  Uses ATR for band width rather
than standard deviation, making the channels less reactive to individual large
moves.

#### `chaikin_volatility(ohlc, ema_period=10, roc_period=10)` → `pl.Series`

Rate of change of the EMA of the high-low range.  Rising values signal
increasing volatility.  Leading nulls: `(ema_period − 1) + roc_period`.

#### `historical_volatility(series, period=20, annualise=True, trading_days=252)` → `pl.Series`

Rolling annualised standard deviation of log returns.  Set `trading_days=365`
for crypto or `260` for FX.  `period` leading nulls.

#### `ulcer_index(series, period=14)` → `pl.Series`

Drawdown-based volatility: `√(mean(pct_drawdown², period))`.  Only penalises
downside moves; useful for risk-adjusted metrics like the Ulcer Performance
Index.  Leading nulls: `2 × (period − 1)` (rolling max then rolling mean).

#### `natr(ohlc, period=14)` → `pl.Series`

Normalised Average True Range — ATR divided by the closing price and expressed
as a percentage.  Makes volatility directly comparable across instruments at
different price levels.  `period − 1` leading nulls.

```python
n = polarticks.natr(df, period=14)
# A value of 1.5 means the ATR is 1.5% of the current close.
```

#### `chandelier_exit(ohlc, period=22, multiplier=3.0)` → `pl.DataFrame`

ATR-based dynamic trailing stops for both long and short positions.

| Column | Description |
|---|---|
| `ce_long_{period}` | `highest_high(period) − multiplier × ATR(period)` |
| `ce_short_{period}` | `lowest_low(period) + multiplier × ATR(period)` |

A close below the long exit (or above the short exit) signals a potential
trend reversal.  Leading nulls: `period − 1`.

```python
ce = polarticks.chandelier_exit(df, period=22, multiplier=3.0)
long_stop  = ce["ce_long_22"]
short_stop = ce["ce_short_22"]
```

#### `mass_index(ohlc, ema_period=9, sum_period=25)` → `pl.Series`

Mass Index — the rolling sum of the ratio of a single EMA to a double EMA of
the high-low range.  A "reversal bulge" is traditionally signalled when the
value rises above 27 then falls back below 26.5.  Leading nulls:
`2·(ema_period − 1) + (sum_period − 1)`.

#### `choppiness_index(ohlc, period=14)` → `pl.Series`

Choppiness Index — `100 × log10(ΣTR / HL_range) / log10(period)`.  Values near
100 indicate choppy markets; low values indicate strong trends.  Thresholds:
>61.8 choppy, <38.2 trending.  `period − 1` leading nulls.

#### `squeeze_momentum(ohlc, length=20, bb_mult=2.0, kc_mult=1.5)` → `pl.DataFrame`

TTM Squeeze — detects Bollinger/Keltner compression and measures breakout momentum
via a linear-regression histogram.  Returns `sqz_on` (bool), `sqz_off` (bool),
`sqz_momentum` (float).

#### `volatility_ratio(ohlc, period=14)` → `pl.Series`

Volatility Ratio — `true_range / rolling_max(true_range, period)`.  Values near 1
signal unusually wide-range breakout bars; values near 0 indicate low-volatility bars.
`period − 1` leading nulls.

---

### Trend

#### `donchian_channels(ohlc, period=20)` → `pl.DataFrame`

Returns `dc_upper_{period}`, `dc_lower_{period}`, `dc_middle_{period}`.
Breakout above the upper channel or below the lower channel signals a
trend initiation.

#### `adx(ohlc, period=14)` → `pl.DataFrame`

Average Directional Index with directional components.

| Column | Description |
|---|---|
| `adx_{period}` | Trend strength (0–100; >25 = trending) |
| `plus_di_{period}` | Bullish directional movement |
| `minus_di_{period}` | Bearish directional movement |

Leading nulls: `period − 1` for the DI columns; `2 × (period − 1)` for ADX
(it is Wilder-smoothed DX, which is itself Wilder-smoothed).

```python
result = polarticks.adx(df, 14)
trending    = result["adx_14"] > 25
bull_trend  = result["plus_di_14"] > result["minus_di_14"]
```

#### `supertrend(ohlc, period=7, multiplier=3.0)` → `pl.DataFrame`

ATR-based trailing stop that also indicates trend direction.

| Column | Description |
|---|---|
| `supertrend` | Band level (support in uptrend, resistance in downtrend) |
| `supertrend_direction` | `+1` (bullish) or `−1` (bearish) |

```python
st = polarticks.supertrend(df, period=10, multiplier=2.0)
entries = st["supertrend_direction"].diff() == 2   # flipped to bullish
```

#### `aroon(ohlc, period=25)` → `pl.DataFrame`

Aroon Indicator — measures how recently within a rolling window the highest
high and lowest low occurred, quantifying trend freshness.

| Column | Value | Meaning |
|---|---|---|
| `aroon_up_{period}` | 100 | New high on the current bar |
| `aroon_up_{period}` | 0 | High was `period` bars ago |
| `aroon_down_{period}` | 100 | New low on the current bar |
| `aroon_osc_{period}` | +100 to −100 | Up minus Down |

Leading nulls: `period` bars (window size is `period + 1`).

```python
a = polarticks.aroon(df, period=25)
bullish = a["aroon_up_25"] > 70
bearish = a["aroon_down_25"] > 70
```

#### `vortex(ohlc, period=14)` → `pl.DataFrame`

Vortex Indicator — compares upward and downward price movements to the
True Range to produce two oscillating directional lines.

| Column | Description |
|---|---|
| `vi_plus_{period}` | Positive Vortex Movement / TR sum |
| `vi_minus_{period}` | Negative Vortex Movement / TR sum |

When VI+ crosses above VI− it signals an uptrend; a cross below signals a
downtrend.  Leading nulls: `period − 1`.

```python
v = polarticks.vortex(df, period=14)
trend_up = polarticks.crossover(v["vi_plus_14"], v["vi_minus_14"])
```

#### `parabolic_sar(ohlc, initial_af=0.02, step_af=0.02, max_af=0.20)` → `pl.DataFrame`

Parabolic SAR dot plot.  Returns `psar` (price level) and `psar_direction`
(`+1` uptrend / `−1` downtrend).  One leading null (initialised from bar 1).

```python
sar = polarticks.parabolic_sar(df)
flip_to_bull = sar["psar_direction"].diff() == 2
```

#### `ichimoku(ohlc, tenkan_period=9, kijun_period=26, senkou_b_period=52, chikou_period=26)` → `pl.DataFrame`

The Ichimoku Cloud system in a single call.  Returns five components:

| Column | Formula | Leading nulls |
|---|---|---|
| `tenkan_sen` | `(HH(9) + LL(9)) / 2` | `tenkan_period − 1` |
| `kijun_sen` | `(HH(26) + LL(26)) / 2` | `kijun_period − 1` |
| `senkou_span_a` | `(tenkan + kijun) / 2` | `kijun_period − 1` |
| `senkou_span_b` | `(HH(52) + LL(52)) / 2` | `senkou_b_period − 1` |
| `chikou_span` | `close.shift(−chikou_period)` | 0 leading, `chikou_period` trailing |

All components are returned at the current bar without any forward/backward
shift.  To display the cloud `kijun_period` bars ahead (the standard chart
convention), apply `.shift(kijun_period)` to the senkou columns.

```python
ichi = polarticks.ichimoku(df)

# Bullish TK cross
tk_cross = polarticks.crossover(ichi["tenkan_sen"], ichi["kijun_sen"])

# Price vs cloud: check if close is above both span edges
above_cloud = (df["close"] > ichi["senkou_span_a"]) & (df["close"] > ichi["senkou_span_b"])
```

#### `linreg_slope(series, period=14)` → `pl.Series`

OLS slope of a rolling linear regression line.  Positive values indicate an
uptrend; the magnitude reflects steepness.  `period − 1` leading nulls.

```python
slope = polarticks.linreg_slope(df["close"], period=14)
accelerating = slope > slope.shift(1)
```

#### `stc(ohlc, fast=23, slow=50, stoch_period=10, smooth=3)` → `pl.Series`

Schaff Trend Cycle — applies a double stochastic to the MACD line for faster
cycle detection.  Values are clipped to [0, 100]; readings above 75 suggest an
uptrend and below 25 a downtrend.

#### `alligator(ohlc, jaw_period=13, jaw_offset=8, teeth_period=8, teeth_offset=5, lips_period=5, lips_offset=3)` → `pl.DataFrame`

Bill Williams Alligator — three Wilder-smoothed median-price lines displaced into
the future.  Returns `jaw`, `teeth`, `lips`.  When `lips > teeth > jaw` the market
is bullish; intertwined lines indicate a sleeping (choppy) market.

#### `fractal(ohlc)` → `pl.DataFrame`

Williams Fractal — 5-bar pivot high/low detector.  A bearish fractal marks a bar
whose high is strictly greater than both neighbours; a bullish fractal marks the
lowest low.  Returns bool columns `fractal_bearish` and `fractal_bullish`.

#### `linreg_channel(series, period=100, num_std=2.0)` → `pl.DataFrame`

Rolling linear regression channel with RMSE-based bands.  Returns `lrc_mid`
(fitted line at the end of the window), `lrc_upper`, and `lrc_lower`.
`period − 1` leading nulls.

#### `tsf(series, period=14)` → `pl.Series`

Time Series Forecast — OLS line projected one bar ahead: `linreg_value + slope`.
`period − 1` leading nulls.

#### `chande_kroll_stop(ohlc, atr_period=10, atr_mult=1.5, stop_period=9)` → `pl.DataFrame`

Two-stage ATR trailing stop.  Returns `cks_long` and `cks_short`.  A close above
`cks_long` is bullish; a close below `cks_short` is bearish.
Leading nulls: `atr_period + stop_period − 2`.

#### `elder_ray(ohlc, period=13)` → `pl.DataFrame`

Elder Ray Index — measures market force by splitting it into two components:

- **Bull Power** = `high − EMA(close, period)` — how far bulls push price above the consensus level
- **Bear Power** = `low − EMA(close, period)` — how far bears push price below it

Bull Power > 0 and rising indicates strengthening bulls; Bear Power < 0 and rising (becoming less negative) indicates weakening bears.  `period − 1` leading nulls.

```python
er = polarticks.elder_ray(df, period=13)
# er["bull_power"], er["bear_power"]

# Classic Elder entry: EMA trending up, bear_power < 0 but rising
ema_rising = polarticks.ema(df["close"], 13) > polarticks.ema(df["close"], 13).shift(1)
bear_diverging = er["bear_power"] > er["bear_power"].shift(1)
entry = ema_rising & (er["bear_power"] < 0) & bear_diverging
```

---

### Volume

#### `ad_line(ohlcv)` → `pl.Series`

Accumulation/Distribution Line — OBV variant that weights each bar's volume
contribution by the position of the close within the high-low range.

```
money_flow_multiplier = (2 × close − high − low) / (high − low)
A/D[t] = A/D[t-1] + multiplier × volume
```

A rising A/D line confirms an uptrend; divergence from price signals
weakening participation.  No leading nulls — starts accumulating from bar 0.

```python
ad = polarticks.ad_line(df)
divergence = (df["close"] > df["close"].shift(20)) & (ad < ad.shift(20))
```

#### `obv(ohlcv)` → `pl.Series`

On-Balance Volume — running cumulative sum of signed volume.  Volume is added
on up-bars and subtracted on down-bars.  No leading nulls; starts accumulating
from bar 0.

```python
obv = polarticks.obv(df)
obv_trend = polarticks.ema(obv, 20)   # smooth OBV to spot divergences
```

#### `vwap(ohlcv, session_start_hour=22)` → `pl.Series`

Session-anchored VWAP.  If your DataFrame has a `time` column (Polars
`Datetime`), a new session is started at every bar whose UTC hour equals
`session_start_hour`.  Without a `time` column the entire series is treated
as one session.  No leading nulls.

```python
vwap = polarticks.vwap(df, session_start_hour=0)   # midnight UTC sessions
above_vwap = df["close"] > vwap
```

#### `vwap_bands(ohlcv, session_start_hour=22)` → `pl.DataFrame`

VWAP with ±1σ and ±2σ volume-weighted standard-deviation bands.  Returns
`vwap`, `upper_1`, `lower_1`, `upper_2`, `lower_2`.  No leading nulls.

#### `kvo(ohlcv, fast=34, slow=55, signal=13)` → `pl.DataFrame`

Klinger Volume Oscillator — EMA difference of a signed volume-force series that
tracks cumulative movement and trend direction.  Returns `kvo_line` and
`kvo_signal`.  Leading nulls for the line: `slow − 1`.

#### `eom(ohlcv, period=14, divisor=10_000.0)` → `pl.Series`

Ease of Movement — compares midpoint displacement to the "box ratio"
(volume / range).  Low absolute values indicate price is moving easily on light
volume; high values indicate effort.  `period` leading nulls.

#### `pvt(ohlcv)` → `pl.Series`

Price Volume Trend — cumulative sum of volume scaled by the bar's percentage
price change.  Similar to OBV but uses the magnitude of the move rather than
just its sign.  No leading nulls.

#### `force_index(ohlcv, period=13)` → `pl.Series`

Elder's Force Index — EMA of `(close − prev_close) × volume`.  Combines the
direction and magnitude of a price move with its volume to measure buying or
selling force.  Positive values indicate buying pressure; negative values
indicate selling pressure.  `period − 1` leading nulls.

```python
fi = polarticks.force_index(df, period=13)
```

#### `nvi(ohlcv)` → `pl.Series`

Negative Volume Index — accumulates price change only on bars where volume is
*lower* than the prior bar, tracking what the "smart money" does on quiet days.
Starts at 1000; no leading nulls.

#### `pvi(ohlcv)` → `pl.Series`

Positive Volume Index — accumulates price change only on bars where volume is
*higher* than the prior bar, tracking crowd activity on busy days.
Starts at 1000; no leading nulls.

```python
nvi = polarticks.nvi(df)
pvi = polarticks.pvi(df)
# When NVI is above its 255-bar EMA, the smart-money trend is up.
nvi_signal = polarticks.ema(nvi, 255)
bull_regime = nvi > nvi_signal
```

#### `chaikin_osc(ohlcv, fast=3, slow=10)` → `pl.Series`

Chaikin Oscillator — `EMA(AD_Line, fast) − EMA(AD_Line, slow)`.  Measures momentum
of money flow.  `slow − 1` leading nulls.

#### `volume_oscillator(volume, fast=5, slow=10)` → `pl.Series`

Volume Oscillator — `100 × (EMA(vol, fast) − EMA(vol, slow)) / EMA(vol, slow)`.
Positive values confirm volume-backed price moves.  `slow − 1` leading nulls.

#### `twap(ohlcv, period=None)` → `pl.Series`

Time-Weighted Average Price of `(high + low + close) / 3`.  With `period=None`
returns the cumulative mean from bar 0 (0 leading nulls).  With a period, returns
a rolling SMA (`period − 1` leading nulls).

---

### Levels (pivot points)

All pivot point functions accept `pl.Series` arguments (one per price
component) and return a `pl.DataFrame`.  Pass yesterday's values — scalar
broadcast or a shifted daily series — aligned to your current-session bars.

#### `pivot_points_floor(prev_high, prev_low, prev_close)` → `pl.DataFrame`

Classic floor-trader pivots.  Columns: `pp`, `r1`, `r2`, `r3`, `s1`, `s2`, `s3`.

#### `pivot_points_camarilla(prev_high, prev_low, prev_close)` → `pl.DataFrame`

Camarilla equation (multiplier 1.1).  Produces tighter intraday levels suited
to mean-reversion scalping.  Columns: `cam_r1`–`cam_r4`, `cam_s1`–`cam_s4`.

#### `pivot_points_fibonacci(prev_high, prev_low, prev_close)` → `pl.DataFrame`

Fibonacci-ratio levels (0.382, 0.618, 1.000 × range).
Columns: `fib_pp`, `fib_r1`–`fib_r3`, `fib_s1`–`fib_s3`.

#### `pivot_points_woodie(prev_high, prev_low, prev_close)` → `pl.DataFrame`

Double-weights the prior close.  Columns: `wood_pp`, `wood_r1`, `wood_r2`,
`wood_s1`, `wood_s2`.

#### `pivot_points_demark(prev_open, prev_high, prev_low, prev_close)` → `pl.DataFrame`

Adapts the formula based on whether the prior session closed above, below, or
equal to its open.  Returns a single resistance and support level.
Columns: `dm_pp`, `dm_r1`, `dm_s1`.

#### `fibonacci_retracement(high, low)` → `pl.DataFrame`

Computes the seven standard Fibonacci retracement levels from a high-low range.
Accepts any `pl.Series` pair — typically the outputs of `rolling_highest` and
`rolling_lowest`.

Columns: `fib_0` (0%), `fib_236` (23.6%), `fib_382` (38.2%), `fib_500` (50%),
`fib_618` (61.8%), `fib_786` (78.6%), `fib_100` (100%).

```python
highs = polarticks.rolling_highest(df["high"], period=20)
lows  = polarticks.rolling_lowest(df["low"],  period=20)
fibs  = polarticks.fibonacci_retracement(highs, lows)
# fibs["fib_618"] — the golden-ratio support/resistance level
```

```python
n = len(df)
levels = polarticks.pivot_points_floor(
    pl.Series([yesterday_high]  * n),
    pl.Series([yesterday_low]   * n),
    pl.Series([yesterday_close] * n),
)
df = df.with_columns(levels.get_columns())
```

---

### Candlestick patterns

All pattern functions accept a `pl.DataFrame` with `open`, `high`, `low`,
`close` columns and return a Boolean `pl.Series` — `True` on bars where the
pattern is present, `False` everywhere else (including leading bars that
cannot satisfy the look-back requirement).

#### Single-bar patterns

| Function | Description |
|---|---|
| `is_doji(ohlc, threshold=0.1)` | Body is < 10% of the bar's range |
| `is_pin_bar_bullish(ohlc, wick_ratio=0.6, body_ratio=0.25)` | Hammer: small body, long lower wick |
| `is_pin_bar_bearish(ohlc, wick_ratio=0.6, body_ratio=0.25)` | Shooting star: small body, long upper wick |

#### Two-bar patterns

| Function | Description |
|---|---|
| `is_bullish_engulfing(ohlc)` | Bearish bar followed by a larger bullish bar that engulfs it |
| `is_bearish_engulfing(ohlc)` | Bullish bar followed by a larger bearish bar that engulfs it |
| `is_inside_bar(ohlc)` | Current bar's range is entirely within the prior bar's range |
| `is_bullish_harami(ohlc)` | Small bullish body inside a large prior bearish body |
| `is_bearish_harami(ohlc)` | Small bearish body inside a large prior bullish body |
| `is_hanging_man(ohlc, wick_ratio=0.6, body_ratio=0.25, trend_period=5)` | Hammer shape after uptrend — potential bearish reversal |
| `is_inverted_hammer(ohlc, wick_ratio=0.6, body_ratio=0.25, trend_period=5)` | Shooting-star shape after downtrend — potential bullish reversal |
| `is_tweezer_top(ohlc, tolerance=0.001, body_ratio=0.3)` | Two bars with equal highs — bearish rejection |
| `is_tweezer_bottom(ohlc, tolerance=0.001, body_ratio=0.3)` | Two bars with equal lows — bullish support |
| `is_dark_cloud_cover(ohlc, penetration=0.5)` | Bearish bar opens above prior high, closes inside prior body |
| `is_piercing_line(ohlc, penetration=0.5)` | Bullish bar opens below prior low, closes inside prior body |

#### Three-bar patterns

| Function | Description |
|---|---|
| `is_three_white_soldiers(ohlc, body_ratio=0.5)` | Three consecutive advancing bullish candles |
| `is_three_black_crows(ohlc, body_ratio=0.5)` | Three consecutive declining bearish candles |
| `is_morning_star(ohlc, body_ratio=0.3, star_body_ratio=0.15)` | Bearish → small star → bullish reversal |
| `is_evening_star(ohlc, body_ratio=0.3, star_body_ratio=0.15)` | Bullish → small star → bearish reversal |
| `is_abandoned_baby_bullish(ohlc, body_ratio=0.3, doji_ratio=0.1)` | Bearish bar → gapped-down doji → bullish bar with gap up |
| `is_abandoned_baby_bearish(ohlc, body_ratio=0.3, doji_ratio=0.1)` | Bullish bar → gapped-up doji → bearish bar with gap down |

#### Five-bar patterns

| Function | Description |
|---|---|
| `is_rising_three_methods(ohlc, body_ratio=0.3, small_body_ratio=0.3)` | Large bull → 3 small bears → large bull (bullish continuation) |
| `is_falling_three_methods(ohlc, body_ratio=0.3, small_body_ratio=0.3)` | Large bear → 3 small bulls → large bear (bearish continuation) |

```python
# Combine patterns and indicators for a signal
bull_signals = (
    polarticks.is_bullish_engulfing(df)
    | polarticks.is_morning_star(df)
    | polarticks.is_pin_bar_bullish(df)
)

df = df.with_columns([
    bull_signals.alias("bull_pattern"),
    polarticks.rsi(df["close"], 14).alias("rsi"),
])

entries = df.filter(pl.col("bull_pattern") & (pl.col("rsi") < 40))
```

---

### Utilities

#### `crossover(fast, slow, atol=0.0)` → `pl.Series[bool]`

`True` on the single bar where `fast` crosses above `slow`.  The optional
`atol` prevents double-signals from floating-point noise right at the crossing
price.

#### `crossunder(fast, slow, atol=0.0)` → `pl.Series[bool]`

`True` on the single bar where `fast` crosses below `slow`.

```python
fast = polarticks.ema(close, 9)
slow = polarticks.ema(close, 21)

long_entry  = polarticks.crossover(fast, slow)
short_entry = polarticks.crossunder(fast, slow)

# Noise-tolerant version for choppy markets
long_entry  = polarticks.crossover(fast, slow, atol=0.05)
```

#### `log_returns(series)` → `pl.Series`

Bar-to-bar log returns: `ln(price[t] / price[t-1])`.  One leading null.

#### `simple_returns(series)` → `pl.Series`

Bar-to-bar simple (arithmetic) returns.  One leading null.

#### `rolling_highest(series, period)` → `pl.Series`

Rolling n-period maximum.  `period − 1` leading nulls.  Useful as a building
block for indicators that need the recent high (e.g. Fibonacci retracement,
Williams VIX Fix, Donchian breakout signals).

#### `rolling_lowest(series, period)` → `pl.Series`

Rolling n-period minimum.  `period − 1` leading nulls.

#### `rolling_std(series, period)` → `pl.Series`

Rolling n-period sample standard deviation (ddof=1).  `period − 1` leading
nulls.  Requires `period ≥ 2`.

#### `percent_rank(series, period)` → `pl.Series`

Rolling percentile rank — the fraction of the last *period* bars whose value is
≤ the current bar, scaled to [0, 100].  A value of 100 means the current bar is
the highest in the window; 0 means it is the lowest.  `period − 1` leading nulls.

```python
# Combine with RSI to find historically extreme readings
rsi_rank = polarticks.percent_rank(polarticks.rsi(df["close"], 14), period=252)
historically_oversold = rsi_rank < 10   # RSI in bottom decile of past year
```

#### `rolling_zscore(series, period)` → `pl.Series`

Rolling Z-score — `(value − rolling_mean) / rolling_std`.  `period − 1` leading nulls.
Null where the window is constant (zero standard deviation).

#### `rolling_beta(series, benchmark, period)` → `pl.Series`

Rolling OLS beta — sensitivity of the series' log returns to the benchmark.
Beta > 1: amplified moves; 0 < β < 1: dampened; β < 0: inverse.
`period` leading nulls (one extra from the log-return diff).

#### `hurst_exponent(series, period=100)` → `pl.Series`

Rolling Hurst Exponent via rescaled range (R/S) analysis.  H > 0.5 indicates a
trending regime; H = 0.5 a random walk; H < 0.5 mean-reversion.
`period` leading nulls.  Requires `period ≥ 10`.

---

### New volatility estimators (v0.3.0)

These estimators use intrabar (OHLC) data for more efficient volatility
measurement than the close-to-close `historical_volatility`.

#### `parkinson(ohlc, period=20, annualise=True, trading_days=252)` → `pl.Series`

Parkinson (1980) estimator — uses the log ratio of high to low.  5× more
efficient than close-to-close HV but ignores overnight gaps and drift.

#### `garman_klass(ohlc, period=20, annualise=True, trading_days=252)` → `pl.Series`

Garman-Klass (1980) estimator — adds an open-to-close drift correction to
Parkinson.  More efficient but still assumes no overnight gaps.

#### `yang_zhang(ohlc, period=20, annualise=True, trading_days=252)` → `pl.Series`

Yang-Zhang (2000) estimator — accounts for overnight gaps, open jumps, and
intrabar drift.  The most efficient unbiased OHLC estimator.  Has `period`
leading nulls (one extra vs. the others, due to the overnight shift).

```python
# Compare estimators side-by-side
pk = polarticks.parkinson(df, period=20)
gk = polarticks.garman_klass(df, period=20)
yz = polarticks.yang_zhang(df, period=20)
```

#### `williams_vix_fix(ohlc, period=22)` → `pl.Series`

Synthetic fear gauge — `100 × (rolling_max(close, period) − low) / rolling_max(close, period)`.
Spikes during sharp selloffs, mimicking the shape of the CBOE VIX without
requiring options data.

#### `fisher_transform(ohlc, period=9)` → `pl.DataFrame`

Fisher Transform (Ehlers 2002) — normalises the HL midpoint to a near-Gaussian
distribution via arctanh.  Extreme readings are statistically significant
turning-point signals.  Returns `fisher` and `fisher_signal` columns.

```python
ft = polarticks.fisher_transform(df, period=9)
# ft["fisher"]        — arctanh-normalised price momentum
# ft["fisher_signal"] — fisher shifted by 1 bar
```

---

## Running tests

```bash
uv run pytest tests/unit/       # 803 unit tests
uv run pytest tests/            # all tests (includes benchmarks — takes ~90 s)
```

To run only the null-prefix consistency audit:

```bash
uv run pytest tests/unit/test_null_prefix.py -v
```

Type-check the package:

```bash
uv run mypy src/polarticks/ --strict
```

---

## Running benchmarks

The benchmark suite exercises every indicator on a 100 000-bar OHLCV series:

```bash
uv run pytest tests/benchmark/ --benchmark-only
```

Save a baseline and compare across changes:

```bash
uv run pytest tests/benchmark/ --benchmark-only --benchmark-save=baseline
# ... make changes ...
uv run pytest tests/benchmark/ --benchmark-only --benchmark-compare=baseline
```
