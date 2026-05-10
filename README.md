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

#### Three-bar patterns

| Function | Description |
|---|---|
| `is_three_white_soldiers(ohlc, body_ratio=0.5)` | Three consecutive advancing bullish candles |
| `is_three_black_crows(ohlc, body_ratio=0.5)` | Three consecutive declining bearish candles |
| `is_morning_star(ohlc, body_ratio=0.3, star_body_ratio=0.15)` | Bearish → small star → bullish reversal |
| `is_evening_star(ohlc, body_ratio=0.3, star_body_ratio=0.15)` | Bullish → small star → bearish reversal |

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

---

## Running tests

```bash
uv run pytest tests/unit/       # 365 unit tests
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
