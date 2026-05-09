"""Null-prefix consistency audit.

Verifies that every indicator returns exactly the expected number of
leading nulls and produces no accidental zero values in the warm-up
region where those nulls are expected.

Expected-null formulas used throughout:
    period - 1          single rolling window of size *period*
    2 * (period - 1)    two sequential rolling windows (DEMA, Ulcer, ADX)
    3 * (period - 1)    three sequential rolling windows (TEMA)
    period              shift-based or diff-seeded indicators (RSI, ROC, HV)
    slow + fast - 1     double-EMA chain seeded by diff (TSI)
    (ema_p - 1) + roc_p two-stage EMA-then-shift (Chaikin Volatility)
    k_period - 1        first stage of stochastic (%K)
    (k_p - 1) + (d_p-1) cascaded stochastic (%D)
    (slow - 1)+(sig-1)  MACD signal / histogram
    period3 - 1         Ultimate Oscillator (driven by longest window)
    (period - 1) + (round(√period) - 1)  HMA
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import polarticks
from polarticks.moving_averages import mcginley_dynamic, wilder_smooth
from polarticks.volatility import true_range

# ---------------------------------------------------------------------------
# Synthetic OHLCV fixture
# ---------------------------------------------------------------------------

_N = 60

_closes = [100.0 + math.sin(i * 0.3) * 10 + i * 0.5 for i in range(_N)]
_highs = [c + abs(math.cos(i * 0.4)) * 2 + 0.5 for i, c in enumerate(_closes)]
_lows = [c - abs(math.cos(i * 0.4)) * 2 - 0.5 for i, c in enumerate(_closes)]
_opens = [c + math.sin(i * 0.2) * 1.5 for i, c in enumerate(_closes)]
_volumes = [1000.0 + math.sin(i * 0.5) * 300 for i in range(_N)]

_DF = pl.DataFrame(
    {
        "open": _opens,
        "high": _highs,
        "low": _lows,
        "close": _closes,
        "volume": _volumes,
    }
)

_CLOSE = _DF["close"]
_VOLUME = _DF["volume"]

_P = 5  # Standard test period used throughout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _leading_nulls(series: pl.Series) -> int:
    """Return the count of leading null values in *series*."""
    for idx, value in enumerate(series.to_list()):
        if value is not None:
            return idx
    return len(series)


def _leading_nulls_col(df: pl.DataFrame, col: str) -> int:
    """Return the leading-null count for column *col* in *df*."""
    return _leading_nulls(df[col])


def _no_accidental_zeros(series: pl.Series, expected_nulls: int) -> bool:
    """Return True when none of the expected-null positions hold a 0.0 value.

    A 0.0 appearing where None is expected indicates the warm-up window was
    filled with a sentinel rather than propagating null.
    """
    if expected_nulls == 0:
        return True
    prefix = series.head(expected_nulls).to_list()
    return not any(v == 0.0 for v in prefix)


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


class TestSMANullPrefix:
    """SMA: single rolling window → period - 1 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.sma(_CLOSE, _P)) == _P - 1

    def test_no_accidental_zeros(self) -> None:
        assert _no_accidental_zeros(polarticks.sma(_CLOSE, _P), _P - 1)


class TestEMANullPrefix:
    """EMA: ewm seeded at period - 1 → period - 1 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.ema(_CLOSE, _P)) == _P - 1

    def test_no_accidental_zeros(self) -> None:
        assert _no_accidental_zeros(polarticks.ema(_CLOSE, _P), _P - 1)


class TestWMANullPrefix:
    """WMA: shift-based weighted sum; null propagates naturally → period - 1 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.wma(_CLOSE, _P)) == _P - 1

    def test_no_accidental_zeros(self) -> None:
        assert _no_accidental_zeros(polarticks.wma(_CLOSE, _P), _P - 1)


class TestWilderSmoothNullPrefix:
    """Wilder smooth (RMA): ewm with α = 1/period → period - 1 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(wilder_smooth(_CLOSE, _P)) == _P - 1


class TestDEMANullPrefix:
    """DEMA: two EMA passes → 2 * (period - 1) leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.dema(_CLOSE, _P)) == 2 * (_P - 1)


class TestTEMANullPrefix:
    """TEMA: three EMA passes → 3 * (period - 1) leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.tema(_CLOSE, _P)) == 3 * (_P - 1)


class TestHMANullPrefix:
    """HMA: WMA(n) warm-up then WMA(√n) warm-up → (n-1) + (√n-1) leading nulls."""

    def test_null_count(self) -> None:
        sqrt_p = round(_P**0.5)
        expected = (_P - 1) + (sqrt_p - 1)
        assert _leading_nulls(polarticks.hma(_CLOSE, _P)) == expected


class TestVWMANullPrefix:
    """VWMA: single rolling window → period - 1 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.vwma(_CLOSE, _VOLUME, _P)) == _P - 1

    def test_no_accidental_zeros(self) -> None:
        assert _no_accidental_zeros(polarticks.vwma(_CLOSE, _VOLUME, _P), _P - 1)


class TestMcginleyDynamicNullPrefix:
    """McGinley Dynamic: Python-loop seeded at period - 1 → period - 1 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(mcginley_dynamic(_CLOSE, _P)) == _P - 1


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


class TestRSINullPrefix:
    """RSI: diff(1) contributes 1 null; wilder_smooth then needs period
    non-null samples → period leading nulls total."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.rsi(_CLOSE, _P)) == _P


class TestMACDNullPrefix:
    """MACD: macd_line limited by slow EMA; signal cascades a further signal-1."""

    _FAST = 5
    _SLOW = 10
    _SIGNAL = 3

    @pytest.fixture()
    def _df(self) -> pl.DataFrame:
        return polarticks.macd(_CLOSE, fast=self._FAST, slow=self._SLOW, signal=self._SIGNAL)

    def test_macd_line_null_count(self, _df: pl.DataFrame) -> None:
        assert _leading_nulls_col(_df, "macd_line") == self._SLOW - 1

    def test_signal_null_count(self, _df: pl.DataFrame) -> None:
        assert _leading_nulls_col(_df, "macd_signal") == (self._SLOW - 1) + (self._SIGNAL - 1)

    def test_histogram_null_count(self, _df: pl.DataFrame) -> None:
        assert _leading_nulls_col(_df, "macd_histogram") == (self._SLOW - 1) + (self._SIGNAL - 1)


class TestStochasticNullPrefix:
    """Stochastic: %K is one rolling window; %D adds a further SMA pass."""

    _K = 5
    _D = 3

    @pytest.fixture()
    def _df(self) -> pl.DataFrame:
        return polarticks.stochastic(_DF, k_period=self._K, d_period=self._D)

    def test_stoch_k_null_count(self, _df: pl.DataFrame) -> None:
        assert _leading_nulls_col(_df, "stoch_k") == self._K - 1

    def test_stoch_d_null_count(self, _df: pl.DataFrame) -> None:
        assert _leading_nulls_col(_df, "stoch_d") == (self._K - 1) + (self._D - 1)


class TestWilliamsRNullPrefix:
    """Williams %R: single rolling max/min → period - 1 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.williams_r(_DF, _P)) == _P - 1


class TestCCINullPrefix:
    """CCI: rolling mean and MAD share the same window → period - 1 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.cci(_DF, _P)) == _P - 1


class TestROCNullPrefix:
    """ROC: implemented as shift(period) → period leading nulls (not period - 1)."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.roc(_CLOSE, _P)) == _P


class TestMFINullPrefix:
    """MFI: first bar's money flow is zeroed (no prior close), so rolling sum
    counts from bar 0 → period - 1 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.mfi(_DF, _P)) == _P - 1


class TestCMFNullPrefix:
    """CMF: single rolling sum/division → period - 1 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.cmf(_DF, _P)) == _P - 1


class TestTSINullPrefix:
    """TSI: diff(1) + two nested EMA passes → slow + fast - 1 leading nulls."""

    _SLOW = 5
    _FAST = 3

    def test_null_count(self) -> None:
        expected = self._SLOW + self._FAST - 1
        assert _leading_nulls(polarticks.tsi(_CLOSE, slow=self._SLOW, fast=self._FAST)) == expected


class TestUltimateOscillatorNullPrefix:
    """UO: min/max_horizontal ignores nulls so buying pressure starts from bar 0;
    the longest rolling window (period3) drives the warm-up → period3 - 1 nulls."""

    _P1, _P2, _P3 = 7, 14, 28

    def test_null_count(self) -> None:
        result = polarticks.ultimate_oscillator(_DF, self._P1, self._P2, self._P3)
        assert _leading_nulls(result) == self._P3 - 1


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


class TestTrueRangeNullPrefix:
    """True Range: gap components filled with 0.0 on bar 0 → 0 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(true_range(_DF)) == 0


class TestATRNullPrefix:
    """ATR: Wilder smooth of true_range → period - 1 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.atr(_DF, _P)) == _P - 1


class TestBollingerBandsNullPrefix:
    """Bollinger Bands: all columns share one rolling window → period - 1 nulls."""

    def test_null_count_all_columns(self) -> None:
        df = polarticks.bollinger_bands(_CLOSE, _P)
        for col in df.columns:
            assert _leading_nulls_col(df, col) == _P - 1, f"unexpected nulls in {col}"


class TestKeltnerChannelsNullPrefix:
    """Keltner Channels: EMA and ATR share the same period → period - 1 nulls."""

    def test_null_count_all_columns(self) -> None:
        df = polarticks.keltner_channels(_DF, ema_period=_P, atr_period=_P)
        for col in df.columns:
            assert _leading_nulls_col(df, col) == _P - 1, f"unexpected nulls in {col}"


class TestChaikinVolatilityNullPrefix:
    """Chaikin Volatility: EMA warm-up then shift → (ema_period - 1) + roc_period nulls."""

    _EMA_P = 5
    _ROC_P = 5

    def test_null_count(self) -> None:
        expected = (self._EMA_P - 1) + self._ROC_P
        result = polarticks.chaikin_volatility(_DF, ema_period=self._EMA_P, roc_period=self._ROC_P)
        assert _leading_nulls(result) == expected


class TestHistoricalVolatilityNullPrefix:
    """Historical Volatility: log-return creates 1 null; rolling_std then needs
    period non-null samples → period leading nulls total."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.historical_volatility(_CLOSE, _P)) == _P


class TestUlcerIndexNullPrefix:
    """Ulcer Index: rolling_max then rolling_mean — two sequential windows of
    the same size → 2 * (period - 1) leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.ulcer_index(_CLOSE, _P)) == 2 * (_P - 1)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


class TestDonchianChannelsNullPrefix:
    """Donchian Channels: rolling max/min → period - 1 leading nulls."""

    def test_null_count_all_columns(self) -> None:
        df = polarticks.donchian_channels(_DF, _P)
        for col in df.columns:
            assert _leading_nulls_col(df, col) == _P - 1, f"unexpected nulls in {col}"


class TestADXNullPrefix:
    """ADX: +DI/-DI share one Wilder pass; ADX is a second Wilder pass over DX.

    +DI, -DI: period - 1 leading nulls.
    ADX:      2 * (period - 1) leading nulls.
    """

    def test_plus_di_null_count(self) -> None:
        df = polarticks.adx(_DF, _P)
        assert _leading_nulls_col(df, f"plus_di_{_P}") == _P - 1

    def test_minus_di_null_count(self) -> None:
        df = polarticks.adx(_DF, _P)
        assert _leading_nulls_col(df, f"minus_di_{_P}") == _P - 1

    def test_adx_null_count(self) -> None:
        df = polarticks.adx(_DF, _P)
        assert _leading_nulls_col(df, f"adx_{_P}") == 2 * (_P - 1)


class TestSupertrendNullPrefix:
    """Supertrend: follows ATR warm-up → period - 1 leading nulls.

    The initialisation branch fires at ``idx == period - 1`` (the first bar
    with a valid ATR); both output arrays remain null only for the warm-up
    span (0 .. period - 2).
    """

    def test_null_count_band(self) -> None:
        df = polarticks.supertrend(_DF, _P)
        assert _leading_nulls_col(df, "supertrend") == _P - 1

    def test_null_count_direction(self) -> None:
        df = polarticks.supertrend(_DF, _P)
        assert _leading_nulls_col(df, "supertrend_direction") == _P - 1

    def test_no_accidental_zeros_in_band(self) -> None:
        df = polarticks.supertrend(_DF, _P)
        assert _no_accidental_zeros(df["supertrend"], _P - 1)


class TestParabolicSARNullPrefix:
    """PSAR: initialised from bar 1; bar 0 is always null → 1 leading null."""

    def test_null_count(self) -> None:
        df = polarticks.parabolic_sar(_DF)
        assert _leading_nulls_col(df, "psar") == 1
        assert _leading_nulls_col(df, "psar_direction") == 1


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class TestOBVNullPrefix:
    """OBV: cumulative sum from bar 0 with no warm-up → 0 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.obv(_DF)) == 0


class TestVWAPNullPrefix:
    """VWAP: session-anchored from the first bar → 0 leading nulls."""

    def test_null_count(self) -> None:
        assert _leading_nulls(polarticks.vwap(_DF)) == 0

    def test_vwap_bands_null_count(self) -> None:
        df = polarticks.vwap_bands(_DF)
        for col in df.columns:
            assert _leading_nulls_col(df, col) == 0, f"unexpected nulls in {col}"
