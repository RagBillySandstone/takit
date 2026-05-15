"""Unit tests for mass_index, kvo, eom, and pvt in polarticks."""

from __future__ import annotations

import math

import polars as pl
import pytest

from polarticks.volatility import mass_index
from polarticks.volume import eom, kvo, pvt

# ---------------------------------------------------------------------------
# Shared OHLCV fixture
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


# ---------------------------------------------------------------------------
# Mass Index
# ---------------------------------------------------------------------------


class TestMassIndex:
    """Tests for the Mass Index volatility indicator."""

    def test_output_length_matches_input(self) -> None:
        """mass_index output must have the same length as the input."""
        assert len(mass_index(_DF, ema_period=3, sum_period=5)) == _N

    def test_leading_nulls_count(self) -> None:
        """Null prefix must be 2*(ema_period-1) + (sum_period-1)."""
        ema_period, sum_period = 3, 5
        expected_nulls = 2 * (ema_period - 1) + (sum_period - 1)
        result = mass_index(_DF, ema_period=ema_period, sum_period=sum_period)
        for idx in range(expected_nulls):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[expected_nulls] is not None

    def test_alias_is_mass_index(self) -> None:
        """Output series name must be 'mass_index'."""
        assert mass_index(_DF, ema_period=3, sum_period=5).name == "mass_index"

    def test_values_positive(self) -> None:
        """Mass Index values must be positive (ratio of two positive EMAs summed)."""
        result = mass_index(_DF, ema_period=3, sum_period=5)
        valid = [v for v in result.to_list() if v is not None]
        assert all(v > 0.0 for v in valid)

    def test_invalid_period_raises(self) -> None:
        """ema_period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            mass_index(_DF, ema_period=0, sum_period=5)


# ---------------------------------------------------------------------------
# Klinger Volume Oscillator
# ---------------------------------------------------------------------------


class TestKVO:
    """Tests for the Klinger Volume Oscillator."""

    def test_output_columns(self) -> None:
        """KVO result must contain kvo_line and kvo_signal."""
        result = kvo(_DF, fast=5, slow=10, signal=3)
        assert set(result.columns) == {"kvo_line", "kvo_signal"}

    def test_output_length_matches_input(self) -> None:
        """All output columns must be as long as the input."""
        result = kvo(_DF, fast=5, slow=10, signal=3)
        assert len(result) == _N

    def test_kvo_line_leading_nulls(self) -> None:
        """kvo_line must have slow-1 leading nulls."""
        slow = 10
        result = kvo(_DF, fast=5, slow=slow, signal=3)
        kvo_line = result["kvo_line"]
        for idx in range(slow - 1):
            assert kvo_line[idx] is None, f"Expected null at index {idx}"
        assert kvo_line[slow - 1] is not None

    def test_kvo_signal_more_nulls_than_line(self) -> None:
        """kvo_signal must have more leading nulls than kvo_line."""
        result = kvo(_DF, fast=5, slow=10, signal=3)
        line_nulls = sum(1 for v in result["kvo_line"].to_list() if v is None)
        sig_nulls = sum(1 for v in result["kvo_signal"].to_list() if v is None)
        assert sig_nulls > line_nulls

    def test_fast_ge_slow_raises(self) -> None:
        """fast >= slow must raise ValueError."""
        with pytest.raises(ValueError, match="fast"):
            kvo(_DF, fast=10, slow=10)

    def test_invalid_period_raises(self) -> None:
        """fast < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            kvo(_DF, fast=0, slow=10, signal=3)

    def test_embedded_nulls_are_skipped(self) -> None:
        """Bars with any null OHLCV value must be skipped without crashing."""
        # Inject a null bar at index 2 — the loop's null-guard should continue past it.
        df = _DF.clone()
        null_bar = pl.DataFrame(
            {
                "open": [None],
                "high": [None],
                "low": [None],
                "close": [None],
                "volume": [None],
            }
        )
        df_with_null = pl.concat([df[:2], null_bar.cast(df.schema), df[3:]])
        result = kvo(df_with_null, fast=5, slow=10, signal=3)
        assert len(result) == _N
        assert set(result.columns) == {"kvo_line", "kvo_signal"}


# ---------------------------------------------------------------------------
# Ease of Movement
# ---------------------------------------------------------------------------


class TestEOM:
    """Tests for the Ease of Movement indicator."""

    def test_output_length_matches_input(self) -> None:
        """EOM output must have the same length as the input."""
        assert len(eom(_DF, period=5)) == _N

    def test_leading_nulls_count(self) -> None:
        """Null prefix must equal period (1 from shift + period-1 from rolling mean)."""
        period = 5
        result = eom(_DF, period=period)
        for idx in range(period):
            assert result[idx] is None, f"Expected null at index {idx}"
        assert result[period] is not None

    def test_alias_includes_period(self) -> None:
        """Output series name must embed the period."""
        assert eom(_DF, period=5).name == "eom_5"

    def test_invalid_period_raises(self) -> None:
        """period < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            eom(_DF, period=0)


# ---------------------------------------------------------------------------
# Price Volume Trend
# ---------------------------------------------------------------------------


class TestPVT:
    """Tests for the Price Volume Trend indicator."""

    def test_output_length_matches_input(self) -> None:
        """PVT output must have the same length as the input."""
        assert len(pvt(_DF)) == _N

    def test_no_leading_nulls(self) -> None:
        """PVT starts accumulating from bar 0 — no leading nulls."""
        result = pvt(_DF)
        assert result[0] is not None

    def test_alias_is_pvt(self) -> None:
        """Output series name must be 'pvt'."""
        assert pvt(_DF).name == "pvt"

    def test_flat_price_no_contribution(self) -> None:
        """On a flat price series (no % change) PVT should remain at 0."""
        flat = pl.DataFrame(
            {
                "close": [50.0] * 20,
                "volume": [1000.0] * 20,
            }
        )
        result = pvt(flat)
        assert all(v == pytest.approx(0.0, abs=1e-9) for v in result.to_list())

    def test_cumulative_monotonic_on_rising_price(self) -> None:
        """On a monotonically rising price series PVT must be non-decreasing."""
        rising = pl.DataFrame(
            {
                "close": [float(i + 100) for i in range(20)],
                "volume": [1000.0] * 20,
            }
        )
        result = pvt(rising)
        vals = result.to_list()
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1] - 1e-9
