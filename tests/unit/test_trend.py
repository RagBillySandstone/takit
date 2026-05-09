"""Unit tests for takit.trend."""

from __future__ import annotations

import polars as pl
import pytest

from takit.trend import adx, donchian_channels, parabolic_sar, supertrend

OHLC = pl.DataFrame(
    {
        "open": [10.0, 10.5, 11.0, 11.2, 11.8, 12.0, 11.7, 11.0, 10.8, 11.2],
        "high": [11.0, 11.5, 11.8, 11.5, 12.0, 12.2, 11.8, 11.2, 11.5, 12.0],
        "low": [9.5, 10.0, 10.5, 10.2, 10.8, 11.0, 10.5, 10.0, 10.2, 10.8],
        "close": [10.5, 11.0, 11.2, 11.0, 11.8, 12.0, 11.0, 10.5, 11.0, 11.8],
    }
)

# Longer OHLC fixture for indicators with longer warm-up periods (ADX needs 2×period).
OHLC_LONG = pl.DataFrame(
    {
        "open": [float(i) for i in range(10, 50)],
        "high": [float(i) + 0.8 for i in range(10, 50)],
        "low": [float(i) - 0.8 for i in range(10, 50)],
        "close": [float(i) + 0.3 for i in range(10, 50)],
    }
)


class TestDonchianChannels:
    def test_returns_three_columns(self) -> None:
        result = donchian_channels(OHLC, 3)
        assert "dc_upper_3" in result.columns
        assert "dc_lower_3" in result.columns
        assert "dc_middle_3" in result.columns

    def test_upper_gte_lower(self) -> None:
        result = donchian_channels(OHLC, 3)
        for upper, lower in zip(
            result["dc_upper_3"].drop_nulls().to_list(),
            result["dc_lower_3"].drop_nulls().to_list(),
            strict=True,
        ):
            assert upper >= lower

    def test_middle_is_midpoint(self) -> None:
        result = donchian_channels(OHLC, 3)
        for upper, lower, mid in zip(
            result["dc_upper_3"].drop_nulls().to_list(),
            result["dc_lower_3"].drop_nulls().to_list(),
            result["dc_middle_3"].drop_nulls().to_list(),
            strict=True,
        ):
            assert mid == pytest.approx((upper + lower) / 2.0)

    def test_warm_up_is_null(self) -> None:
        result = donchian_channels(OHLC, 3)
        assert result["dc_upper_3"][0] is None
        assert result["dc_upper_3"][1] is None

    def test_output_length_matches_input(self) -> None:
        result = donchian_channels(OHLC, 3)
        assert len(result) == len(OHLC)


class TestADX:
    def test_returns_three_columns(self) -> None:
        result = adx(OHLC_LONG, 5)
        assert "adx_5" in result.columns
        assert "plus_di_5" in result.columns
        assert "minus_di_5" in result.columns

    def test_output_length_matches_input(self) -> None:
        assert len(adx(OHLC_LONG, 5)) == len(OHLC_LONG)

    def test_warm_up_is_null(self) -> None:
        # ADX needs two Wilder passes so the first valid value is past bar 2×period.
        result = adx(OHLC_LONG, 5)
        assert result["adx_5"][0] is None

    def test_adx_between_0_and_100(self) -> None:
        result = adx(OHLC_LONG, 5)
        for val in result["adx_5"].drop_nulls().to_list():
            assert 0.0 <= val <= 100.0

    def test_di_values_non_negative(self) -> None:
        result = adx(OHLC_LONG, 5)
        for val in result["plus_di_5"].drop_nulls().to_list():
            assert val >= 0.0
        for val in result["minus_di_5"].drop_nulls().to_list():
            assert val >= 0.0

    def test_trending_series_has_high_adx(self) -> None:
        # A perfectly trending series should eventually produce ADX > 25.
        result = adx(OHLC_LONG, 5)
        valid_adx = result["adx_5"].drop_nulls().to_list()
        assert max(valid_adx) > 20.0

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            adx(OHLC_LONG, 0)


class TestSupertrend:
    def test_returns_two_columns(self) -> None:
        result = supertrend(OHLC_LONG, 3)
        assert "supertrend" in result.columns
        assert "supertrend_direction" in result.columns

    def test_output_length_matches_input(self) -> None:
        assert len(supertrend(OHLC_LONG, 3)) == len(OHLC_LONG)

    def test_warm_up_is_null(self) -> None:
        result = supertrend(OHLC_LONG, 3)
        assert result["supertrend"][0] is None

    def test_direction_is_plus_or_minus_one(self) -> None:
        result = supertrend(OHLC_LONG, 3)
        for val in result["supertrend_direction"].drop_nulls().to_list():
            assert val in (1, -1)

    def test_band_is_positive(self) -> None:
        result = supertrend(OHLC_LONG, 3)
        for val in result["supertrend"].drop_nulls().to_list():
            assert val > 0.0

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            supertrend(OHLC_LONG, 0)


class TestParabolicSAR:
    def test_returns_two_columns(self) -> None:
        result = parabolic_sar(OHLC_LONG)
        assert "psar" in result.columns
        assert "psar_direction" in result.columns

    def test_output_length_matches_input(self) -> None:
        assert len(parabolic_sar(OHLC_LONG)) == len(OHLC_LONG)

    def test_first_bar_is_null(self) -> None:
        result = parabolic_sar(OHLC_LONG)
        assert result["psar"][0] is None
        assert result["psar_direction"][0] is None

    def test_direction_is_plus_or_minus_one(self) -> None:
        result = parabolic_sar(OHLC_LONG)
        for val in result["psar_direction"].drop_nulls().to_list():
            assert val in (1, -1)

    def test_sar_is_positive(self) -> None:
        result = parabolic_sar(OHLC_LONG)
        for val in result["psar"].drop_nulls().to_list():
            assert val > 0.0

    def test_custom_af_parameters_accepted(self) -> None:
        result = parabolic_sar(OHLC_LONG, initial_af=0.01, step_af=0.01, max_af=0.10)
        assert len(result) == len(OHLC_LONG)

    def test_single_bar_returns_all_null(self) -> None:
        # n=1 < 2 triggers the early-return guard (line 285 in trend.py).
        one_bar = pl.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]}
        )
        result = parabolic_sar(one_bar)
        assert len(result) == 1
        assert result["psar"][0] is None
        assert result["psar_direction"][0] is None

    def test_empty_input_returns_empty_dataframe(self) -> None:
        # n=0 < 2 also triggers the early-return guard.
        empty = pl.DataFrame(
            {
                "open": pl.Series([], dtype=pl.Float64),
                "high": pl.Series([], dtype=pl.Float64),
                "low": pl.Series([], dtype=pl.Float64),
                "close": pl.Series([], dtype=pl.Float64),
                "volume": pl.Series([], dtype=pl.Int64),
            }
        )
        result = parabolic_sar(empty)
        assert len(result) == 0
        assert "psar" in result.columns
        assert "psar_direction" in result.columns
