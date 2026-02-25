"""
Tests for src.backtest.weights:
  - equal_weight_long_short
  - volatility_adjusted_weights
"""

import pandas as pd
import pytest

from src.backtest.weights import equal_weight_long_short, volatility_adjusted_weights

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
SCORES = pd.DataFrame({"ticker": TICKERS, "score": [1.5, 0.8, 0.0, -0.6, -1.2]})


# ── equal_weight_long_short ───────────────────────────────────────────────────


class TestEqualWeightLongShort:
    def test_highest_score_is_long(self):
        result = equal_weight_long_short(SCORES, long_pct=0.2, short_pct=0.2)
        long_tickers = result[result["weight"] > 0]["ticker"].tolist()
        assert "AAPL" in long_tickers  # highest score

    def test_lowest_score_is_short(self):
        result = equal_weight_long_short(SCORES, long_pct=0.2, short_pct=0.2)
        short_tickers = result[result["weight"] < 0]["ticker"].tolist()
        assert "TSLA" in short_tickers  # lowest score

    def test_long_weights_sum_to_one(self):
        result = equal_weight_long_short(SCORES, long_pct=0.2, short_pct=0.2)
        long_sum = result[result["weight"] > 0]["weight"].sum()
        assert long_sum == pytest.approx(1.0)

    def test_short_weights_sum_to_minus_one(self):
        result = equal_weight_long_short(SCORES, long_pct=0.2, short_pct=0.2)
        short_sum = result[result["weight"] < 0]["weight"].sum()
        assert short_sum == pytest.approx(-1.0)

    def test_tiny_pct_still_produces_one_position_each_side(self):
        # k = max(1, int(5 * 0.01)) = max(1, 0) = 1
        result = equal_weight_long_short(SCORES, long_pct=0.01, short_pct=0.01)
        assert (result["weight"] > 0).sum() >= 1
        assert (result["weight"] < 0).sum() >= 1


# ── volatility_adjusted_weights ───────────────────────────────────────────────


class TestVolatilityAdjustedWeights:
    _VOL = pd.Series(
        [0.20, 0.25, 0.15, 0.30, 0.18],
        index=TICKERS,
    )

    def test_gross_normalization_target_one(self):
        result = volatility_adjusted_weights(SCORES, self._VOL, target_gross=1.0, normalization="gross")
        assert result["weight"].abs().sum() == pytest.approx(1.0, abs=1e-9)

    def test_gross_normalization_custom_target(self):
        result = volatility_adjusted_weights(SCORES, self._VOL, target_gross=2.0, normalization="gross")
        assert result["weight"].abs().sum() == pytest.approx(2.0, abs=1e-9)

    def test_zero_vol_ticker_dropped(self):
        vol_with_zero = pd.Series([0.20, 0.0, 0.15, 0.30, 0.18], index=TICKERS)
        result = volatility_adjusted_weights(SCORES, vol_with_zero, normalization="gross")
        assert "MSFT" not in result["ticker"].tolist()

    def test_unit_normalization_sum_to_one(self):
        result = volatility_adjusted_weights(SCORES, self._VOL, normalization="unit")
        assert result["weight"].sum() == pytest.approx(1.0, abs=1e-9)

    def test_unknown_normalization_raises(self):
        with pytest.raises(ValueError):
            volatility_adjusted_weights(SCORES, self._VOL, normalization="invalid")
