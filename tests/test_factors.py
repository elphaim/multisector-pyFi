"""
Tests for src.features.compute_factors:
  - compute_momentum
  - compute_rolling_vol
  - build_price_panel
  - _latest_fundamentals_for_date
"""

import numpy as np
import pandas as pd
import pytest

from src.features.compute_factors import (
    _latest_fundamentals_for_date,
    build_price_panel,
    compute_momentum,
    compute_rolling_vol,
)


# ── compute_momentum ──────────────────────────────────────────────────────────


class TestComputeMomentum:
    def test_rising_prices_positive(self):
        prices = pd.Series(range(1, 51), dtype=float)
        # lookback=10, exclude_recent=0: need 10+0+1=11 rows; have 50 ✓
        # end=prices[-1]=50, start=prices[-11]=40 → return > 0
        result = compute_momentum(prices, lookback_days=10, exclude_recent=0)
        assert result is not None and result > 0

    def test_falling_prices_negative(self):
        prices = pd.Series(range(50, 0, -1), dtype=float)
        result = compute_momentum(prices, lookback_days=10, exclude_recent=0)
        assert result is not None and result < 0

    def test_insufficient_data_returns_none(self):
        prices = pd.Series([100.0] * 5)
        # need lookback=10 + exclude_recent=5 + 1 = 16 > 5 → None
        assert compute_momentum(prices, lookback_days=10, exclude_recent=5) is None

    def test_flat_prices_zero(self):
        prices = pd.Series([100.0] * 30)
        result = compute_momentum(prices, lookback_days=10, exclude_recent=0)
        assert result == pytest.approx(0.0)

    def test_exclude_recent_skips_spike(self):
        # 20 flat prices then a spike; exclude_recent=1 hides the spike
        prices = pd.Series([100.0] * 20 + [1000.0])
        result = compute_momentum(prices, lookback_days=10, exclude_recent=1)
        assert result == pytest.approx(0.0)


# ── compute_rolling_vol ───────────────────────────────────────────────────────


class TestComputeRollingVol:
    def test_positive_vol_on_random_returns(self):
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0, 0.01, 100))
        result = compute_rolling_vol(rets, window=20)
        assert result is not None and result > 0

    def test_zero_returns_zero_vol(self):
        rets = pd.Series([0.0] * 50)
        result = compute_rolling_vol(rets, window=20)
        assert result == pytest.approx(0.0)

    def test_insufficient_data_returns_none(self):
        rets = pd.Series([0.01, -0.01, 0.01])
        assert compute_rolling_vol(rets, window=10) is None

    def test_annualisation(self):
        # alternating ±0.01 → sample std ≈ 0.01 for large n
        # annualized vol ≈ 0.01 * sqrt(252)
        rets = pd.Series([0.01, -0.01] * 200)
        result = compute_rolling_vol(rets, window=200)
        expected = 0.01 * np.sqrt(252)
        assert result == pytest.approx(expected, rel=0.05)


# ── build_price_panel ─────────────────────────────────────────────────────────


class TestBuildPricePanel:
    def test_shape_and_axes(self, prices_long):
        panel = build_price_panel(prices_long)
        assert panel.shape == (5, 300)
        assert panel.index.name == "ticker"

    def test_columns_sorted(self, prices_long):
        panel = build_price_panel(prices_long)
        assert list(panel.columns) == sorted(panel.columns)

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=["ticker", "trade_date", "adj_close"])
        assert build_price_panel(empty).empty


# ── _latest_fundamentals_for_date ─────────────────────────────────────────────


class TestLatestFundamentalsForDate:
    def _make_fund_df(self):
        return pd.DataFrame({
            "ticker":      ["AAPL",       "AAPL",       "MSFT"],
            "report_date": pd.to_datetime(["2023-01-01", "2023-03-01", "2023-02-01"]),
            "eps_ttm":     [5.0,           6.0,           3.0],
        })

    def test_picks_latest_report_before_cutoff(self):
        df = self._make_fund_df()
        # as_of=2023-04-01, lag=30 → cutoff=2023-03-02
        # AAPL: 2023-03-01 ≤ cutoff → picks 6.0 (the later one)
        result = _latest_fundamentals_for_date(df, pd.Timestamp("2023-04-01"), lag_days=30)
        assert "AAPL" in result.index
        assert result.loc["AAPL", "eps_ttm"] == 6.0

    def test_lag_blocks_recent_report(self):
        df = self._make_fund_df()
        # as_of=2023-04-01, lag=90 → cutoff=2023-01-01
        # AAPL: 2023-01-01 ≤ cutoff; 2023-03-01 > cutoff → picks 5.0
        result = _latest_fundamentals_for_date(df, pd.Timestamp("2023-04-01"), lag_days=90)
        assert "AAPL" in result.index
        assert result.loc["AAPL", "eps_ttm"] == 5.0

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=["ticker", "report_date", "eps_ttm"])
        result = _latest_fundamentals_for_date(empty, pd.Timestamp("2023-04-01"))
        assert result.empty

    def test_all_after_cutoff_returns_empty(self):
        df = self._make_fund_df()
        # lag=365 → cutoff ≈ 2022-04-01; all reports are in 2023 → empty
        result = _latest_fundamentals_for_date(df, pd.Timestamp("2023-04-01"), lag_days=365)
        assert result.empty
