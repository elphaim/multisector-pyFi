"""
Tests for src.backtest.optimizer:
  - compute_forward_returns
  - compute_ic_series
  - summarize_ic
  - fit_ridge_model
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.optimizer import (
    compute_forward_returns,
    compute_ic_series,
    fit_ridge_model,
    summarize_ic,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_price_panel(n: int = 100, n_tickers: int = 5) -> pd.DataFrame:
    """Linear price panel: 100, 101, 102, … for each ticker."""
    dates = pd.bdate_range("2023-01-02", periods=n)
    data = {f"T{i}": 100.0 + np.arange(n, dtype=float) for i in range(n_tickers)}
    panel = pd.DataFrame(data, index=dates)
    panel.index.name = "trade_date"
    return panel


def _make_ic_inputs(n_dates: int = 3, n_tickers: int = 10):
    """
    MultiIndex (rebalance_date, ticker) X and y.
    factor_a = [0..n_tickers-1] repeated per date; y = same → IC = 1.0
    """
    dates = pd.bdate_range("2023-01-02", periods=n_dates)
    rows = [(d, f"T{i}") for d in dates for i in range(n_tickers)]
    idx = pd.MultiIndex.from_tuples(rows, names=["rebalance_date", "ticker"])
    factor_vals = np.tile(np.arange(n_tickers, dtype=float), n_dates)
    X = pd.DataFrame({"factor_a": factor_vals}, index=idx)
    y = pd.Series(factor_vals.copy(), index=idx)
    return X, y


# ── compute_forward_returns ───────────────────────────────────────────────────


class TestComputeForwardReturns:
    def test_shape_preserved(self):
        panel = _make_price_panel(n=50)
        result = compute_forward_returns(panel, forward_days=5)
        assert result.shape == panel.shape

    def test_last_n_rows_nan(self):
        panel = _make_price_panel(n=50)
        result = compute_forward_returns(panel, forward_days=5)
        assert result.iloc[-5:].isna().all().all()

    def test_deterministic_value_linear_prices(self):
        panel = _make_price_panel(n=20)
        # At t=0: price=100, price[t+5]=105 → return = 5/100 = 0.05
        result = compute_forward_returns(panel, forward_days=5)
        assert result.iloc[0, 0] == pytest.approx(105.0 / 100.0 - 1.0)

    def test_empty_panel_returns_empty(self):
        assert compute_forward_returns(pd.DataFrame()).empty


# ── compute_ic_series ─────────────────────────────────────────────────────────


class TestComputeICSeries:
    def test_returns_one_row_per_rebalance_date(self):
        X, y = _make_ic_inputs(n_dates=3)
        result = compute_ic_series(X, y)
        assert len(result) == 3

    def test_perfect_rank_correlation(self):
        X, y = _make_ic_inputs(n_dates=1, n_tickers=10)
        result = compute_ic_series(X, y)
        assert abs(result["factor_a"].iloc[0] - 1.0) < 1e-9

    def test_values_bounded(self):
        X, y = _make_ic_inputs()
        result = compute_ic_series(X, y)
        assert (result.abs() <= 1.0 + 1e-9).all().all()

    def test_empty_inputs_returns_empty(self):
        assert compute_ic_series(pd.DataFrame(), pd.Series()).empty

    def test_unknown_method_raises(self):
        X, y = _make_ic_inputs()
        with pytest.raises(ValueError):
            compute_ic_series(X, y, method="unknown")


# ── summarize_ic ─────────────────────────────────────────────────────────────


class TestSummarizeIC:
    def _make_ic_df(self) -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-02", periods=5)
        return pd.DataFrame(
            {
                "factor_a": [0.1, 0.2, -0.1, 0.3, 0.0],
                "factor_b": [0.05, -0.05, 0.1, 0.2, 0.15],
            },
            index=dates,
        )

    def test_required_columns_present(self):
        result = summarize_ic(self._make_ic_df())
        assert {"mean_ic", "std_ic", "t_stat"}.issubset(result.columns)

    def test_t_stat_formula(self):
        ic_df = self._make_ic_df()
        result = summarize_ic(ic_df)
        for factor in ic_df.columns:
            mean = ic_df[factor].mean()
            std = ic_df[factor].std(ddof=1)
            n = ic_df[factor].count()
            expected_t = mean / (std / np.sqrt(n))
            assert result.loc[factor, "t_stat"] == pytest.approx(expected_t)

    def test_index_is_factors(self):
        ic_df = self._make_ic_df()
        result = summarize_ic(ic_df)
        assert set(result.index) == set(ic_df.columns)

    def test_empty_returns_empty(self):
        assert summarize_ic(pd.DataFrame()).empty


# ── fit_ridge_model ───────────────────────────────────────────────────────────


class TestFitRidgeModel:
    def _make_xy(self, n_dates: int = 5, n_tickers: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2023-01-02", periods=n_dates)
        rows = [(d, f"T{i}") for d in dates for i in range(n_tickers)]
        idx = pd.MultiIndex.from_tuples(rows, names=["rebalance_date", "ticker"])
        n = n_dates * n_tickers
        X = pd.DataFrame(
            {"f1": rng.normal(size=n), "f2": rng.normal(size=n)}, index=idx
        )
        y = pd.Series(rng.normal(size=n), index=idx)
        return X, y

    def test_returns_series_indexed_by_factor_names(self):
        X, y = self._make_xy()
        result = fit_ridge_model(X, y)
        assert isinstance(result, pd.Series)
        assert list(result.index) == ["f1", "f2"]

    def test_coefficient_sign_matches_dgp(self):
        # y = 3 * f1 + small noise → coef of f1 should be positive
        rng = np.random.default_rng(1)
        dates = pd.bdate_range("2023-01-02", periods=10)
        rows = [(d, f"T{i}") for d in dates for i in range(10)]
        idx = pd.MultiIndex.from_tuples(rows, names=["rebalance_date", "ticker"])
        f1 = rng.normal(size=100)
        X = pd.DataFrame({"f1": f1}, index=idx)
        y = pd.Series(3.0 * f1 + rng.normal(scale=0.1, size=100), index=idx)
        coefs = fit_ridge_model(X, y)
        assert coefs["f1"] > 0
