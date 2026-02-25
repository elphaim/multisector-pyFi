"""
Tests for src.backtest.cv:
  - generate_walkforward_folds
  - purge_training_indices
  - impute_and_standardize
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.cv import (
    generate_walkforward_folds,
    impute_and_standardize,
    purge_training_indices,
)

DATES = list(pd.bdate_range("2023-01-02", periods=20))


# ── generate_walkforward_folds ────────────────────────────────────────────────


class TestGenerateWalkforwardFolds:
    def test_first_fold_indices(self):
        folds = generate_walkforward_folds(DATES, train_window=10, val_window=3, step=1)
        t_start, t_end, v_start, v_end = folds[0]
        assert t_start == 0
        assert t_end == 9
        assert v_start == 10
        assert v_end == 12

    def test_expanding_train_start_always_zero(self):
        folds = generate_walkforward_folds(
            DATES, train_window=10, val_window=3, step=1, expanding=True
        )
        for t_start, _, _, _ in folds:
            assert t_start == 0

    def test_step_2_halves_fold_count(self):
        folds_step1 = generate_walkforward_folds(DATES, train_window=10, val_window=3, step=1)
        folds_step2 = generate_walkforward_folds(DATES, train_window=10, val_window=3, step=2)
        assert len(folds_step2) == len(folds_step1) // 2

    def test_no_train_val_overlap(self):
        folds = generate_walkforward_folds(DATES, train_window=10, val_window=3, step=1)
        for t_start, t_end, v_start, v_end in folds:
            assert t_end < v_start

    def test_too_few_dates_returns_empty(self):
        short = list(pd.bdate_range("2023-01-02", periods=5))
        assert generate_walkforward_folds(short, train_window=10, val_window=3) == []


# ── purge_training_indices ────────────────────────────────────────────────────


class TestPurgeTrainingIndices:
    def test_removed_positions_satisfy_condition(self):
        # val starts at index 15; forward_horizon=5 calendar days
        train = np.arange(0, 15)
        val = np.arange(15, 18)
        kept = purge_training_indices(train, val, forward_horizon_days=5, rebalance_dates=DATES)
        val_start = DATES[15]
        # All kept positions must satisfy: train_date + 5 days < val_start
        for t in kept:
            forward_end = DATES[t] + np.timedelta64(5, "D")
            assert forward_end < np.datetime64(val_start)
        # Some positions kept, some removed (sanity check)
        assert len(kept) > 0
        assert len(kept) < len(train)

    def test_empty_train_returns_empty(self):
        result = purge_training_indices(
            np.array([], dtype=int), np.arange(15, 18), 30, DATES
        )
        assert len(result) == 0

    def test_empty_val_returns_train_unchanged(self):
        train = np.arange(0, 10)
        result = purge_training_indices(train, np.array([], dtype=int), 30, DATES)
        np.testing.assert_array_equal(result, train)


# ── impute_and_standardize ────────────────────────────────────────────────────


class TestImputeAndStandardize:
    def _make_X(self, n_dates: int = 3, n_tickers: int = 10) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2023-01-02", periods=n_dates)
        rows = [(d, f"T{i}") for d in dates for i in range(n_tickers)]
        idx = pd.MultiIndex.from_tuples(rows, names=["rebalance_date", "ticker"])
        data = rng.normal(size=(n_dates * n_tickers, 2))
        return pd.DataFrame(data, index=idx, columns=["f1", "f2"])

    def test_cross_sectional_mean_near_zero(self):
        X = self._make_X()
        result = impute_and_standardize(X)
        for d in result.index.get_level_values(0).unique():
            group_mean = result.loc[d].mean()
            np.testing.assert_allclose(group_mean.values, 0.0, atol=1e-9)

    def test_nan_filled_with_zero(self):
        X = self._make_X()
        X.iloc[0] = np.nan
        result = impute_and_standardize(X)
        assert not result.isna().any().any()

    def test_single_observation_per_date_returns_zero(self):
        # std is NaN for a single obs (ddof=1) → fillna(0)
        dates = pd.bdate_range("2023-01-02", periods=3)
        rows = [(d, "T0") for d in dates]
        idx = pd.MultiIndex.from_tuples(rows, names=["rebalance_date", "ticker"])
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=idx)
        result = impute_and_standardize(X)
        np.testing.assert_allclose(result["f1"].values, 0.0, atol=1e-9)
