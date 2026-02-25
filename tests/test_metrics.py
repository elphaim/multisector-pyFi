"""Tests for src.backtest.metrics.compute_sharpe."""

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import compute_sharpe


def test_empty_returns_nan():
    assert math.isnan(compute_sharpe(pd.Series([], dtype=float)))


def test_all_nan_returns_nan():
    assert math.isnan(compute_sharpe(pd.Series([np.nan, np.nan, np.nan])))


def test_zero_variance_returns_nan():
    # identical returns → std = 0 → nan
    assert math.isnan(compute_sharpe(pd.Series([0.01, 0.01, 0.01])))


def test_positive_returns_positive_sharpe():
    rng = np.random.default_rng(0)
    r = pd.Series(np.abs(rng.normal(0.002, 0.005, 200)))
    assert compute_sharpe(r) > 0


def test_negative_returns_negative_sharpe():
    rng = np.random.default_rng(1)
    r = pd.Series(-np.abs(rng.normal(0.002, 0.005, 200)))
    assert compute_sharpe(r) < 0


def test_rf_reduces_sharpe():
    # high signal-to-noise so sample mean is reliably positive
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.01, 0.001, 500))
    s_no_rf = compute_sharpe(r, risk_free_rate=0.0)
    s_with_rf = compute_sharpe(r, risk_free_rate=0.05)
    assert s_no_rf > s_with_rf


def test_annualisation_ratio():
    """Ratio of daily-annualised vs monthly-annualised Sharpe = sqrt(252)/sqrt(12)."""
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    s_daily = compute_sharpe(r, periods_per_year=252)
    s_monthly = compute_sharpe(r, periods_per_year=12)
    expected_ratio = np.sqrt(252) / np.sqrt(12)
    assert abs(s_daily / s_monthly - expected_ratio) < 1e-10
