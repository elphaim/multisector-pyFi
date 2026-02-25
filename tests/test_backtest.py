"""
Tests for:
  - src.backtest.log.generate_rebalance_dates
  - src.backtest.runner.run_backtest  (load_price_panel monkeypatched)
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.log import generate_rebalance_dates
from src.backtest.runner import run_backtest

TICKERS = ["AAPL", "MSFT", "GOOG"]
N_DAYS = 60
START = "2024-01-01"
END = "2024-03-31"


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def price_panel_mock():
    """60 business-day GBM prices for 3 tickers; returned by monkeypatched load_price_panel."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(START, periods=N_DAYS)
    data = {
        t: 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, N_DAYS)))
        for t in TICKERS
    }
    panel = pd.DataFrame(data, index=dates)
    panel.index.name = "trade_date"
    return panel


@pytest.fixture
def simple_weights():
    """Weights across 2 rebalance dates for 3 tickers."""
    return pd.DataFrame({
        "rebalance_date": ["2024-01-02"] * 3 + ["2024-02-01"] * 3,
        "ticker": TICKERS * 2,
        "weight": [0.4, 0.4, -0.2, 0.3, 0.5, -0.3],
    })


# ── generate_rebalance_dates ──────────────────────────────────────────────────


class TestGenerateRebalanceDates:
    def test_monthly_jan_to_jun_count(self):
        dates = generate_rebalance_dates("2025-01-01", "2025-06-30", freq="monthly")
        assert len(dates) == 6

    def test_monthly_all_first_of_month(self):
        dates = generate_rebalance_dates("2025-01-01", "2025-06-30", freq="monthly")
        assert all(d.endswith("-01") for d in dates)

    def test_weekly_all_mondays(self):
        dates = generate_rebalance_dates("2025-01-01", "2025-03-31", freq="weekly")
        for d in dates:
            assert pd.to_datetime(d).dayofweek == 0  # Monday

    def test_output_type_is_str(self):
        dates = generate_rebalance_dates("2025-01-01", "2025-03-31", freq="monthly")
        assert all(isinstance(d, str) for d in dates)

    def test_unknown_freq_raises(self):
        with pytest.raises(ValueError):
            generate_rebalance_dates("2025-01-01", "2025-12-31", freq="daily")


# ── run_backtest ──────────────────────────────────────────────────────────────


class TestRunBacktest:
    def test_result_columns(self, monkeypatch, price_panel_mock, simple_weights):
        import src.backtest.runner as runner
        monkeypatch.setattr(runner, "load_price_panel", lambda *a, **kw: price_panel_mock)
        result = run_backtest(None, simple_weights, START, END)
        required = {"pnl", "gross_exposure", "net_exposure", "cum_return", "drawdown"}
        assert required.issubset(result.columns)

    def test_cum_return_first_day(self, monkeypatch, price_panel_mock, simple_weights):
        import src.backtest.runner as runner
        monkeypatch.setattr(runner, "load_price_panel", lambda *a, **kw: price_panel_mock)
        result = run_backtest(None, simple_weights, START, END)
        assert result["cum_return"].iloc[0] == pytest.approx(1.0 + result["pnl"].iloc[0])

    def test_drawdown_nonpositive(self, monkeypatch, price_panel_mock, simple_weights):
        import src.backtest.runner as runner
        monkeypatch.setattr(runner, "load_price_panel", lambda *a, **kw: price_panel_mock)
        result = run_backtest(None, simple_weights, START, END)
        assert (result["drawdown"] <= 1e-9).all()

    def test_cost_reduces_return(self, monkeypatch, price_panel_mock, simple_weights):
        import src.backtest.runner as runner
        monkeypatch.setattr(runner, "load_price_panel", lambda *a, **kw: price_panel_mock)
        result_free = run_backtest(None, simple_weights.copy(), START, END, cost_bps=0)
        result_costly = run_backtest(None, simple_weights.copy(), START, END, cost_bps=50)
        assert result_free["cum_return"].iloc[-1] >= result_costly["cum_return"].iloc[-1]

    def test_empty_weights_zero_pnl(self, monkeypatch, price_panel_mock):
        import src.backtest.runner as runner
        monkeypatch.setattr(runner, "load_price_panel", lambda *a, **kw: price_panel_mock)
        empty_weights = pd.DataFrame(columns=["rebalance_date", "ticker", "weight"])
        result = run_backtest(None, empty_weights, START, END, cost_bps=0)
        np.testing.assert_allclose(result["pnl"].values, 0.0, atol=1e-12)
