"""
Shared pytest fixtures for multisector-pyFi tests.
All fixtures use in-memory data — no database required.
"""

import numpy as np
import pandas as pd
import pytest

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
N_DAYS = 300
SEED = 42


@pytest.fixture(scope="session")
def price_panel():
    """
    5 tickers × 300 business-day GBM prices.
    index = trade_date (DatetimeIndex), columns = ticker.
    """
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2023-01-02", periods=N_DAYS)
    data = {}
    for ticker in TICKERS:
        log_returns = rng.normal(0.0005, 0.015, N_DAYS)
        prices = 100.0 * np.exp(np.cumsum(log_returns))
        data[ticker] = prices
    panel = pd.DataFrame(data, index=dates)
    panel.index.name = "trade_date"
    return panel


@pytest.fixture(scope="session")
def prices_long(price_panel):
    """
    price_panel stacked into long form.
    columns = [trade_date, ticker, adj_close]
    """
    df = price_panel.stack().reset_index()
    df.columns = ["trade_date", "ticker", "adj_close"]
    return df


@pytest.fixture(scope="session")
def sample_scores():
    """
    5-ticker DataFrame with hand-picked scores spread from +1.5 to -1.2.
    columns = [ticker, score]
    """
    return pd.DataFrame({
        "ticker": TICKERS,
        "score": [1.5, 0.8, 0.0, -0.6, -1.2],
    })
