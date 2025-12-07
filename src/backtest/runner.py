"""
Backtest runner that simulates daily PnL using:
    - daily prices (adj_close)
    - rebalance dates and weights
    - simple execution model: trade at next day's open
    - optional transaction costs

Usage:
    from src.backtest.runner import run_backtest
    results = run_backtest(engine, weights_df, start_date, end_date)
"""

import pandas as pd
#import numpy as np

from sqlalchemy import text
#from src.db.client import get_engine


# ============================================================
# HELPER UTILITY
# ============================================================


def load_price_panel(engine, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load daily adj_close prices between start_date and end_date.
    Returns DataFrame: index = trade_date, columns = tickers
    """
    q = text("""
        SELECT ticker, trade_date, adj_close
        FROM raw_prices
        WHERE trade_date BETWEEN :start AND :end
        """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"start": start_date, "end": end_date})
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    panel = df.pivot(index="trade_date", columns="ticker", values="adj_close").sort_index()
    return panel


# ============================================================
# BACKTEST
# ============================================================


def run_backtest(engine, weights_df: pd.DataFrame, start_date: str, end_date: str, cost_bps: int = 10) -> pd.DataFrame:
    """
    Run a simple backtest using daily prices and rebalance weights.

    weights_df: DataFrame with columns [rebalance_date, ticker, weight]
    Returns DataFrame with daily portfolio value, PnL and drawdown
    """
    prices = load_price_panel(engine, start_date, end_date)
    returns = prices.pct_change().fillna(0)

    weights_df["rebalance_date"] = pd.to_datetime(weights_df["rebalance_date"])
    weights_by_date = {d: g.set_index("ticker")["weight"] for d, g in weights_df.groupby("rebalance_date")}

    portfolio = pd.DataFrame(index=returns.index, columns=["pnl", "gross_exposure", "net_exposure"])
    current_weights = pd.Series(0.0, index=returns.columns)

    for date in returns.index:
        if date in weights_by_date:
            new_weights = weights_by_date[date].reindex_like(current_weights).fillna(0.0)
            turnover = (new_weights - current_weights).abs().sum()
            cost = turnover * cost_bps / 10000
            current_weights = new_weights
        else:
            cost = 0.0

        daily_ret = returns.loc[date].fillna(0.0)
        pnl = (current_weights * daily_ret).sum() - cost
        portfolio.loc[date, "pnl"] = pnl
        portfolio.loc[date, "gross_exposure"] = current_weights.abs().sum()
        portfolio.loc[date, "net_exposure"] = current_weights.sum()

    portfolio["cum_return"] = (1 + portfolio["pnl"]).cumprod()
    peak = portfolio["cum_return"].cummax()
    portfolio["drawdown"] = portfolio["cum_return"] / peak - 1.0

    return portfolio