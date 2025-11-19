"""
Compute common backtest performance metrics:
    - Sharpe ratio (annualized)
    - Drawdown series and max drawdown

Expected inputs:
    - returns: pd.Series of periodic portfolio returns (index = dates)
"""

import numpy as np
import pandas as pd


# ----- metrics ----------------------------------------------------

def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252, ddof: int = 1) -> float:
    """
    Compute annualized Sharpe ratio.

    Parameters:
    - returns: periodic returns (pd.Series), e.g., daily returns (not cumulative)
    - risk_free_rate: annualized risk-free rate (decimal). If provided, it is converted to period rf.
    - periods_per_year: e.g., 252 for daily, 12 for monthly
    - ddof: degrees of freedom for sample std (1 by default)

    Returns:
    - annualized Sharpe (float). If returns are all NaN or zero variance, returns np.nan.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")

    # convert annual rf to per-period rf
    rf_period = (1 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0 if risk_free_rate != 0 else 0.0
    excess = r - rf_period
    mean_excess = excess.mean()
    std_excess = excess.std(ddof=ddof)
    if std_excess == 0 or np.isnan(std_excess):
        return float("nan")
    sharpe_period = mean_excess / std_excess
    sharpe_annual = sharpe_period * np.sqrt(periods_per_year)
    return float(sharpe_annual)


def compute_cumulative_returns_from_returns(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """
    Convert periodic returns series into cumulative returns / NAV series starting at start_value.
    """
    r = returns.fillna(0.0)
    cum = (1.0 + r).cumprod() * start_value
    return cum


def compute_drawdown_series(cum_returns: pd.Series) -> pd.Series:
    """
    Compute drawdown series from cumulative returns (NAV-like series).
    Output is negative or zero values: e.g., -0.25 for 25% drawdown.
    """
    if cum_returns.empty:
        return pd.Series(dtype=float)
    peak = cum_returns.cummax()
    dd = cum_returns / peak - 1.0
    dd.name = "drawdown"
    return dd