"""
src/backtest/metrics.py

Compute common backtest performance metrics:
- Sharpe ratio (annualized)

Expected inputs:
- returns: pd.Series of periodic portfolio returns (index = dates)
"""

import numpy as np
import pandas as pd


# -------------------------
# Sharpe and basic helpers
# -------------------------

def annualize_factor(periods_per_year: int = 252):
    return periods_per_year


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