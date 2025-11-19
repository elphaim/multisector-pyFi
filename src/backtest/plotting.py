"""
Plotting utilities for backtest results.
Functions:
    - plot_equity_curve(df)
    - plot_drawdown(df)
    - plot_exposure(df)

Input: DataFrame with daily backtest results, typically from run_backtest()
Expected columns:
    - 'cum_return': cumulative return
    - 'pnl': daily return
    - 'net_exposure' (optional)
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(df: pd.DataFrame, ax=None):
    """
    Plot cumulative return over time.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    df["cum_return"].plot(ax=ax, color="blue", lw=2)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True)
    return ax


def plot_drawdown(df: pd.DataFrame, ax=None):
    """
    Plot drawdown over time.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    peak = df["cum_return"].cummax()
    drawdown = df["cum_return"] / peak - 1.0
    drawdown.plot(ax=ax, color="red", lw=1.5)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    return ax


def plot_exposure(df: pd.DataFrame, ax=None):
    """
    Plot net exposure if available.
    """
    if "net_exposure" not in df.columns:
        print("No net_exposure column found; skipping plot.")
        return None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
    df["net_exposure"].plot(ax=ax, color="gray", lw=1.5)
    ax.set_title("Net Exposure")
    ax.set_ylabel("Exposure")
    ax.grid(True)
    return ax


def plot_all(df: pd.DataFrame):
    """
    Plot equity, drawdown, and exposure in one figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    plot_equity_curve(df, ax=axes[0])
    plot_drawdown(df, ax=axes[1])
    plot_exposure(df, ax=axes[2])
    plt.tight_layout()
    plt.show()