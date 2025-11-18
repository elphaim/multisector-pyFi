import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db.client import get_engine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def load_factor_history(engine, factors: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load factor_snapshots for requested factors and return a long DataFrame:
    columns: ticker, rebalance_date (datetime), <factors...>, gics_sector (optional)

    Optionally restrict by rebalance_date between start_date and end_date (inclusive).
    """
    cols = ["ticker", "rebalance_date"] + factors + ["gics_sector"]
    # build SQL selecting available columns (skip missing ones)
    # We will read the whole table and subset to be robust to missing columns.
    q = text("SELECT * FROM factor_snapshots WHERE 1=1" + (" AND rebalance_date >= :start" if start_date else "") + (" AND rebalance_date <= :end" if end_date else ""))
    params = {}
    if start_date:
        params["start"] = start_date
    if end_date:
        params["end"] = end_date
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params=params)
    if df.empty:
        return df
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    # keep only columns present
    present = [c for c in cols if c in df.columns]
    return df[present]


def load_price_panel(engine, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load adj_close prices between start_date and end_date.
    Returns DataFrame indexed by trade_date with columns = tickers.
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


def compute_forward_returns(price_panel: pd.DataFrame, forward_days: int = 21) -> pd.DataFrame:
    """
    Compute forward returns over forward_days for each ticker.
    price_panel: DataFrame indexed by trade_date with columns tickers (adj_close)
    Returns DataFrame indexed by trade_date with same columns, where each cell is the forward return
    from that date to date + forward_days:
      forward_ret_t = price_{t+forward_days} / price_t - 1
    Missing values where future price not available.
    """
    if price_panel.empty:
        return pd.DataFrame()
    shifted = price_panel.shift(-forward_days)
    forward_ret = shifted / price_panel - 1.0
    forward_ret.index.name = "trade_date"
    return forward_ret


def align_factors_to_forward_returns(factors_long: pd.DataFrame, price_panel: pd.DataFrame, forward_days: int = 21) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align factor snapshots (rebalance dates) with forward returns.
    - factors_long: DataFrame with columns ticker, rebalance_date, factor columns...
    - price_panel: price panel with daily adj_close indexed by trade_date
    - forward_days: number of trading days ahead to compute forward return

    Returns:
    - X: DataFrame with index = timestamp (rebalance_date), columns = flattened MultiIndex (rebalance_date, factor, ticker?)
         We'll return a long table: rows = (rebalance_date, ticker), columns = factors
    - y: Series indexed like X (MultiIndex (rebalance_date, ticker)) of forward returns
    """
    if factors_long.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # build forward returns panel
    forward = compute_forward_returns(price_panel, forward_days=forward_days)

    # create long factor table with one row per (rebalance_date, ticker)
    factor_cols = [c for c in factors_long.columns if c not in ("ticker", "rebalance_date", "gics_sector")]
    long = factors_long.copy().drop(columns=["gics_sector"], errors="ignore")
    long = long.dropna(subset=factor_cols, how="all")  # keep rows where at least one factor exists

    # For each (rebalance_date, ticker) pick the forward return starting the next trading day on or after rebalance_date.
    # We will find the first trade_date >= rebalance_date in forward.index
    trade_dates = forward.index
    if len(trade_dates) == 0:
        return pd.DataFrame(), pd.Series(dtype=float)

    # helper to map rebalance_date to nearest trade_date index
    trade_ts = trade_dates.values.astype("datetime64[ns]")

    rows = []
    y_vals = []
    for _, r in long.iterrows():
        rb_date = np.datetime64(r["rebalance_date"].astype("datetime64[ns]"))
        # find first trade_date >= rb_date
        pos = int(np.searchsorted(trade_ts, rb_date, side="left"))
        if pos >= len(trade_ts):
            continue
        trade_date = trade_dates[pos]
        ticker = r["ticker"]
        # forward return at trade_date for ticker
        try:
            fr = forward.at[trade_date, ticker]
        except Exception:
            fr = np.nan
        if pd.isna(fr):
            continue
        rows.append((r["rebalance_date"], ticker, *[r[f] for f in factor_cols]))
        y_vals.append(fr)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)

    cols = ["rebalance_date", "ticker"] + factor_cols
    X = pd.DataFrame(rows, columns=cols)
    X["rebalance_date"] = pd.to_datetime(X["rebalance_date"])
    y = pd.Series(y_vals, index=pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=["rebalance_date", "ticker"]))
    X = X.set_index(["rebalance_date", "ticker"])
    return X[factor_cols], y

# -------------------------
# Regression-based scoring
# -------------------------

def fit_ridge_model(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0, normalize: bool = False) -> Tuple["Ridge", pd.DataFrame]:
    """
    Fit a Ridge regression model to predict y from X.
    X: DataFrame indexed by MultiIndex (rebalance_date, ticker) with factor columns
    y: Series with same index (forward returns)
    Returns fitted model and a DataFrame of coefficients indexed by factor.
    """
    try:
        from sklearn.linear_model import Ridge
    except Exception as e:
        raise ImportError("sklearn is required for regression-based scoring") from e

    # align X and y
    df = X.join(y.rename("y")).dropna(subset=["y"])
    if df.empty:
        raise ValueError("No overlapping training samples for Ridge")

    cols = X.columns
    X_train = df[cols].values
    y_train = df["y"].values
    model = Ridge(alpha=alpha, normalize=normalize)
    model.fit(X_train, y_train)
    coefs = pd.Series(model.coef_, index=cols, name="coef")
    return model, coefs


def ridge_scoring(engine,
                  rebalance_date: str,
                  factors: List[str],
                  training_window_days: int = 365,
                  forward_days: int = 21,
                  alpha: float = 1.0) -> pd.DataFrame:
    """
    Train a ridge model on historical factor snapshots and forward returns, then score current cross-section.

    Steps:
    - Load factor history for the training window before rebalance_date
    - Load price panel and align X,y
    - Fit Ridge
    - Load target factors at rebalance_date, compute score = X_target.dot(coefs)

    Returns DataFrame with ticker and score sorted descending.
    """
    end = pd.to_datetime(rebalance_date)
    start = (end - pd.Timedelta(days=training_window_days)).strftime("%Y-%m-%d")
    factors_hist = load_factor_history(engine, factors, start_date=start, end_date=rebalance_date)
    if factors_hist.empty:
        return pd.DataFrame(columns=["ticker", "score"])

    price_panel = load_price_panel(engine, start, (end + pd.Timedelta(days=forward_days + 10)).strftime("%Y-%m-%d"))
    X, y = align_factors_to_forward_returns(factors_hist, price_panel, forward_days=forward_days)
    if X.empty or y.empty:
        return pd.DataFrame(columns=["ticker", "score"])

    model, coefs = fit_ridge_model(X, y, alpha=alpha)
    # load target snapshot
    target = load_factor_history(engine, factors, start_date=rebalance_date, end_date=rebalance_date)
    if target.empty:
        return pd.DataFrame(columns=["ticker", "score"])
    target = target.set_index("ticker")
    present = [c for c in factors if c in target.columns]
    X_target = target[present].fillna(0.0)
    # align columns with coef index (missing coef -> 0)
    coef_vec = coefs.reindex(present).fillna(0.0)
    scores = X_target[present].dot(coef_vec)
    out = scores.reset_index().rename(columns={0: "score"})
    out.columns = ["ticker", "score"]
    return out.sort_values("score", ascending=False)