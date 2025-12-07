"""
Optimizer and advanced scoring helpers.

Features:
- Compute forward returns from price series for IC calculation
- Compute Information Coefficients (rank and pearson) for each factor
- IC-weighted scoring (use past ICs to weight z-scored factors)
- Regression-based scoring using Ridge (with optional cross-validation)
- Utilities to load factor history from DB and align with forward returns

Expectations:
- factor_snapshots table: ticker, rebalance_date, factor columns...
- raw_prices table: ticker, trade_date, adj_close
- rebalance_date and trade_date are DATE-like strings accepted by pandas.to_datetime
- sklearn is used for Ridge; if not installed, regression-based scoring will raise ImportError
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

#from src.db.client import get_engine
from src.backtest.runner import load_price_panel

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ============================================================
# HELPER UTILITY
# ============================================================


def load_factor_history(engine, factors: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load factor_snapshots for requested factors and return a long DataFrame:
    columns: ticker, rebalance_date (datetime), <factors...>, gics_sector (optional)

    Optionally restrict by rebalance_date between start_date and end_date (inclusive).
    """
    cols = ["ticker", "rebalance_date"] + factors + ["gics_sector"]
    # build SQL selecting available columns (skip missing ones):
    # read the whole table and subset to be robust to missing columns.
    q = text("SELECT * FROM factor_snapshots WHERE 1=1" 
             + (" AND rebalance_date >= :start" if start_date else "") 
             + (" AND rebalance_date <= :end" if end_date else "")
            )
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


#def load_price_panel(engine, start_date: str, end_date: str) -> pd.DataFrame:
#    """
#    Load adj_close prices between start_date and end_date.
#    Returns DataFrame indexed by trade_date with columns = tickers.
#    """
#    q = text("""
#        SELECT ticker, trade_date, adj_close
#        FROM raw_prices
#        WHERE trade_date BETWEEN :start AND :end
#        """)
#    with engine.connect() as conn:
#        df = pd.read_sql(q, conn, params={"start": start_date, "end": end_date})
#    if df.empty:
#        return pd.DataFrame()
#    df["trade_date"] = pd.to_datetime(df["trade_date"])
#    panel = df.pivot(index="trade_date", columns="ticker", values="adj_close").sort_index()
#    return panel


# ============================================================
# FORWARD RETURNS
# ============================================================


def compute_forward_returns(price_panel: pd.DataFrame, forward_days: int = 21) -> pd.DataFrame:
    """
    Compute forward returns over forward_days for each ticker.
    price_panel: DataFrame indexed by trade_date with columns tickers (adj_close)

    Returns DataFrame indexed by trade_date with same columns, where each cell is the forward return
    from that date to date + forward_days:
      forward_ret_t = price_{t+forward_days} / price_t - 1
    Returns missing values when future price not available.
    """
    if price_panel.empty:
        return pd.DataFrame()
    shifted = price_panel.shift(-forward_days)
    forward_ret = shifted / price_panel - 1.0
    forward_ret.index.name = "trade_date"
    return forward_ret


def align_factors_to_forward_returns(factors_long: pd.DataFrame, 
                                     price_panel: pd.DataFrame, 
                                     forward_days: int = 21) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align factor snapshots (rebalance dates) with forward returns.
    - factors_long: DataFrame with columns ticker, rebalance_date, <factors...>
    - price_panel: price panel with daily adj_close indexed by trade_date
    - forward_days: number of trading days ahead to compute forward return

    Returns:
    - X: DataFrame with MultiIndex = (rebalance_date, ticker), columns = <factors...>
    - y: Series indexed like X (MultiIndex (rebalance_date, ticker)) of forward returns
    """
    if factors_long.empty:
        return pd.DataFrame(), pd.Series()

    # build forward returns panel
    forward = compute_forward_returns(price_panel, forward_days=forward_days)

    # create long factor table with one row per (rebalance_date, ticker)
    factor_cols = [c for c in factors_long.columns if c not in ("ticker", "rebalance_date", "gics_sector")]
    long = factors_long.copy().drop(columns=["gics_sector"], errors="ignore")
    # keep rows where at least one factor exists
    long = long.dropna(subset=factor_cols, how="all")

    # for each (rebalance_date, ticker) pick the forward return starting the next trading day on or after rebalance_date.
    # find the first trade_date >= rebalance_date in forward.index
    trade_dates = forward.index
    if len(trade_dates) == 0:
        return pd.DataFrame(), pd.Series()

    # helper to map rebalance_date to nearest trade_date index
    trade_ts = trade_dates.values.astype("datetime64[ns]")
    rows = []
    y_vals = []
    for _, r in long.iterrows():
        rb_date = np.datetime64(r["rebalance_date"])
        # find first trade_date > rb_date
        # strict ineq is important, otherwise introduce spurious correlations when trade_date = rb_date
        pos = int(np.searchsorted(trade_ts, rb_date, side="right"))
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
        return pd.DataFrame(), pd.Series()

    cols = ["rebalance_date", "ticker"] + factor_cols
    X = pd.DataFrame(rows, columns=cols)
    X["rebalance_date"] = pd.to_datetime(X["rebalance_date"])
    y = pd.Series(y_vals, index=pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=["rebalance_date", "ticker"]))
    X = X.set_index(["rebalance_date", "ticker"])
    return X[factor_cols], y


# ============================================================
# INFORMATION COEFFICIENTS
# ============================================================


def compute_ic_series(X: pd.DataFrame, y: pd.Series, method: str = "spearman") -> pd.DataFrame:
    """
    Compute ICs (per rebalance_date) between each factor and forward returns y.
    - X: DataFrame indexed by MultiIndex (rebalance_date, ticker), columns = factors
    - y: Series indexed by same MultiIndex (rebalance_date, ticker)

    Returns DataFrame indexed by rebalance_date, columns = factors, values = IC at that rebalance.
    """
    if X.empty or y.empty:
        return pd.DataFrame()

    # join X and y
    df = X.join(y.rename("forward_ret"))
    df = df.dropna(subset=["forward_ret"], how="any")
    factor_cols = list(X.columns)
    out_rows = []
    reb_dates = sorted({idx[0] for idx in df.index})
    for d in reb_dates:
        sub = df.loc[d]
        if sub.empty:
            continue
        vals = {}
        for f in factor_cols:
            a = sub[f]
            b = sub["forward_ret"]
            # drop pairs with nan
            mask = a.notna() & b.notna()
            if mask.sum() < 5:
                ic = np.nan
            else:
                if method == "spearman":
                    ic = a.rank().corr(b.rank())
                elif method == "pearson":
                    ic = a.corr(b)
                else:
                    raise ValueError("Unknown method for IC: " + method)
            vals[f] = ic
        vals["rebalance_date"] = d
        out_rows.append(vals)
    if not out_rows:
        return pd.DataFrame()
    ic_df = pd.DataFrame(out_rows).set_index("rebalance_date").sort_index()
    return ic_df


def summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics for ICs per factor: mean, std, t-stat (mean / (std/sqrt(N)))
    """
    if ic_df.empty:
        return pd.DataFrame()
    # orientation: rows = factors
    stats = pd.DataFrame({
        "mean_ic": ic_df.mean(),
        "std_ic": ic_df.std(ddof=1),
        "count": ic_df.count()
    })
    stats["t_stat"] = stats["mean_ic"] / (stats["std_ic"] / np.sqrt(stats["count"].replace(0, np.nan)))
    return stats


# ============================================================
# IC-SCORING
# ============================================================


def ic_weighted_scores(engine,
                       rebalance_date: str,
                       factors: List[str],
                       lookback_rebalances: int = 12,
                       forward_days: int = 21,
                       ic_method: str = "spearman",
                       ic_aggregation: str = "mean") -> pd.DataFrame:
    """
    Compute IC-weighted score for tickers at a given rebalance_date.

    Steps:
    1. Load factor history for the requested factors, up to rebalance_date (lookback_rebalances windows).
    2. Load price panel to compute forward returns.
    3. Compute ICs per factor at each historical rebalance and aggregate (mean by default).
    4. For the target rebalance_date, load factor snapshot and z-score each factor cross-sectionally.
    5. Score = sum_k (z_k * weight_k) where weight_k is aggregated IC (signed).

    Returns DataFrame with columns ticker and score for the target rebalance_date.
    """
    # load recent factor history
    end = pd.to_datetime(rebalance_date)
    # load a generous history window (10 years) to ensure we have at least lookback_rebalances points
    hist_start = (end - pd.Timedelta(days=10*365)).strftime("%Y-%m-%d")
    factors_hist = load_factor_history(engine, factors, start_date=hist_start, end_date=rebalance_date)
    if factors_hist.empty:
        return pd.DataFrame(columns=["ticker", "score"])

    # load price panel covering from earliest hist rebalance to forward horizon
    earliest_rb = factors_hist["rebalance_date"].min()
    price_start = earliest_rb.strftime("%Y-%m-%d")
    price_end = (end + pd.Timedelta(days=forward_days + 21)).strftime("%Y-%m-%d")
    price_panel = load_price_panel(engine, price_start, price_end)
    X, y = align_factors_to_forward_returns(factors_hist, price_panel, forward_days=forward_days)
    if X.empty or y.empty:
        log.warning(" No aligned X/y available for IC computation")
        return pd.DataFrame(columns=["ticker", "score"])

    ic_df = compute_ic_series(X, y, method=ic_method)
    if ic_df.empty:
        return pd.DataFrame(columns=["ticker", "score"])

    # take the most recent lookback_rebalances rows
    ic_recent = ic_df.tail(lookback_rebalances)
    if ic_recent.empty:
        return pd.DataFrame(columns=["ticker", "score"])

    # aggregate ICs across the lookback window
    if ic_aggregation == "mean":
        ic_weights = ic_recent.mean(axis=0)
    elif ic_aggregation == "median":
        ic_weights = ic_recent.median(axis=0)
    elif ic_aggregation == "last":
        ic_weights = ic_recent.iloc[-1]
    else:
        raise ValueError("Unknown ic_aggregation: " + ic_aggregation)

    # normalize weights by sum(abs) to get relative importance, preserve sign
    w = ic_weights.copy().astype(float)
    denom = w.abs().sum()
    if denom == 0 or np.isnan(denom):
        log.warning(" IC weights sum to zero; returning unweighted z-score average")
        w = w.fillna(0.0)
    else:
        w = w / denom

    # load target snapshot
    target = load_factor_history(engine, factors, start_date=rebalance_date, end_date=rebalance_date)
    if target.empty:
        return pd.DataFrame(columns=["ticker", "score"])
    target = target.set_index("ticker")
    # z-score cross-sectionally per factor (use population std ddof=0)
    zs = target[factors].apply(lambda col: (col - col.mean()) / col.std(ddof=0), axis=0)
    zs = zs.fillna(0.0)

    # compute score = sum_k z_k * w_k
    present_factors = [f for f in factors if f in zs.columns and f in w.index]
    if not present_factors:
        return pd.DataFrame(columns=["ticker", "score"])
    score = zs[present_factors].dot(w.loc[present_factors])
    out = score.reset_index().rename(columns={0: "score"})
    out.columns = ["ticker", "score"]
    return out.sort_values("score", ascending=False)


# ============================================================
# RIDGE-BASED SCORING
# ============================================================


def fit_ridge_model(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> pd.Series:
    """
    Fit a Ridge regression model to predict y from X.
    X: DataFrame indexed by MultiIndex (rebalance_date, ticker) with factor columns
    y: Series with same index (forward returns)
    Returns fitted model and a DataFrame of coefficients indexed by factor.
    """
    try:
        from sklearn.linear_model import Ridge
    except Exception as e:
        raise ImportError(" sklearn is required for regression-based scoring") from e

    # align X and y
    df = X.join(y.rename("y")).dropna(subset=["y"])
    df = df.dropna()
    if df.empty:
        raise ValueError(" No overlapping training samples for Ridge")

    cols = X.columns
    X_train = df[cols]
    y_train = df["y"]
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    coefs = pd.Series(model.coef_, index=cols, name="coef")
    return coefs


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

    price_end = (end + pd.Timedelta(days=forward_days + 21)).strftime("%Y-%m-%d")
    price_panel = load_price_panel(engine, start, price_end)
    X, y = align_factors_to_forward_returns(factors_hist, price_panel, forward_days=forward_days)
    if X.empty or y.empty:
        return pd.DataFrame(columns=["ticker", "score"])

    coefs = fit_ridge_model(X, y, alpha=alpha)
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