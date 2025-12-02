"""
Implements a purged walk-forward CV for time-series factor models, 
provides two fast OOS proxy objectives (mean OOS IC and OOS Sharpe of predicted*forward return), 
and a simple grid or Bayesian-style search wrapper that uses the fast proxy to pick candidate alphas. 

This module avoids running full portfolio backtests for every alpha and is intended to be used as 
the fast stage before final grid-search / full backtests.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import logging

from sklearn.linear_model import Ridge

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -------------------------
# Helpers: folding and purging
# -------------------------

def generate_walkforward_folds(all_rebalance_dates: List[pd.Timestamp],
                               train_window: int,
                               val_window: int,
                               step: int = 1,
                               expanding: bool = False) -> List[Tuple[int, int, int, int]]:
    """
    Generate integer-indexed fold boundaries over an ordered list of rebalance dates.

    Parameters:
    - all_rebalance_dates: sorted list of pd.Timestamp (rebalance dates)
    - train_window: number of rebalance periods to use for training
    - val_window: number of rebalance periods used for validation (forward)
    - step: move the train/val window forward by this many rebalance periods each fold
    - expanding: if True, training window expands from start to current fold end; ignore train_window

    Returns:
    - list of tuples (train_start_idx, train_end_idx_inclusive, val_start_idx, val_end_idx_inclusive)
    """
    n = len(all_rebalance_dates)
    folds = []
    start_idx = 0
    # start the first validation so that we have enough train points
    first_val_start = train_window if not expanding else train_window  # for expanding, still skip initial warmup
    i = first_val_start
    while i + val_window - 1 < n:
        val_start = i
        val_end = i + val_window - 1
        if expanding:
            train_start = 0
            train_end = i - 1
        else:
            train_end = i - 1
            train_start = max(0, train_end - train_window + 1)
        folds.append((train_start, train_end, val_start, val_end))
        i += step
    return folds


def purge_training_indices(train_index: np.ndarray,
                           val_index: np.ndarray,
                           forward_horizon: int,
                           rebalance_dates: np.ndarray) -> np.ndarray:
    """
    Purge training indices whose forward horizons overlap validation window.

    - train_index, val_index: arrays of integer indices into rebalance_dates
    - forward_horizon: number of trading days used to compute forward returns (or number of rebalance steps if using rebalance-based forward)
    - rebalance_dates: np.array of pd.Timestamp for each rebalance idx

    Strategy:
    - For each training index t_idx, compute the date at which forward returns begin:
        trade_date = rebalance_dates[t_idx]
      and forward end date = trade_date + forward_horizon (days)
    - If the forward end date >= rebalance_dates[val_start], that training sample is purged (overlaps)
    """
    if len(train_index) == 0 or len(val_index) == 0:
        return train_index
    val_start_date = rebalance_dates[val_index.min()]
    keep = []
    # treat forward_horizon in calendar days for simplicity; if you want trading days,
    # you can map rebalance dates to trade calendar indices
    for t in train_index:
        train_date = rebalance_dates[t]
        forward_end = train_date + np.timedelta64(int(forward_horizon), 'D')
        if forward_end < np.datetime64(val_start_date): # type: ignore
            keep.append(t)
    return np.array(keep, dtype=int)


# -------------------------
# Model training helpers
# -------------------------

def fit_ridge_np(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute Ridge coefficients via closed-form solve: (X^T X + alpha I)^{-1} X^T y
    Returns coef array of shape (n_features,)
    """
    n, p = X.shape
    XtX = X.T @ X
    reg = alpha * np.eye(p)
    try:
        coef = np.linalg.solve(XtX + reg, X.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(XtX + reg, X.T @ y, rcond=None)[0]
    return coef


def train_ridge(X: pd.DataFrame, y: pd.Series, alpha: float, use_sklearn: bool = True) -> np.ndarray:
    """
    Fit Ridge and return coefficients indexed like X.columns.
    """
    Xv = X.values
    yv = y.values
    if Ridge is not None and use_sklearn:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Xv, yv) # type: ignore
        coef = model.coef_
        intercept = model.intercept_
        # We'll ignore intercept for scoring since we typically z-score cross-sections; keep coef only
        return coef
    else:
        return fit_ridge_np(Xv, yv, alpha) # type: ignore


# -------------------------
# Scoring + evaluation per fold
# -------------------------

def compute_oos_ic(y_true: pd.Series, y_pred: pd.Series, method: str = "spearman") -> float:
    """
    Compute Spearman or Pearson correlation between predictions and realized forward returns.
    """
    df = pd.concat([y_true, y_pred], axis=1).dropna()
    if df.shape[0] < 5:
        return float("nan")
    a = df.iloc[:, 0]
    b = df.iloc[:, 1]
    if method == "spearman":
        return a.rank().corr(b.rank())
    else:
        return a.corr(b)


def compute_sharpe_from_series(series: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe of a series of periodic returns (series may be s_pred * realized_return for cross-section);
    we treat the series as a time series and compute Sharpe across the time points.
    """
    r = series.dropna()
    if r.empty:
        return float("nan")
    rf_period = (1 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0 if risk_free_rate != 0 else 0.0
    excess = r - rf_period
    mean = excess.mean()
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(mean / std * np.sqrt(periods_per_year))


def evaluate_alpha_via_walkforward(X_all: pd.DataFrame,
                                   y_all: pd.Series,
                                   rebalance_dates: List[pd.Timestamp],
                                   alphas: List[float],
                                   train_window: int,
                                   val_window: int,
                                   step: int,
                                   forward_horizon_days: int,
                                   objective: str = "mean_oos_ic",
                                   risk_free_rate: float = 0.0,
                                   expanding: bool = False,
                                   use_sklearn: bool = True,
                                   periods_per_year: int = 252) -> Dict[float, Dict]:
    """
    Perform walk-forward CV across a set of alphas.

    Inputs:
    - X_all: DataFrame indexed by MultiIndex (rebalance_date, ticker) with factor columns
    - y_all: Series indexed by same MultiIndex with forward returns
    - rebalance_dates: sorted pd.Timestamp list corresponding to unique rebalance_date values in X_all index
    - alphas: list of alpha values to evaluate
    - train_window, val_window, step: window sizes and step (in rebalance periods)
    - forward_horizon_days: used for purging training samples that leak into validation window (calendar days)
    - objective: 'mean_oos_ic' or 'mean_oos_sharpe'
    - expanding: if True, training window expands from start; else fixed-length sliding window

    Returns:
    - dict mapping alpha -> { 'fold_scores': [...], 'mean_score': float, 'coef_std': Series or None }
    """
    # map rebalance_dates to integer positions
    unique_rb = np.array(sorted(pd.to_datetime(rebalance_dates)))
    date_to_pos = {d: i for i, d in enumerate(unique_rb)}
    # build folds
    folds = generate_walkforward_folds(list(unique_rb), train_window, val_window, step, expanding=expanding)
    if not folds:
        raise ValueError("No folds generated; check window lengths and available rebalance dates")

    # pre-split index positions for each rebalance date
    # X_all index is MultiIndex (rebalance_date, ticker)
    rb_index = np.array([np.datetime64(idx[0]) for idx in X_all.index])
    # group by rebalance_date to get indices of rows belonging to each rebalance
    rb_to_row_indices = {}
    for i, rb in enumerate(np.unique(rb_index)):
        mask = rb_index == rb
        rb_to_row_indices[pd.Timestamp(rb)] = np.where(mask)[0]

    alpha_results = {}
    for alpha in alphas:
        fold_scores = []
        coefs_list = []
        for (t_start, t_end, v_start, v_end) in folds:
            # map to rebalance timestamp windows
            train_rb_pos = np.arange(t_start, t_end + 1)
            val_rb_pos = np.arange(v_start, v_end + 1)
            # collect row indices for train and val
            train_rows = np.concatenate([rb_to_row_indices[pd.Timestamp(unique_rb[i])] for i in train_rb_pos  # type: ignore
                                         if pd.Timestamp(unique_rb[i]) in rb_to_row_indices])  # type: ignore
            # if len(train_rb_pos) else np.array([], dtype=int)
            val_rows = np.concatenate([rb_to_row_indices[pd.Timestamp(unique_rb[i])] for i in val_rb_pos  # type: ignore
                                       if pd.Timestamp(unique_rb[i]) in rb_to_row_indices])  # type: ignore
            # if len(val_rb_pos) else np.array([], dtype=int)

            if train_rows.size == 0 or val_rows.size == 0:
                # skip fold with no data
                continue

            # Purge training rows overlapping validation forward horizon
            train_rows_kept = purge_training_indices(train_rows, val_rows, forward_horizon_days, rb_index)
            if train_rows_kept.size == 0:
                continue

            X_train = X_all.iloc[train_rows_kept]
            y_train = y_all.iloc[train_rows_kept]
            X_val = X_all.iloc[val_rows]
            y_val = y_all.iloc[val_rows]

            # require no NaNs in any feature column and no NaN in target
            train_mask = (~y_train.isna()) & (X_train.notna().all(axis=1))
            X_train = X_train.loc[train_mask]
            y_train = y_train.loc[train_mask]

            # Fit coefficients
            try:
                coef = train_ridge(X_train, y_train, alpha, use_sklearn=use_sklearn) # type: ignore
            except Exception as e:
                log.warning(" Ridge fit failed on fold: %s", e)
                continue

            coefs_list.append(coef)

            # Predict on validation
            # Align columns: ensure X_val columns order matches X_train
            X_val_aligned = X_val.reindex(columns=X_train.columns).fillna(0.0)
            y_pred_vals = X_val_aligned.values @ coef
            y_pred = pd.Series(y_pred_vals, index=X_val.index)

            # compute fold-level objective
            if objective == "mean_oos_ic":
                # measure Spearman correlation per validation rebalance_date then average
                # compute per-rebalance ICs
                v_rb_dates = sorted({idx[0] for idx in X_val.index})
                ic_list = []
                for d in v_rb_dates:
                    sub_idx = X_val.index.get_level_values(0) == d
                    sub_true = y_val[sub_idx]
                    sub_pred = y_pred[sub_idx]
                    dfp = pd.concat([sub_true, sub_pred], axis=1).dropna()
                    if dfp.shape[0] < 5:
                        continue
                    ic = dfp.iloc[:,0].rank().corr(dfp.iloc[:,1].rank())
                    ic_list.append(ic)
                fold_score = np.nanmean(ic_list) if ic_list else np.nan
            elif objective == "mean_oos_sharpe":
                # compute a time series of s_pred * realized forward return aggregated per rebalance date
                # aggregate per rebalance_date by summing (or mean) the cross-sectional contribution
                v_rb_dates = sorted({idx[0] for idx in X_val.index})
                series = []
                for d in v_rb_dates:
                    sub_idx = X_val.index.get_level_values(0) == d
                    sub_true = y_val[sub_idx]
                    sub_pred = y_pred[sub_idx]
                    dfp = pd.concat([sub_true, sub_pred], axis=1).dropna()
                    if dfp.shape[0] == 0:
                        continue
                    # cross-sectional scalar return for that rebalance (score-weighted realized return)
                    cs_val = (dfp.iloc[:,1] * dfp.iloc[:,0]).sum()
                    series.append(cs_val)
                if len(series) == 0:
                    fold_score = np.nan
                else:
                    s = pd.Series(series)
                    fold_score = compute_sharpe_from_series(s, risk_free_rate, periods_per_year=periods_per_year)
            else:
                raise ValueError("Unknown objective: " + objective)

            fold_scores.append(fold_score)

        # aggregate across folds
        if len(fold_scores) == 0:
            mean_score = float("nan")
            coef_std = None
        else:
            mean_score = float(np.nanmean(fold_scores))
            if coefs_list:
                coef_std = np.nanstd(np.vstack(coefs_list), axis=0)
                coef_std = pd.Series(coef_std, index=X_train.columns) # type: ignore
            else:
                coef_std = None

        alpha_results[alpha] = {"fold_scores": fold_scores, "mean_score": mean_score, "coef_std": coef_std}

    return alpha_results


# -------------------------
# Example usage helper
# -------------------------

def run_fast_grid_search(engine,
                         factors: List[str],
                         rebalance_dates: List[pd.Timestamp],
                         X_all: pd.DataFrame,
                         y_all: pd.Series,
                         alphas: List[float],
                         train_window: int = 12,
                         val_window: int = 1,
                         step: int = 1,
                         forward_horizon_days: int = 21,
                         objective: str = "mean_oos_ic",
                         risk_free_rate: float = 0.0,
                         expanding: bool = False) -> Dict:
    """
    High-level convenience wrapper:
    - X_all and y_all should be prepared by align_factors_to_forward_returns (MultiIndex index)
    - rebalance_dates is the sorted list of unique rebalance_date timestamps
    - returns dict with alpha->mean_score and recommended best alpha
    """
    res = evaluate_alpha_via_walkforward(X_all, y_all, rebalance_dates,
                                         alphas, train_window, val_window, step,
                                         forward_horizon_days, objective, risk_free_rate, expanding)
    # pick best alpha
    best_alpha = max(res.items(), key=lambda kv: (np.nan_to_num(kv[1]["mean_score"], nan=-np.inf), -0.0))[0]
    return {"alpha_results": res, "best_alpha": best_alpha}