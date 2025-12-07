"""
Rolling / Walk-Forward Cross-Validation for Factor Models

This module implements:
- Walk-forward fold generation
- Purged CV (removes training samples whose forward horizon overlaps validation)
- Ridge training per fold
- OOS IC evaluation
- Safe NaN handling and cross-sectional standardization

Used by Optuna to optimize Ridge alpha.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import logging

#from sklearn.linear_model import Ridge

from src.backtest.optimizer import fit_ridge_model
#from src.backtest.metrics import compute_sharpe

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ============================================================
# FOLD GENERATION
# ============================================================


def generate_walkforward_folds(rebalance_dates: List[pd.Timestamp],
                               train_window: int,
                               val_window: int,
                               step: int = 1,
                               expanding: bool = False) -> List[Tuple[int, int, int, int]]:
    """
    Generate walk-forward folds over rebalance dates.

    Returns list of tuples:
        (train_start_idx, train_end_idx, val_start_idx, val_end_idx)
    """
    n = len(rebalance_dates)
    folds = []

    first_val_start = train_window
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


# ============================================================
# PURGING
# ============================================================


def purge_training_indices(train_rb_pos: np.ndarray,
                           val_rb_pos: np.ndarray,
                           forward_horizon_days: int,
                           rebalance_dates: List[pd.Timestamp]) -> np.ndarray:
    """
    Purge training rebalance positions whose forward horizon overlaps validation.

    train_rb_pos, val_rb_pos: integer positions into rebalance_dates
    """
    if len(train_rb_pos) == 0 or len(val_rb_pos) == 0:
        return train_rb_pos

    val_start_date = rebalance_dates[val_rb_pos.min()]
    keep = []

    for t_pos in train_rb_pos:
        train_date = rebalance_dates[t_pos]
        forward_end = train_date + np.timedelta64(int(forward_horizon_days), "D")

        if forward_end < np.datetime64(val_start_date):
            keep.append(t_pos)

    return np.array(keep, dtype=int)


# ============================================================
# IMPUTATION + STANDARDIZATION
# ============================================================


def impute_and_standardize(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score per rebalance_date (level=0),
    preserving the original MultiIndex shape.
    """
    df = X.copy()

    # Compute mean and std per rebalance_date
    means = df.groupby(level=0).transform("mean")
    stds  = df.groupby(level=0).transform("std")

    # Z-score
    df = (df - means) / stds

    # Replace NaNs (from zero std or missing data)
    df = df.fillna(0.0)

    return df


# ============================================================
# RIDGE TRAINING
# ============================================================


#def fit_ridge_np(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
#    """Closed-form ridge solution."""
#    XtX = X.T @ X
#    reg = alpha * np.eye(X.shape[1])
#    try:
#        coef = np.linalg.solve(XtX + reg, X.T @ y)
#    except np.linalg.LinAlgError:
#        coef = np.linalg.lstsq(XtX + reg, X.T @ y, rcond=None)[0]
#    return coef


#def train_ridge(X: pd.DataFrame, y: pd.Series, alpha: float, use_sklearn: bool = True) -> np.ndarray:
#    """Fit Ridge and return coefficient vector."""
#
#    if Ridge is not None and use_sklearn:
#        model = Ridge(alpha=alpha, fit_intercept=True)
#        model.fit(X, y)
#        return model.coef_
#
#    return fit_ridge_np(X, y, alpha)


# ============================================================
# METRICS
# ============================================================

#def compute_sharpe_from_series(series: pd.Series, periods_per_year: int = 252) -> float:
#    r = series.dropna()
#    if r.empty:
#        return float("nan")
#    mean = r.mean()
#    std = r.std(ddof=1)
#    if std == 0 or np.isnan(std):
#        return float("nan")
#    return float(mean / std * np.sqrt(periods_per_year))


# ============================================================
# 6. MAIN CV EVALUATION
# ============================================================

def evaluate_alpha_via_walkforward(X: pd.DataFrame, 
                                   y: pd.Series,
                                   rebalance_dates: List[str],
                                   alphas: List[float],
                                   train_window: int,
                                   val_window: int,
                                   step: int,
                                   forward_horizon_days: int,
                                   objective: str = "mean_oos_ic",
                                   expanding: bool = False) -> Dict[float, Dict]:
    """
    Evaluate each alpha using walk-forward CV.
    """

    # Convert rebalance dates to sorted list
    unique_rb = sorted(pd.to_datetime(rebalance_dates))

    # Map each rebalance date to row indices in X
    rb_index = np.array([np.datetime64(idx[0]) for idx in X.index])
    rb_to_row_indices = {
        pd.Timestamp(rb): np.where(rb_index == rb)[0]
        for rb in np.unique(rb_index)
    }

    folds = generate_walkforward_folds(unique_rb, train_window, val_window, step, expanding)

    alpha_results = {}

    for alpha in alphas:
#        fold_id = 0
        fold_scores = []
        coefs_list = []

        for (t_start, t_end, v_start, v_end) in folds:

            train_rb_pos = np.arange(t_start, t_end + 1)
            val_rb_pos = np.arange(v_start, v_end + 1)

            # Purge training positions
            train_rb_pos_kept = purge_training_indices(train_rb_pos, val_rb_pos, forward_horizon_days, unique_rb)

            if train_rb_pos_kept.size == 0:
                continue

            # Convert rebalance positions → row indices
            train_rows_list = [
                rb_to_row_indices[unique_rb[i]]
                for i in train_rb_pos_kept
                if unique_rb[i] in rb_to_row_indices
            ]

            val_rows_list = [
                rb_to_row_indices[unique_rb[i]]
                for i in val_rb_pos
                if unique_rb[i] in rb_to_row_indices
            ]

            if len(train_rows_list) == 0 or len(val_rows_list) == 0:
                continue

            train_rows = np.concatenate(train_rows_list)
            val_rows = np.concatenate(val_rows_list)

            y_finite = y.replace([np.inf, -np.inf], np.nan)

            X_train = X.iloc[train_rows]
            y_train = y_finite.iloc[train_rows]
            X_val = X.iloc[val_rows]
            y_val = y_finite.iloc[val_rows]

            # Drop NaNs in y_train and keep X_train rows where at least one feature is not NaN
            mask = (~y_train.isna()) & (X_train.notna().sum(axis=1) > 0)
            if mask.sum() < 5:
                continue

            X_train = X_train.loc[mask]
            y_train = y_train.loc[mask]

            # Impute + standardize
            X_train = impute_and_standardize(X_train)
            X_val = impute_and_standardize(X_val)

            # Align columns
            X_val = X_val.reindex(columns=X_train.columns).fillna(0.0)

            # Optional diagnostics for fold
            def _diagnose_fold(X_train, y_train, X_val, y_val, alpha, fold_id):
                print("\n====================")
                print(f"FOLD {fold_id} — alpha={alpha}")
                print("====================")
                # Basic shapes
                print("X_train shape:", X_train.shape)
                print("y_train shape:", y_train.shape)
                print("X_val shape:", X_val.shape)
                print("y_val shape:", y_val.shape)
                # Check for NaNs
                print("X_train NaNs:", X_train.isna().sum().sum())
                print("y_train NaNs:", y_train.isna().sum())
                print("X_val NaNs:", X_val.isna().sum().sum())
                print("y_val NaNs:", y_val.isna().sum())
                # Variation in y
                print("y_train mean/std/min/max:", 
                      float(y_train.mean()), 
                      float(y_train.std()), 
                      float(y_train.min()),
                      float(y_train.max()))
                # Variation in factors
                print("Factor std (train):")
                print(X_train.std().sort_values())
                # Cross-sectional size per rebalance date
                print("Cross-sectional counts (train):")
                print(X_train.groupby(level=0).size().describe())
                # Check if factors are constant within cross-sections
                print("Cross-sectional factor std per date (train):")
                cs_std = X_train.groupby(level=0).std()
                print(cs_std.mean().sort_values())
                # Correlation between factors and y (raw)
                try:
                    df = X_train.copy()
                    df["y"] = y_train
                    print("Raw factor-y correlations:")
                    print(df.corr()["y"].sort_values())
                except Exception as e:
                    print("Correlation failed:", e)
                print("====================\n")

            # Fit Ridge
            coef = fit_ridge_model(X_train, y_train, alpha)
            coefs_list.append(coef)

            # Predict
            y_pred_vals = X_val.values @ coef
            y_pred = pd.Series(y_pred_vals, index=X_val.index)

            # Objective: mean IC Spearman
            if objective == "mean_oos_ic":
                ic_list = []
                for d in sorted({idx[0] for idx in X_val.index}):
                    sub_idx = X_val.index.get_level_values(0) == d
                    dfp = pd.concat([y_val[sub_idx], y_pred[sub_idx]], axis=1).dropna()
                    if dfp.shape[0] >= 5:
                        ic = dfp.iloc[:, 0].rank().corr(dfp.iloc[:, 1].rank())
                        ic_list.append(ic)
                fold_score = np.nanmean(ic_list) if ic_list else np.nan
            else:
                raise ValueError(f"Unknown objective: {objective}")

            fold_scores.append(fold_score)

#            fold_id += 1
#            _diagnose_fold(X_train, y_train, X_val, y_val, alpha, fold_id)

        # Aggregate
        if len(fold_scores) == 0:
            alpha_results[alpha] = {
                "fold_scores": [],
                "mean_score": float("nan"),
                "coef_std": None
            }
        else:
            coef_std = (
                pd.Series(np.nanstd(np.vstack(coefs_list), axis=0), index=X_train.columns) # pyright: ignore[reportPossiblyUnboundVariable]
                if coefs_list else None
            )
            alpha_results[alpha] = {
                "fold_scores": fold_scores,
                "mean_score": float(np.nanmean(fold_scores)),
                "coef_std": coef_std
            }

    return alpha_results