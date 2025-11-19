"""
Scoring module.
Reads factor_snapshots for a given rebalance_date and computes a score per ticker.
Supports:
    - Single-factor scoring
    - Equal-weight multi-factor scoring
    - IC-weighted scoring (to be implemented)
    - Sector-neutral z-score normalization

Usage:
    from src.backtest.scoring import score_tickers
    scores = score_tickers(engine, rebalance_date="2025-10-31", factors=["mom_3m", "size", "pe_ratio"])
"""

import logging
from typing import List

import pandas as pd
from sqlalchemy import text
from src.db.client import get_engine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ----- helper utilities ----------------------------------------------------

def _load_factors(engine, rebalance_date: str, factors: List[str]) -> pd.DataFrame:
    """
    Load factor snapshot for a given rebalance_date.
    Returns DataFrame with ticker, gics_sector, and selected factor columns.
    """
    cols = ["ticker", "gics_sector"] + factors
    col_sql = ", ".join(cols)
    q = text(f"""
        SELECT {col_sql}
        FROM factor_snapshots
        WHERE rebalance_date = :rebalance_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"rebalance_date": rebalance_date})
    return df


def _sector_zscore(df: pd.DataFrame, factor: str) -> pd.Series:
    """
    Compute sector-normalized z-score for a single factor.
    Returns a Series of z-scores indexed by ticker.
    """
    def zscore(x):
        return (x - x.mean()) / x.std(ddof=0)

    return df.groupby("gics_sector")[factor].transform(zscore)


# ----- scoring function ----------------------------------------------------

def score_tickers(engine, rebalance_date: str, factors: List[str], method: str = "equal_weight", sector_neutral: bool = True) -> pd.DataFrame:
    """
    Compute score per ticker for a given rebalance_date using selected factors.

    method:
        - "equal_weight": average of z-scores
        - "single": use first factor only

    Returns DataFrame with ticker, score
    """
    df = _load_factors(engine, rebalance_date, factors)
    if df.empty:
        log.warning(" No factor data found for %s", rebalance_date)
        return pd.DataFrame(columns=["ticker", "score"])

    df = df.set_index("ticker")

    # Normalize each factor
    for f in factors:
        if sector_neutral:
            df[f + "_z"] = _sector_zscore(df.reset_index(), f).values
        else:
            df[f + "_z"] = (df[f] - df[f].mean()) / df[f].std(ddof=0)

    # Compute score
    if method == "equal_weight":
        df["score"] = df[[f + "_z" for f in factors]].mean(axis=1)
    elif method == "single":
        df["score"] = df[factors[0] + "_z"]
    else:
        raise ValueError(f"Unknown scoring method: {method}")

    return df[["score"]].reset_index()