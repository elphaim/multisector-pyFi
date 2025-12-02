"""
Weighting strategies for converting scores into portfolio weights.
Supports:
    - Equal-weight long-short (top/bottom quantiles)
    - Volatility-adjusted weights
"""

import pandas as pd

from src.backtest.scoring import _load_factors


def equal_weight_long_short(scores: pd.DataFrame, long_pct=0.1, short_pct=0.1) -> pd.DataFrame:
    """
    Assign equal weights to top and bottom quantiles of scores.
    Returns DataFrame with ticker, weight.
    """
    n = len(scores)
    k_long = max(1, int(n * long_pct))
    k_short = max(1, int(n * short_pct))

    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
    longs = scores.head(k_long).copy()
    shorts = scores.tail(k_short).copy()

    longs["weight"] = 1.0 / k_long
    shorts["weight"] = -1.0 / k_short

    combined = pd.concat([longs, shorts], axis=0)
    return combined[["ticker", "weight"]]


def load_volatility(engine, rebalance_date: str, lookback: str = "1m") -> pd.Series:
    """
    Fetches annualized volatility over lookback (1m or 3m) from factor_snapshots
    Returns Series indexed by ticker
    """
    factors = _load_factors(engine, rebalance_date, ["vol_1m", "vol_3m"])
    factors.set_index("ticker", inplace=True)
    v = "vol" + "_{}".format(lookback)
    return factors[v]


def volatility_adjusted_weights(scores: pd.DataFrame, vol: pd.Series, target_gross: float = 1.0, normalization: str = "gross") -> pd.DataFrame:
    """
    Assign weights proportional to score / volatility.
    vol: Series indexed by ticker with annualized volatility.
    - normalization: one of {"gross", "unit"}:
        - "gross": scale so sum(abs(w)) == target_gross (default)
        - "unit": scale so sum(w) == 1

    Returns DataFrame with ticker, weight.
    """
    df = scores.set_index("ticker").join(vol.rename("vol"))
    # drop tickers with 0 volatility and a score, and drop tickers without either
    df = df.dropna(subset=["score", "vol"]).loc[lambda d: d["vol"] > 0]
    df["raw"] = df["score"] / df["vol"]
    if normalization == "gross":
        total = df["raw"].abs().sum()
        df["weight"] = target_gross * df["raw"] / total
    elif normalization == "unit":
        total = df["raw"].sum()
        df["weight"] = df["raw"] / total
    else:
        raise ValueError("normalization must be one of {'gross','unit'}")
    return df[["weight"]].reset_index()