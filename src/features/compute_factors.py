"""
Compute factor snapshots and upsert into factor_snapshots table.

Features implemented:
- momentum_3m, momentum_6m, momentum_12m (total return using adj_close)
- vol_1m, vol_3m (annualized rolling std of daily returns)
- size (log of market cap)
- pe_ratio (price / eps_ttm when eps_ttm available)
- roe (net_income / total_equity when available)
- net_margin (net_income / revenue when available)

Assumptions & notes:
- Price table: raw_prices with columns ticker, trade_date, adj_close
- Fundamentals expected (optional): market_cap, eps_ttm, net_income, total_equity, revenue
- Factor snapshots are computed for a single rebalance_date passed as argument.
- All returns use adj_close. Momentum lookbacks count trading days (approximate: 21 trading days ~ 1 month)
- Uses src.db.client.upsert_df_to_table to write factor_snapshots
"""

import logging
import numpy as np
import pandas as pd

from typing import Optional

from sqlalchemy import text

from src.db.client import upsert_df_to_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# HELPER UTILITIES
# ============================================================


def _load_prices(engine, as_of_date: str) -> pd.DataFrame:
    """
    Load historical prices up to as_of_date (inclusive).
    Returns DataFrame with columns: ticker, trade_date, adj_close
    """
    q = text(
        """
        SELECT ticker, trade_date, adj_close
        FROM raw_prices
        WHERE trade_date <= :as_of
        ORDER BY ticker, trade_date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"as_of": as_of_date})
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _load_fundamentals(engine) -> pd.DataFrame:
    """
    Load fundamentals.
    Returns DataFrame with ticker, report_date (datetime), and the available fundamentals.
    """
    q = text(
        """
        SELECT ticker, report_date, market_cap, eps_ttm, revenue, net_income, total_equity
        FROM raw_fundamentals
        ORDER BY ticker, report_date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    if df.empty:
        return df
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def _latest_fundamentals_for_date(fund_df: pd.DataFrame, as_of_date: pd.Timestamp, lag_days: int = 30) -> pd.DataFrame:
    """
    For each ticker choose the latest report_date <= (as_of_date - lag_days).
    Returns wide DF indexed by ticker with fundamentals.
    """
    if fund_df.empty:
        return pd.DataFrame(columns=["ticker"])
    cutoff = pd.to_datetime(as_of_date) - pd.Timedelta(days=lag_days)
    cand = fund_df[fund_df["report_date"] <= cutoff].copy()
    if cand.empty:
        return pd.DataFrame(columns=["ticker"])
    # choose max report_date per ticker
    idx = cand.groupby("ticker")["report_date"].idxmax().dropna().astype(int)
    chosen = cand.loc[idx].copy()
    chosen = chosen.set_index("ticker")
    chosen.index.name = "ticker"
    # drop report_date from wide output (not needed in factor snapshot)
    chosen = chosen.drop(columns=["report_date"], errors="ignore")
    return chosen


#def _daily_returns_from_adj_close(df_prices: pd.DataFrame) -> pd.DataFrame:
#    """
#    Given prices with ticker, trade_date, adj_close (datetime), compute daily returns per ticker.
#    Returns DataFrame with ticker, trade_date, ret (simple return).
#    """
#    if df_prices.empty:
#        return pd.DataFrame(columns=["ticker", "trade_date", "ret"])
#    df = df_prices.sort_values(["ticker", "trade_date"]).copy()
#    df["ret"] = df.groupby("ticker")["adj_close"].pct_change()
#    return df[["ticker", "trade_date", "ret"]]


# ============================================================
# FACTOR COMPUTATIONS
# ============================================================


def compute_momentum(adj_close: pd.Series, lookback_days: int, exclude_recent: int = 5) -> Optional[float]:
    """
    Total return over lookback_days using adj_close (pandas Series indexed by date sorted ascending).
    exclude_recent: number of most recent days to exclude (e.g., skip last 5 trading days).
    Returns None if insufficient data.
    """
    if adj_close.dropna().shape[0] < (lookback_days + exclude_recent + 1):
        return None
    # use positions relative to last available date minus exclude_recent
    end_idx = -1 - exclude_recent if exclude_recent > 0 else -1
    start_idx = end_idx - lookback_days
    try:
        start_price = adj_close.iloc[start_idx]
        end_price = adj_close.iloc[end_idx]
        if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
            return None
        return float(end_price / start_price - 1.0)
    except Exception:
        return None


def compute_rolling_vol(returns: pd.Series, window: int) -> Optional[float]:
    """
    Annualized rolling volatility from daily returns series (window in trading days).
    Returns None if insufficient data.
    """
    if returns.dropna().shape[0] < window:
        return None
    sigma = returns.dropna().rolling(window).std().iloc[-1]
    if pd.isna(sigma):
        return None
    # annualize (assume 252 trading days)
    return float(sigma * (252 ** 0.5))


def compute_short_term_reversal(adj_close: pd.Series, window: int = 5):
    """Short-term reversal: negative recent return."""
    return -adj_close.pct_change(window)


def compute_trend_slope(adj_close: pd.Series, window: int = 63):
    """Trend strength via linear regression slope on log prices."""
    logp = pd.Series(data=np.log(adj_close), index=adj_close.index)
    x = np.arange(window)
    return logp.rolling(window).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)


def compute_volume_zscore(volume: pd.Series, window: int = 20):
    """Volume z-score."""
    return (volume - volume.rolling(window).mean()) / volume.rolling(window).std()


def compute_volume_trend(volume: pd.Series, window: int = 63):
    """Volume trend via regression slope."""
    x = np.arange(window)
    return volume.rolling(window).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)


def compute_downside_vol(returns: pd.Series, window: int = 63):
    """Downside volatility: only negative returns."""
    neg = returns.where(returns < 0, 0)
    return np.sqrt((neg**2).rolling(window).mean())


def compute_range_vol(high: pd.Series, low: pd.Series):
    """High-low volatility proxy."""
    return np.log(high / low)


def compute_earnings_yield(eps: pd.Series, price: pd.Series):
    return eps / price.replace(0, np.nan)


def compute_growth(series: pd.Series, window: int = 252):
    """YoY growth for series (e.g. revenue, EPS, net income)."""
    return series.pct_change(window)


def compute_stability(series: pd.Series, window: int = 252):
    """Rolling standard deviation as a stability measure."""
    return series.rolling(window).std()


def compute_amihud_illiquidity(returns: pd.Series, volume: pd.Series, adj_close: pd.Series):
    """Amihud illiquidity: |r| / dollar volume."""
    dollar_vol = volume * adj_close
    return (returns.abs() / dollar_vol.replace(0, np.nan))


def compute_spread_proxy(high: pd.Series, low: pd.Series, close: pd.Series):
    return (high - low) / close.replace(0, np.nan)


# ============================================================
# ORCHESTRATOR FUNCTION
# ============================================================


def build_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a price panel (adj_close) pivoted by trade_date columns.
    Returns DataFrame indexed by ticker, columns trade_date as DatetimeIndex, values adj_close.
    """
    if prices.empty:
        return pd.DataFrame()
    panel = prices.pivot(index="ticker", columns="trade_date", values="adj_close")
    # sort columns (dates)
    panel = panel.reindex(sorted(panel.columns), axis=1)
    return panel


def compute_factors_for_rebalance(engine, rebalance_date: str, 
                                  lookbacks_days: dict = {"m3": 63, "m6": 126, "m12": 252, "v1": 21, "v3": 63}, 
                                  fund_lag_days: int = 30) -> pd.DataFrame:
    """
    Compute factor snapshot for the rebalance_date (string "YYYY-MM-DD").
    Returns DataFrame with one row per ticker and the factor columns.
    """
    log.info(" Loading prices up to %s", rebalance_date)
    prices = _load_prices(engine, rebalance_date)
    if prices.empty:
        log.warning(" No prices found up to %s", rebalance_date)
        return pd.DataFrame()

    price_panel = build_price_panel(prices)  # index=ticker, cols=dates
    # For per-ticker series access, we will use the sorted date columns
    date_cols = list(price_panel.columns)
    if not date_cols:
        return pd.DataFrame()

    # compute fundamentals
    fund_df = _load_fundamentals(engine)
    fund_wide = _latest_fundamentals_for_date(fund_df, pd.to_datetime(rebalance_date), lag_days=fund_lag_days)

    out_rows = []
    log.info(" Computing factors for %d tickers", len(price_panel))
    for ticker, row in price_panel.iterrows():
        # row is adj_close series indexed by date (DatetimeIndex sorted ascending)
        adj = row.dropna()
        # require at least two prices to compute anything
        if adj.shape[0] < 2:
            continue

        # momentum
        m3 = compute_momentum(adj, lookbacks_days["m3"], exclude_recent=5)
        m6 = compute_momentum(adj, lookbacks_days["m6"], exclude_recent=5)
        m12 = compute_momentum(adj, lookbacks_days["m12"], exclude_recent=5)

        # short reversals
        rev5 = compute_short_term_reversal(adj, window=5)
        rev10 = compute_short_term_reversal(adj, window=10)

        # trend slopes
        trend63 = compute_trend_slope(adj, lookbacks_days["m3"])
        trend126 = compute_trend_slope(adj, lookbacks_days["m6"])

        # returns series for vol
        # build a returns series for the last max window days
        # compute daily returns from adj series
        adj_ret = adj.pct_change().dropna()
        v1 = compute_rolling_vol(adj_ret, lookbacks_days["v1"])
        v3 = compute_rolling_vol(adj_ret, lookbacks_days["v3"])

        # volume z-scores


        size = None
        pe = None
        roe = None
        net_margin = None
        price_to_sales = None
        price_to_book = None

        if ticker in fund_wide.index:
            f = fund_wide.loc[ticker]   # type: ignore

            # size: log of market_cap (if market cap present and > 1)
            try:
                mc = f.get("market_cap", None)
                if mc is not None and mc > 1:
                    size = np.log(float(mc))
                else:
                    size = None
            except Exception:
                size = None

            # pe ratio = price / eps_ttm (if eps_ttm present and non-zero)
            try:
                eps = f.get("eps_ttm", None)
                if eps is not None and float(eps) != 0:
                    pe = float(adj.iat[-1]) / float(eps)
            except Exception:
                pe = None

            # roe = net_income / total_equity
            try:
                ni = f.get("net_income", None)
                te = f.get("total_equity", None)
                if ni is not None and te is not None and float(te) != 0:
                    roe = float(ni) / float(te)
            except Exception:
                roe = None

            # net_margin = net_income / revenue
            try:
                ni = f.get("net_income", None)
                rv = f.get("revenue", None)
                if ni is not None and rv is not None and float(rv) != 0:
                    net_margin = float(ni) / float(rv)
            except Exception:
                net_margin = None

            # price to sales = market_cap / revenue
            try:
                mc = f.get("market_cap", None)
                rv = f.get("revenue", None)
                if mc is not None and rv is not None and float(rv) != 0:
                    price_to_sales = float(mc) / float(rv)
            except Exception:
                price_to_sales = None

            # price to book = market_cap / total_equity
            try:
                mc = f.get("market_cap", None)
                te = f.get("total_equity", None)
                if mc is not None and te is not None and float(te) != 0:
                    price_to_book = float(mc) / float(te)
            except Exception:
                price_to_book = None

        out_rows.append({
            "ticker": ticker,
            "rebalance_date": pd.to_datetime(rebalance_date).date(),
            "mom_3m": m3,
            "mom_6m": m6,
            "mom_12m": m12,
            "vol_1m": v1,
            "vol_3m": v3,
            "size": size,
            "pe_ratio": pe,
            "roe": roe,
            "net_margin": net_margin,
        })

    if not out_rows:
        return pd.DataFrame()
    df_out = pd.DataFrame(out_rows)
    
    # Optionally attach sector information from tickers table if available
    try:
        with engine.connect() as conn:
            tks = pd.read_sql(text("SELECT ticker, gics_sector FROM tickers"), conn)
            if not tks.empty:
                df_out = df_out.merge(tks, how="left", on="ticker")
                # rename to gics_sector as in schema
                if "gics_sector" in df_out.columns:
                    pass
    except Exception:
        log.debug(" Could not attach sector mapping; continuing without it")

    # reorder columns to match schema expectations
    cols_order = ["ticker", "rebalance_date", "mom_3m", "mom_6m", "mom_12m",
                  "vol_1m", "vol_3m", "size", "pe_ratio", "roe", "net_margin", "gics_sector"
                  ]
    cols = [c for c in cols_order if c in df_out.columns]
    df_out = df_out[cols]
    return df_out


# ============================================================
# UPSERT TO TABLE
# ============================================================


def upsert_factor_snapshot(engine, df_factors: pd.DataFrame) -> int:
    """
    Upsert computed factor snapshot into factor_snapshots table.
    Returns number of rows processed (len of df_factors).
    """
    if df_factors is None or df_factors.empty:
        log.info(" No factor rows to upsert")
        return 0
    # ensure types: rebalance_date as date
    df = df_factors.copy()
    if "rebalance_date" in df.columns:
        df["rebalance_date"] = pd.to_datetime(df["rebalance_date"]).dt.date
    # upsert into DB
    count = upsert_df_to_table(engine, df, "factor_snapshots", pk_cols=["ticker", "rebalance_date"])
    log.info(" Upserted %d factor snapshot rows", len(df))
    return count