"""
ETL collection helpers

Provides:
- seed_from_csv(engine, prices_path, fundamentals_path, tickers_path, source)
- ingest_prices_from_csv(engine, csv_path, source)
- ingest_fundamentals_from_csv(engine, csv_path, source)
- ingest_tickers_from_csv(engine, csv_path, source)
- (optional) fetch_prices_yfinance(tickers, start, end)  -- lightweight helper, only used when yfinance installed
- CLI entrypoint to seed sample files into the DB (uses src.db.client.upsert_df_to_table)

Usage (seed sample CSVs):
    python -m src.etl.collect --db-url "postgresql://postgres:pwd@localhost:5332/postgres" \
      --prices data/sample_prices.csv --fundamentals data/sample_fundamentals.csv --tickers data/sample_tickers.csv \
      --source sample_csv

Usage (single table CSV ingest):
    python -m src.etl.collect --db-url "postgresql://postgres:pwd@localhost:5332/postgres" \
      --prices data/sample_prices.csv --source sample_csv
"""

import argparse
import logging
import os
from typing import Optional

import pandas as pd

from src.db.client import get_engine, upsert_df_to_table, wait_for_db

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _read_csv(csv_path: str) -> pd.DataFrame:
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # normalize column names to lower snake_case
    df.columns = [c.strip() for c in df.columns]
    return df


def ingest_prices_from_csv(engine, csv_path: str, source: str = "csv") -> int:
    """
    Read a CSV of prices and upsert into raw_prices.
    Expected columns: ticker(pk), trade_date(pk), open, high, low, close, adj_close, volume, source(optional)
    """
    df = _read_csv(csv_path)
    # normalise column names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    required = {"ticker", "trade_date"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"prices CSV must contain columns: {required}. Found: {list(df.columns)}")

    # ensure types
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    # add source column if missing
    if "source" not in df.columns:
        df["source"] = source
    # reorder to match schema
    cols = ["ticker", "trade_date", "open", "high", "low", "close", "adj_close", "volume", "source"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    count = upsert_df_to_table(engine, df, "raw_prices", pk_cols=["ticker", "trade_date"])
    log.info("Ingested prices rows: %d", len(df))
    return count


def ingest_fundamentals_from_csv(engine, csv_path: str, source: str = "csv") -> int:
    """
    Read a CSV of fundamentals and upsert into raw_fundamentals.
    Expected columns: ticker(pk), report_date(pk), market_cap, trailing_pe, eps_diluted, revenue, income, dividend_yield, source(optional)
    """
    df = _read_csv(csv_path)
    required = {"ticker", "report_date", "market_cap", "trailing_pe", "eps_diluted", "revenue", "income", "dividend_yield"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"fundamentals CSV must contain columns: {required}. Found: {list(df.columns)}")

    df["report_date"] = pd.to_datetime(df["report_date"]).dt.date
    if "source" not in df.columns:
        df["source"] = source

    cols = ["ticker", "report_date", "market_cap", "trailing_pe", "eps_diluted", "revenue", "income", "dividend_yield", "source"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    count = upsert_df_to_table(engine, df, "raw_fundamentals", pk_cols=["ticker", "report_date"])
    log.info("Ingested fundamentals rows: %d", len(df))
    return count


def ingest_tickers_from_csv(engine, csv_path: str, source: str = "csv") -> int:
    """
    Read a CSV of tickers and upsert into tickers table.
    Expected columns: ticker(pk), name, exchange, gics_sector, gics_industry, currency, source
    """
    df = _read_csv(csv_path)
    if "ticker" not in df.columns:
        raise ValueError("tickers CSV must contain column: ticker")
    
    if "source" not in df.columns:
        df["source"] = source

    cols = ["ticker", "name", "exchange", "gics_sector", "gics_industry", "currency", "source"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    count = upsert_df_to_table(engine, df, "tickers", pk_cols=["ticker"])
    log.info("Ingested tickers rows: %d", len(df))
    return count


def seed_from_csv(engine, prices_path: Optional[str], fundamentals_path: Optional[str], tickers_path: Optional[str], source: str = "sample_csv"):
    """
    Convenience wrapper to seed all three tables from CSVs. Logs summary and returns a dict of counts.
    """
    results = {}
    if tickers_path:
        results["tickers"] = ingest_tickers_from_csv(engine, tickers_path)
    if fundamentals_path:
        results["fundamentals"] = ingest_fundamentals_from_csv(engine, fundamentals_path, source=source)
    if prices_path:
        results["prices"] = ingest_prices_from_csv(engine, prices_path, source=source)
    log.info("Seed complete: %s", results)
    return results


# Optional lightweight yfinance helper. yfinance is an optional dependency; guard the import.
def fetch_prices_yfinance(tickers, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily prices from yfinance and return a DataFrame in the same shape as sample_prices.csv:
    ticker, trade_date, open, high, low, close, adj_close, volume, source
    Requires yfinance to be installed.
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance is not installed; install with `pip install yfinance` to use this helper") from e

    out_rows = []
    for t in tickers:
        hist = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
        if hist is None or hist.empty:
            continue
        hist = hist.reset_index()
        hist["Ticker"] = t
        hist = hist.rename(columns={
            "Date": "trade_date", "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
        })
        hist["trade_date"] = pd.to_datetime(hist["trade_date"]).dt.date
        hist["ticker"] = t
        cols = ["ticker", "trade_date", "open", "high", "low", "close", "adj_close", "volume"]
        hist = hist[[c for c in cols if c in hist.columns]]
        hist["source"] = "yfinance"
        out_rows.append(hist)
    if not out_rows:
        return pd.DataFrame(columns=["ticker", "trade_date", "open", "high", "low", "close", "adj_close", "volume", "source"])
    return pd.concat(out_rows, ignore_index=True)


def _build_engine_and_wait(db_url: str, wait_seconds: int = 30):
    engine = get_engine(db_url)
    # wait for DB (use admin DB url if connecting to default postgres)
    ok = wait_for_db(db_url, timeout=wait_seconds)
    if not ok:
        raise RuntimeError(f"Database not reachable at {db_url}")
    return engine


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--db-url", required=True, help="SQLAlchemy DB URL")
    p.add_argument("--prices", help="CSV file path for prices")
    p.add_argument("--fundamentals", help="CSV file path for fundamentals")
    p.add_argument("--tickers", help="CSV file path for tickers")
    p.add_argument("--source", default="sample_csv", help="source label to write into source column")
    p.add_argument("--wait", type=int, default=30, help="seconds to wait for DB to become available")
    args = p.parse_args(argv)

    engine = _build_engine_and_wait(args.db_url, wait_seconds=args.wait)

    # seed provided files
    results = seed_from_csv(engine, prices_path=args.prices, fundamentals_path=args.fundamentals, tickers_path=args.tickers, source=args.source)
    print("Seed results:", results)


if __name__ == "__main__":
    main()
