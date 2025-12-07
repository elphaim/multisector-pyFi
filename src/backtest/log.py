"""
CLI wrapper for multi-date backtest with logging.

Usage:
python -m src.backtest.cli_run_backtest \
  --db-url "postgresql://demo:demo_pass@localhost:5432/demo_db" \
  --start-date "2025-10-01" \
  --end-date "2025-12-31" \
  --rebalance-frequency "monthly" \
  --factors "momentum_3m,size,pe_ratio" \
  --method "equal_weight" \
  --cost-bps 10
"""

import argparse
import logging
import uuid
from typing import List

import pandas as pd
from sqlalchemy import text

from src.db.client import get_engine, wait_for_db
from src.backtest.scoring import score_tickers
from src.backtest.weights import equal_weight_long_short
from src.backtest.runner import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def generate_rebalance_dates(start: str, end: str, freq: str = "monthly") -> List[str]:
    """
    Generate rebalance dates between start and end.
    freq: "monthly", "weekly"
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    if freq == "monthly":
        dates = pd.date_range(start=start_dt, end=end_dt, freq="MS")
    elif freq == "weekly":
        dates = pd.date_range(start=start_dt, end=end_dt, freq="W-MON")
    else:
        raise ValueError(f"Unknown frequency: {freq}")
    return [d.strftime("%Y-%m-%d") for d in dates]


def log_backtest_metadata(engine, run_id: str, name: str, start: str, end: str, params: dict, metrics: dict):
    """
    Insert into backtests table.
    """
    q = text("""
        INSERT INTO backtests (run_id, name, start_date, end_date, params, metrics, created_at)
        VALUES (:run_id, :name, :start_date, :end_date, :params::jsonb, :metrics::jsonb, now())
    """)
    with engine.begin() as conn:
        conn.execute(q, {
            "run_id": run_id,
            "name": name,
            "start_date": start,
            "end_date": end,
            "params": params,
            "metrics": metrics,
        })


def log_positions(engine, run_id: str, rebalance_date: str, weights: pd.DataFrame, prices: pd.DataFrame):
    """
    Insert positions into positions table.
    weights: DataFrame with ticker, weight
    prices: DataFrame with ticker, entry_price
    """
    df = weights.set_index("ticker").join(prices.set_index("ticker"))
    df = df.reset_index()
    df["run_id"] = run_id
    df["as_of"] = pd.to_datetime(rebalance_date).date()
    df["exit_price"] = None
    df["pnl"] = None
    cols = ["run_id", "as_of", "ticker", "weight", "entry_price", "exit_price", "pnl"]
    df = df[cols]
    df.to_sql("positions", engine, if_exists="append", index=False)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--db-url", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--rebalance-frequency", choices=["monthly", "weekly"], default="monthly")
    p.add_argument("--factors", required=True)
    p.add_argument("--method", choices=["equal_weight"], default="equal_weight")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--wait", type=int, default=30)
    args = p.parse_args(argv)

    ok = wait_for_db(args.db_url, timeout=args.wait)
    if not ok:
        log.error("Database not reachable at %s", args.db_url)
        raise SystemExit(2)

    engine = get_engine(args.db_url)
    run_id = str(uuid.uuid4())
    rebalance_dates = generate_rebalance_dates(args.start_date, args.end_date, args.rebalance_frequency)
    all_weights = []

    log.info("Starting backtest run_id=%s with %d rebalances", run_id, len(rebalance_dates))

    for rebalance_date in rebalance_dates:
        scores = score_tickers(engine, rebalance_date, args.factors.split(","))
        if scores.empty:
            log.warning("No scores for %s", rebalance_date)
            continue
        weights = equal_weight_long_short(scores)
        weights["rebalance_date"] = rebalance_date
        all_weights.append(weights)

        # log entry prices
        q = text("SELECT ticker, adj_close FROM raw_prices WHERE trade_date = :d")
        with engine.connect() as conn:
            prices = pd.read_sql(q, conn, params={"d": rebalance_date})
        prices = prices.rename(columns={"adj_close": "entry_price"})
        log_positions(engine, run_id, rebalance_date, weights, prices)

    if not all_weights:
        log.warning("No weights generated; exiting.")
        return

    weights_df = pd.concat(all_weights, ignore_index=True)
    results = run_backtest(engine, weights_df, args.start_date, args.end_date, cost_bps=args.cost_bps)

    metrics = {
        "final_return": float(results["cum_return"].iloc[-1]),
        "max_drawdown": float(results["cum_return"].max() - results["cum_return"].min()),
        "total_days": int(len(results)),
    }

    log_backtest_metadata(engine, run_id, name="demo_backtest", start=args.start_date, end=args.end_date,
                          params={"factors": args.factors, "method": args.method, "cost_bps": args.cost_bps},
                          metrics=metrics)

    log.info("Backtest complete. Final return: %.4f", metrics["final_return"])
    print(results.tail())


if __name__ == "__main__":
    main()