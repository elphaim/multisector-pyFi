"""
Ingestion CLI wrapper.

Uses src.etl.collect helpers to ingest one or more CSVs into the demo DB and logs the run into etl_log.
Example:
  python -m src.etl.ingest --db-url "postgresql://postgres:pwd@localhost:5332/postgres" \
    --prices data/sample_prices.csv --fundamentals data/sample_fundamentals.csv --tickers data/sample_tickers.csv --source sample_csv
"""

import argparse
import uuid
import datetime as dt
import logging
from typing import Optional

from sqlalchemy import text

from src.db.client import get_engine, wait_for_db
from src.etl.collect import seed_from_csv, ingest_prices_from_csv, ingest_fundamentals_from_csv, ingest_tickers_from_csv
from src.utils.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _write_etl_log(engine, run_id: str, source: str, table_name: str, start_ts: dt.datetime, end_ts: dt.datetime, rows: int, status: str, notes: Optional[str] = None):
    """
    Simple insert into etl_log table. Uses INSERT ... ON CONFLICT DO UPDATE for idempotency.
    """
    with engine.begin() as conn:
        q = text(
            """
            INSERT INTO etl_log (run_id, source, table_name, start_ts, end_ts, rows_inserted, status, notes)
            VALUES (:run_id, :source, :table_name, :start_ts, :end_ts, :rows_inserted, :status, :notes)
            ON CONFLICT (run_id) DO UPDATE
            SET end_ts = EXCLUDED.end_ts,
                rows_inserted = EXCLUDED.rows_inserted,
                status = EXCLUDED.status,
                notes = EXCLUDED.notes
            """
        )
        conn.execute(q, {
            "run_id": run_id,
            "source": source,
            "table_name": table_name,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "rows_inserted": rows,
            "status": status,
            "notes": notes,
        })


def _ingest_single(engine, csv_path: Optional[str], ingest_fn, run_id: str, source: str):
    """
    Ingest one CSV using provided ingest_fn and log the run. Returns rows processed.
    """
    start = dt.datetime.today()
    try:
        rows = ingest_fn(engine, csv_path, source) if csv_path else 0
        status = "success"
        notes = None
    except Exception as e:
        rows = 0
        status = "failed"
        notes = str(e)
        log.exception("Ingest failed for %s", csv_path)
    end = dt.datetime.today()
    table_name = ingest_fn.__name__.replace("ingest_", "")
    _write_etl_log(engine, run_id=run_id, source=source, table_name=table_name, start_ts=start, end_ts=end, rows=rows, status=status, notes=notes)
    return rows, status, notes


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--db-url", help="SQLAlchemy DB URL; if omitted will read from environment via src.utils.config.Config", required=False)
    p.add_argument("--prices", help="CSV file path for prices", required=False)
    p.add_argument("--fundamentals", help="CSV file path for fundamentals", required=False)
    p.add_argument("--tickers", help="CSV file path for tickers", required=False)
    p.add_argument("--source", help="source label to write into source column (overrides env)", required=False)
    p.add_argument("--wait", type=int, help="seconds to wait for DB to become available (overrides env)", required=False)
    args = p.parse_args(argv)

    cfg = Config.from_env() if not args.db_url else Config(db_url=args.db_url,
                                                           default_source=(args.source or "sample_csv"),
                                                           wait_for_db_seconds=(args.wait or 30))
    db_url = args.db_url or cfg.db_url
    source_label = args.source or cfg.default_source
    wait_seconds = args.wait or cfg.wait_for_db_seconds

    # Wait for DB to become available before starting
    ok = wait_for_db(db_url, timeout=wait_seconds)
    if not ok:
        log.error("Database not reachable at %s after %s seconds", db_url, wait_seconds)
        raise SystemExit(2)

    engine = get_engine(db_url)
    run_id = str(uuid.uuid4())
    log.info("Starting ETL run %s source=%s", run_id, source_label)

    # If user passed a single file per table or they want to seed all, use seed wrapper
    if args.prices or args.fundamentals or args.tickers:
        # ingest each supplied file individually so we can log per-table status
        totals = {}
        if args.tickers:
            rows, status, notes = _ingest_single(engine, args.tickers, ingest_tickers_from_csv, run_id, source_label)
            totals["tickers"] = {"rows": rows, "status": status, "notes": notes}
        if args.fundamentals:
            rows, status, notes = _ingest_single(engine, args.fundamentals, ingest_fundamentals_from_csv, run_id, source_label)
            totals["fundamentals"] = {"rows": rows, "status": status, "notes": notes}
        if args.prices:
            rows, status, notes = _ingest_single(engine, args.prices, ingest_prices_from_csv, run_id, source_label)
            totals["prices"] = {"rows": rows, "status": status, "notes": notes}
    else:
        # nothing provided; nothing to do
        log.info("No input CSVs provided; nothing to ingest.")
        totals = {}

    log.info("ETL run %s complete: %s", run_id, totals)
    print({"run_id": run_id, "results": totals})


if __name__ == "__main__":
    main()