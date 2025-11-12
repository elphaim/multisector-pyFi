"""
Database client helpers for Postgres (Docker-friendly).

Usage examples:
    from src.db.client import get_engine, wait_for_db, apply_schema, upsert_df_to_table, get_last_ingested_date

    engine = get_engine("postgresql://demo:demo_pass@localhost:5432/demo_db")
    wait_for_db(engine.url, timeout=60)
    apply_schema(engine, "schema/schema.sql")
    count = upsert_df_to_table(engine, df, "raw_prices", pk_cols=["ticker", "trade_date"])
"""

import time
import logging
import uuid
import pandas as pd

from typing import List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import OperationalError, SQLAlchemyError

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_engine(db_url: str) -> Engine:
  """
  Create and return a SQLAlchemy Engine.
  Example db_url: postgresql://user:pass@localhost:5432/dbname
  """
  engine = create_engine(db_url)
  return engine

def wait_for_db(db_url: str, timeout: int = 60, interval: float = 1.0) -> bool:
    """
    Wait for the DB server to accept connections.
    Returns True if a connection was successful within timeout seconds, False otherwise.
    """
    start = time.time()
    while True:
        try:
            eng = create_engine(db_url)
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("Database reachable at %s", db_url)
            return True
        except OperationalError as e:
            if time.time() - start > timeout:
                log.error("Timed out waiting for DB: %s", e)
                return False
            time.sleep(interval)


def apply_schema(engine: Engine, schema_path: str) -> None:
  """
  Apply SQL schema file to the connected database.
  - Reads the schema file and executes statements in a single transaction.
  - Intended for Postgres; keep schema compatible with Postgres.
  """
  with open(schema_path, "r", encoding="utf8") as f:
    sql = f.read()
  
  # Split by semicolon and execute statements sequentially to improve error visibility.
  stmts = [s.strip() for s in sql.split(";") if s.strip()]
  with engine.begin() as conn:
    for stmt in stmts:
      conn.execute(text(stmt))
  log.info("Applied schema from %s", schema_path)

def upsert_df_to_table(engine: Engine, df: pd.DataFrame, table_name: str, pk_cols: List[str]) -> int:
    """
    Upsert a pandas DataFrame into a Postgres table using a staging table and ON CONFLICT.
    Strategy:
      1. Write df to a temporary staging table (stg_<table>_<uuid>).
      2. Run an INSERT ... SELECT FROM staging ON CONFLICT (pk...) DO UPDATE SET ...
      3. Drop staging table.
    Returns number of rows processed from staging (approx).
    """
    if df is None or df.empty:
        log.info("Empty DataFrame for upsert to %s, nothing to do.", table_name)
        return 0

    tmp_table = f"stg_{table_name}_{uuid.uuid4().hex[:8]}"
    # use a single connection / transaction
    with engine.begin() as conn:
        # write staging table
        # pandas to_sql needs a connection with a dialect; using SQLAlchemy connection object is OK
        df.to_sql(tmp_table, conn, if_exists="replace", index=False, method="multi")  # batch inserts

        # build column list and SQL fragments
        cols = list(df.columns)
        quoted_cols = ", ".join(f'"{c}"' for c in cols)
        pk_quoted = ", ".join(f'"{c}"' for c in pk_cols)
        update_cols = [c for c in cols if c not in pk_cols]
        if update_cols:
            update_assign = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
        else:
            update_assign = ""

        upsert_sql = f"""
        INSERT INTO "{table_name}" ({quoted_cols})
        SELECT {quoted_cols} FROM "{tmp_table}"
        ON CONFLICT ({pk_quoted})
        DO UPDATE SET {update_assign}
        ;
        DROP TABLE IF EXISTS "{tmp_table}";
        """
        conn.execute(text(upsert_sql))
        
        # Attempt to count rows in staging to report processed row count
        try:
            res = conn.execute(text(f"SELECT COUNT(1) FROM \"{table_name}\""))
            total_after = int(res.scalar() or 0)
            log.info("Upsert to %s completed; target table now has %d rows (total)", table_name, total_after)
        except SQLAlchemyError:
            log.info("Upsert to %s completed (row count unavailable)", table_name)
    return len(df)

def get_last_ingested_date(engine: Engine, table_name: str, source: Optional[str] = None) -> Optional[str]:
    """
    Return the latest date value for common date columns (trade_date or report_date) for a given table.
    If source is provided, it will add WHERE source = :source when the column exists.
    Returns string representation of the max date or None if not found.
    """
    candidates = ["trade_date", "report_date", "rebalance_date"]
    with engine.connect() as conn:
        for col in candidates:
            # check whether column exists using information_schema
            try:
                exists_q = text(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = :tbl AND column_name = :col"
                )
                res = conn.execute(exists_q, {"tbl": table_name, "col": col})
                if res.first() is None:
                    continue
                q_text = f"SELECT MAX(\"{col}\") as last_date FROM \"{table_name}\""
                if source:
                    q_text += " WHERE source = :source"
                    res2 = conn.execute(text(q_text), {"source": source})
                else:
                    res2 = conn.execute(text(q_text))
                val = res2.scalar()
                if val is not None:
                    return str(val)
            except SQLAlchemyError:
                # skip this column if any DB error occurs and try the next
                continue
    return None
