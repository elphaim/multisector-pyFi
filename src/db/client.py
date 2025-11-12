import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import OperationalError, SQLAlchemyError

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
