from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

def get_engine(db_url: str) -> Engine:
  return create_engine(db_url, echo=True)

def apply_schema(engine: Engine, schema_path: str) -> None:
  with engine.begin() as conn:
    conn.execute(text(schema_path))
