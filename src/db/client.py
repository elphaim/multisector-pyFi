from sqlalchemy import create_engine

def get_engine(db_url):
  create_engine(db_url, echo=True)
  pass

def apply_schema(engine, schema_path):
  with engine.begin() as conn:
    conn.execute(text("schema_path"))
