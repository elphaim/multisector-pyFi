#!/usr/bin/env python3
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

HOST = "127.0.0.1"
PORT = 5432
USER = "demo"
PASSWORD = "demo_pwd"
DB_TO_CREATE = "demo_db"

def database_exists(conn, name):
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        return cur.fetchone() is not None

def create_database():
    # connect to the default 'postgres' database as the existing user
    conn = psycopg2.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, dbname="postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        if database_exists(conn, DB_TO_CREATE):
            print(f"Database '{DB_TO_CREATE}' already exists.")
            return
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_TO_CREATE)))
            print(f"Database '{DB_TO_CREATE}' created successfully.")
    except Exception as e:
        print("Error creating database:", e)
    finally:
        conn.close()

if __name__ == "__main__":
    create_database()
