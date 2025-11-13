"""
Simple configuration loader.

Loads environment variables from a .env file (if present) and exposes a small Config dataclass.
Requires python-dotenv (pip install python-dotenv).
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env from repo root if present
load_dotenv()


@dataclass
class Config:
    db_url: str
    default_source: str = "sample_csv"
    wait_for_db_seconds: int = 30

    @staticmethod
    def from_env() -> "Config":
        db_url = os.getenv("DB_URL")
        if not db_url:
            raise RuntimeError("DB_URL must be set in environment or .env")
        default_source = os.getenv("DEFAULT_SOURCE", "sample_csv")
        wait_seconds = int(os.getenv("WAIT_FOR_DB_SECONDS", "30"))
        return Config(db_url=db_url, default_source=default_source, wait_for_db_seconds=wait_seconds)
