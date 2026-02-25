# multisector-pyFi

A modular Python engine for multi-sector equity analysis: ingest market data, compute quantitative factors, score and weight a portfolio, then backtest the resulting strategy with transaction costs. Designed to run end-to-end from a single Jupyter notebook against a local PostgreSQL instance.

## What It Does

The pipeline walks through four stages, each backed by a dedicated module:

1. **Ingest** — Upsert sample price, fundamentals, and ticker-metadata CSVs (sourced from Yahoo Finance) into PostgreSQL via a staging-table-based `ON CONFLICT` upsert.
2. **Feature engineering** — Compute factor snapshots per rebalance date: momentum (3 / 6 / 12 month), annualized volatility (1 / 3 month), log market-cap (size), P/E, ROE, and net margin. Additional factor primitives (short-term reversal, trend slope, Amihud illiquidity, etc.) are implemented but not yet wired into the snapshot pipeline.
3. **Score & weight** — Three scoring strategies with sector-neutral z-score normalization:
   - **Equal-weight z-score** averaging across selected factors.
   - **IC-weighted** — weight each factor by its trailing Information Coefficient (Spearman or Pearson), computed via rolling walk-forward windows with purged cross-validation.
   - **Ridge regression** — predict forward returns from factor cross-sections; regularization strength tunable via Optuna over walk-forward folds.
4. **Backtest** — Simulate daily PnL with configurable transaction costs (bps), compute Sharpe ratio and drawdown, and plot equity curves. A CLI wrapper (`log.py`) logs run metadata and positions back into the database.

## Architecture

```
├── src/
│   ├── db/
│   │   └── client.py            # SQLAlchemy engine, wait_for_db, schema apply, generic upsert
│   ├── etl/
│   │   └── collect.py           # CSV → raw_prices / raw_fundamentals / tickers upsert
│   ├── features/
│   │   └── compute_factors.py   # Factor computation and factor_snapshots upsert
│   └── backtest/
│       ├── scoring.py           # Z-score-based ticker scoring (single/multi-factor)
│       ├── weights.py           # Equal-weight long/short, volatility-adjusted weighting
│       ├── runner.py            # Daily backtest simulation with transaction costs
│       ├── metrics.py           # Annualized Sharpe ratio
│       ├── cv.py                # Walk-forward CV with purging for Ridge alpha selection
│       ├── optimizer.py         # IC computation, IC-weighted scoring, Ridge-based scoring
│       ├── plotting.py          # Equity curve, drawdown, exposure plots
│       └── log.py               # CLI: run backtest, log metadata + positions to DB
├── schema/
│   └── schema.sql               # PostgreSQL DDL (7 tables, indexes)
├── data/
│   ├── sample_prices.csv
│   ├── sample_fundamentals.csv
│   └── sample_tickers.csv
├── pipeline.ipynb               # End-to-end demo notebook
├── docker-compose.yml           # One-command Postgres setup
├── .env                         # DB connection and date-range defaults
└── requirements.txt
```

## Getting Started

**Prerequisites:** Docker, Python ≥ 3.10

```bash
# 1. Start the database
docker compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook pipeline.ipynb
```

The notebook seeds the database from the sample CSVs, computes factors, scores tickers under each strategy, runs backtests, and compares Sharpe ratios.

## Database Schema

Seven tables managed via `schema/schema.sql`:

| Table | Purpose |
|---|---|
| `tickers` | Ticker metadata (exchange, GICS sector/industry) |
| `raw_prices` | Daily OHLCV + adj close |
| `raw_fundamentals` | Quarterly fundamentals (market cap, EPS, revenue, etc.) |
| `factor_snapshots` | Computed factors per (ticker, rebalance_date) |
| `backtests` | Run-level metadata and aggregate metrics (JSONB) |
| `positions` | Per-ticker weights and entry/exit prices per rebalance |
| `etl_log` | Ingestion audit trail |

## Scoring Strategies

**Z-score (baseline):** Cross-sectional z-scores of selected factors, optionally sector-neutralized, averaged with equal weights.

**IC-weighted:** Each factor's weight is its trailing mean Spearman rank-IC against realized forward returns, computed over a configurable lookback window. Factors with low or negative predictive power are automatically down-weighted or shorted.

**Ridge regression:** Fit a regularized linear model on (factors → forward returns) using walk-forward cross-validation with temporal purging to prevent look-ahead bias. Optuna can be used to search over the L2 penalty. Out-of-sample evaluation uses mean cross-sectional Spearman IC per fold.

## Tech Stack

Python · pandas · NumPy · SQLAlchemy · PostgreSQL · scikit-learn · matplotlib · Docker

## License

MIT