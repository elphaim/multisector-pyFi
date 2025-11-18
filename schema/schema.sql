CREATE TABLE IF NOT EXISTS tickers (
  ticker TEXT PRIMARY KEY, 
  name TEXT, 
  exchange TEXT, 
  gics_sector TEXT, 
  gics_industry TEXT,
  currency TEXT,
  source TEXT
  );

CREATE TABLE IF NOT EXISTS raw_prices (
  ticker TEXT NOT NULL,
  trade_date DATE NOT NULL,
  open NUMERIC,
  high NUMERIC,
  low NUMERIC,
  close NUMERIC,
  adj_close NUMERIC,
  volume BIGINT,
  source TEXT,
  PRIMARY KEY (ticker, trade_date)
  );

CREATE TABLE IF NOT EXISTS raw_fundamentals (
  ticker TEXT NOT NULL,
  report_date DATE NOT NULL,
  market_cap NUMERIC, 
  eps_ttm NUMERIC, 
  revenue NUMERIC, 
  net_income NUMERIC,
  total_equity NUMERIC,
  source TEXT,
  PRIMARY KEY (ticker, report_date)
  );

CREATE TABLE IF NOT EXISTS factor_snapshots (
  ticker TEXT NOT NULL,
  rebalance_date DATE NOT NULL,
  mom_3m NUMERIC,
  mom_6m NUMERIC,
  mom_12m NUMERIC,
  vol_1m NUMERIC,
  vol_3m NUMERIC,
  size NUMERIC,
  pe_ratio NUMERIC,
  roe NUMERIC,
  net_margin NUMERIC,
  gics_sector TEXT,
  PRIMARY KEY (ticker, rebalance_date)
  );

CREATE TABLE IF NOT EXISTS backtests (
  run_id TEXT NOT NULL,
  name TEXT,
  start_date DATE,
  end_date DATE,
  params JSONB,
  metrics JSONB,
  created_at TIMESTAMP DEFAULT now()
  );

CREATE TABLE IF NOT EXISTS positions (
  run_id TEXT NOT NULL,
  as_of DATE NOT NULL,
  ticker TEXT NOT NULL,
  weight NUMERIC,
  entry_price NUMERIC,
  exit_price NUMERIC,
  pnl NUMERIC,
  PRIMARY KEY (run_id, as_of, ticker)
  );


CREATE TABLE IF NOT EXISTS etl_log (
  run_id UUID PRIMARY KEY,
  source TEXT,
  table_name TEXT,
  start_ts TIMESTAMP,
  end_ts TIMESTAMP,
  rows_inserted INTEGER,
  status TEXT,
  notes TEXT
  );

CREATE INDEX IF NOT EXISTS idx_raw_prices_trade_date ON raw_prices (trade_date);
CREATE INDEX IF NOT EXISTS idx_raw_prices_ticker_trade_date ON raw_prices (ticker, trade_date);
CREATE INDEX IF NOT EXISTS idx_factor_snapshots_rebalance ON factor_snapshots (rebalance_date);
CREATE INDEX IF NOT EXISTS idx_positions_run_id ON positions (run_id);
CREATE INDEX IF NOT EXISTS idx_backtests_created_at ON backtests (created_at);
