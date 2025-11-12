CREATE DATABASE multisector_pyFi;

CREATE TABLE IF NOT EXISTS tickers (
  ticker TEXT PRIMARY KEY, 
  name TEXT, 
  exchange TEXT, 
  gics_sector TEXT, 
  gics_industry TEXT,
  currency TEXT
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
  trailing_pe NUMERIC, 
  eps_basic NUMERIC, 
  eps_diluted NUMERIC,
  revenue NUMERIC, 
  income NUMERIC,
  dividend_yield NUMERIC, 
  source TEXT,
  PRIMARY KEY (ticker, report_date)
  );

CREATE TABLE IF NOT EXISTS factor_snapshots (
  ticker TEXT NOT NULL,
  rebalance_date DATE NOT NULL,
  mom_3m NUMERIC,
  mom_6m NUMERIC,
  vol_1m NUMERIC,
  vol_3m NUMERIC,
  size NUMERIC,
  pe_ratio NUMERIC,
  net_margin NUMERIC,
  profit_margin NUMERIC, 
  gics_sector TEXT,
  mom_3m_zsec NUMERIC,
  mom_6m_zsec NUMERIC,
  PRIMARY KEY (ticker, rebalance_date)
  );

CREATE TABLE backtests (
  run_id UUID PRIMARY KEY,
  name TEXT,
  start_date DATE,
  end_date DATE,
  params JSONB,
  metrics JSONB,
  created_at TIMESTAMP DEFAULT now()
  );

CREATE TABLE etl_log (
  run_id UUID PRIMARY KEY,
  source TEXT,
  table_name TEXT,
  start_ts TIMESTAMP,
  end_ts TIMESTAMP,
  rows_inserted INTEGER,
  status TEXT,
  notes TEXT
  );
