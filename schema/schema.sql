CREATE DATABASE multisector_pyFi

CREATE TABLE IF NOT EXISTS tickers (
  ticker TEXT PRIMARY KEY, 
  name TEXT, 
  exchange TEXT, 
  gics_sector TEXT, 
  gics_industry TEXT,
  currency TEXT, 
  first_date DATE, 
  last_date DATE
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
  EPS_basic NUMERIC, 
  EPS_diluted NUMERIC,
  revenue NUMERIC, 
  income NUMERIC, 
  profit_margin NUMERIC, 
  dividend_yield NUMERIC, 
  source TEXT,
  PRIMARY KEY (ticker, report_date)
  );
