# multisector-pyFi
A multi-sector ingestion, analysis and financial strategy engine.

Use the docker-compose to start a demo PostgreSQL database to interact with.

Once the db is up, run the Jupyter notebook to demo the engine: 
  1. Upsert a sample of data from yahoo finance (provided in data/*.csv) to db.
  2. Factor computation (momentum, volatility, etc...) from sample prices and fundamentals.
  3. Assign ticker scores and weights in portfolio.
  4. Run backtests and compare Sharpe and Max Drawdown across strategies.

Scoring strategies provided:
  1. z-scores
  2. Information Coefficients (Spearman & Pearson)
  3. Ridge regression with tunable train, validation and look-forward windows

For Ridge regression, an optuna optimization utility is included to find the optimal L2-regularization strength. 
This optimization brings Sharpe up from 0.8 to 1.2.