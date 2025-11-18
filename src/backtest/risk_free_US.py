import io
import requests
import pandas as pd
from typing import Optional, Tuple


CSV_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2025/all?type=daily_treasury_yield_curve&field_tdr_date_value=2025&page&_format=csv"
TIMEOUT = 10  # seconds


def fetch_latest_10y_from_treasury_csv() -> Tuple[Optional[float], Optional[str]]:
    """
    Returns (yield_decimal, date_str)
    yield_decimal is None on failure.
    """
    try:
        r = requests.get(CSV_URL, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        return None, None

    # load CSV into DataFrame
    try:
        df = pd.read_csv(io.StringIO(r.text), parse_dates=['Date'], dayfirst=False)
    except Exception:
        return None, None

    # keep rows with non-missing 10-year yield and sort by date descending
    df_valid = df[['Date', "10 Yr"]].dropna(subset=["10 Yr"]).sort_values('Date', ascending=False)
    if df_valid.empty:
        return None, None

    latest_row = df_valid.iloc[0]
    date_str = latest_row['Date'].strftime('%Y-%m-%d')
    # Treasury CSV reports yields in percent (e.g., 4.11)
    try:
        value_pct = float(latest_row["10 Yr"])
        value_decimal = value_pct / 100.0
    except Exception:
        return None, date_str

    return value_decimal, date_str


if __name__ == "__main__":
    yld, date = fetch_latest_10y_from_treasury_csv()
    if yld is None:
        print("Could not retrieve 10-year yield from Treasury CSV.")
    else:
        print(f"10-year US Treasury yield (nominal): {yld:.4%}  — published on {date}")