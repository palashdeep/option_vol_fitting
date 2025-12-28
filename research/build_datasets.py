"""
data_acquisition.py

Purpose:
- Pull underlying daily history for tickers (via yfinance)
- Pull current option chains for each expiry
- Compute mid prices, time-to-expiry, forward estimate, log-moneyness
- Compute implied vols via Black-Scholes inversion
- Compute realized vol (21d rolling) from historical closes
- Save cleaned outputs to data/raw and data/processed
"""

import os
import math
from core.arbitrage.black_scholes import implied_vol_from_price
from core.viz.utils import assign_tenor_series
from functools import reduce
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["SPY"]
START_DATE = "2023-10-01"
END_DATE = "2025-10-15"
OUTPUT_DIR = "data"
RISK_FREE_RATE = 0.05
REALIZED_LOOKBACK_DAYS = 30
DIVIDEND_YIELD = 0.016  # approximately for SPY

RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
PROC_DIR = os.path.join(OUTPUT_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)


def fetch_underlying_history(ticker, start, end):
    """Fetch Underlying history from yfinance"""
    
    print(f"Fetching underlying history for {ticker} from {start} to {end} ...")
    if os.path.exists(os.path.join(RAW_DIR, f"underlying_{ticker}.csv")):
        print("Processed underlying data already exists, loading from there.")
        return pd.read_csv(os.path.join(RAW_DIR, f"underlying_{ticker}.csv"), parse_dates=['date'])
    
    tkr = yf.Ticker(ticker)
    df = tkr.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No historical data for {ticker}")
    
    df = df.reset_index().rename(columns={"Date":"date"})
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['Spot'] = df['Close']
    df['log_return'] = np.log(df['Spot']).diff()
    df['return'] = df['Spot'].pct_change()
    df = df[['date','Open','High','Low','Close','Spot','Volume','log_return','return']].copy()
    
    under_out_proc = os.path.join(RAW_DIR, f"underlying_{ticker}.csv")
    df.to_csv(under_out_proc, index=False)
    print("Saved underlying to", under_out_proc)
    
    return df

def fetch_option_chain(ticker, start, end):
    """Fetch option chain from stored data"""
    
    print("Fetching option chain from stored data")
    if os.path.exists(os.path.join(RAW_DIR, f"options_process_data_{ticker}.csv")):
        print("Processed data already exists, loading from there.")
        return pd.read_csv(os.path.join(RAW_DIR, f"options_process_data_{ticker}.csv"), parse_dates=['date','expiration'])
    
    df_opts = pd.read_csv(os.path.join(RAW_DIR, f"options_chain_data.csv"), parse_dates=['date','expiration'])
    df = df_opts[df_opts['act_symbol'] == ticker].copy()
    df = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]
    if df.empty:
        raise RuntimeError(f"No option chain data for {ticker} in the given date range")
    
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['expiration'] = pd.to_datetime(df['expiration']).dt.date
    df = df.drop_duplicates()
    
    res = assign_tenor_series(df['date'], df['expiration'])
    res = res.drop_duplicates()

    df = df.merge(res[['date','expiration','tenor']], on=['date','expiration'], how='inner')
    df.rename(columns={'vol':'impliedVol', 'act_symbol':'ticker'}, inplace=True)
    df['mid'] = df[['bid','ask']].mean(axis=1)
    df['spread'] = (df['ask'] - df['bid'])
    df = df[(df['spread']/ (df['mid']+1e-9)) < 0.2]  # filter illiquid
    
    option_out_proc = os.path.join(RAW_DIR, f"options_process_data_{ticker}.csv")
    df.to_csv(option_out_proc, index=False)
    print("Saved processed options to", option_out_proc)
    
    return df

def fetch_iv_historical_data(ticker, start, end):
    """Fetch historical implied vol data from stored data"""
    
    print("Fetching historical implied vol data from stored data")
    if os.path.exists(os.path.join(RAW_DIR, f"iv_historical_data_{ticker}.csv")):
        print("Processed IV historical data already exists, loading from there.")
        return pd.read_csv(os.path.join(RAW_DIR, f"iv_historical_data_{ticker}.csv"), parse_dates=['date'])
    
    df_iv = pd.read_csv(os.path.join(RAW_DIR, f"options_vol_data.csv"), parse_dates=['date'])
    df_iv.rename(columns={'act_symbol':'ticker'}, inplace=True)
    
    df = df_iv[df_iv['ticker'] == ticker].copy()
    df = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df[["date","ticker","hv_current","iv_current"]].copy()
    
    vol_out_proc = os.path.join(RAW_DIR, f"iv_historical_data_{ticker}.csv")
    df.to_csv(vol_out_proc, index=False)
    print("Saved processed IV historical data to", vol_out_proc)
    
    return df

def compute_realized_vol(under_df, lookback=REALIZED_LOOKBACK_DAYS):
    """Compute rolling realized vol (annualized) using log returns."""
    df = under_df.copy()
    df['log_return'] = df['log_return'].astype(float)
    df['rv_30d'] = df['log_return'].rolling(window=lookback, min_periods=1).std() * math.sqrt(252)
    df['hv_30d_pct'] = df['return'].rolling(window=lookback, min_periods=1).std() * math.sqrt(252)
    df = df[['date','Spot','log_return','rv_30d','hv_30d_pct']]
    df = df.dropna()
    return df

def main():
    for ticker in TICKERS:
        print("=== Processing", ticker, "===")
        # underlying & options history
        under_df = fetch_underlying_history(ticker, START_DATE, END_DATE)
        options_df = fetch_option_chain(ticker, START_DATE, END_DATE)
        
        # compute realized vol
        rv_df = compute_realized_vol(under_df, lookback=REALIZED_LOOKBACK_DAYS)
        rv_csv = os.path.join(PROC_DIR, f"realized_vol_{ticker}.csv")
        rv_df.to_csv(rv_csv, index=False)
        print("Saved realized vol to", rv_csv)

        # historical IV data
        iv_hist_df = fetch_iv_historical_data(ticker, START_DATE, END_DATE)
        dfs = [iv_hist_df, rv_df[['date', 'Spot', 'rv_30d', 'hv_30d_pct']].rename(columns={'rv_30d':'realizedVol', 'hv_30d_pct':'historicalVol'}), options_df[['date', 'call_put', 'tenor', 'strike', 'impliedVol']]]
        vols_data = reduce(lambda  left, right: pd.merge(left, right, on='date', how='left'), dfs)
        vols_data = vols_data.dropna()
        vols_data_csv = os.path.join(PROC_DIR, f"vol_data_{ticker}.csv")
        vols_data.to_csv(vols_data_csv, index=False)
        print("Saved processed vol data to", vols_data_csv)

if __name__ == "__main__":
    main()