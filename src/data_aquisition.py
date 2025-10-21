"""
data_acquisition.py

Purpose:
- Pull underlying daily history for tickers (via yfinance)
- Pull current option chains (via yfinance) for each expiry
- Compute mid prices, time-to-expiry, forward estimate, log-moneyness
- Compute implied vols via Black-Scholes inversion
- Compute realized vol (21d rolling) from historical closes
- Save cleaned outputs to data/raw and data/processed
"""

import os
import math
from arb.black_scholes import implied_vol_from_price
from viz.utils import assign_tenor_series
from functools import reduce
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------
# Config
# ---------------------------
TICKERS = ["SPY"]      # change/add tickers (e.g., "AAPL", "^GSPC", "NIFTY 50" not supported by yfinance directly)
START_DATE = "2023-10-01"
END_DATE = "2025-10-15"
OUTPUT_DIR = "data"
RISK_FREE_RATE = 0.05   # default annual continuous risk-free approximation; replace with curve if available
REALIZED_LOOKBACK_DAYS = 30
DIVIDEND_YIELD = 0.016  # approximately for SPY

# Create folders
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
PROC_DIR = os.path.join(OUTPUT_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# ---------------------------
# Main pipeline
# ---------------------------
def fetch_underlying_history(ticker, start, end):
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
    # compute log returns and simple returns
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
    print(len(df), "option rows fetched for", ticker)
    res = assign_tenor_series(df['date'], df['expiration'])
    res = res.drop_duplicates()
    print("Assigned tenors:", len(res))
    df = df.merge(res[['date','expiration','tenor']],
              on=['date','expiration'], how='inner')
    print(len(df), "option rows after tenor assignment for", ticker)
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

def fetch_option_chain_snapshot(ticker):
    """Fetch current option chain via yfinance for all expiries (snapshot).
    Note: This is only a *current* snapshot. Historical snapshots require stored data or paid provider."""
    print(f"Fetching option chain snapshot for {ticker} ...")
    tkr = yf.Ticker(ticker)
    expiries = tkr.options
    rows = []
    spot = tkr.history(period="1d", auto_adjust=True).iloc[-1]['Close']
    info = tkr.info or {}
    div_yield = info.get('dividendYield', 0.0) or 0.0
    for exp in expiries:
        try:
            oc = tkr.option_chain(exp)
            calls = oc.calls.copy()
            puts = oc.puts.copy()
        except Exception as e:
            print(f"Failed to get chain for expiry {exp}: {e}")
            continue
        # common columns in yfinance: contractSymbol, lastTradeDate, strike, lastPrice, bid, ask, change, percentChange, volume, openInterest, impliedVol, inTheMoney
        for df_side, opt_type in [(calls, 'C'), (puts, 'P')]:
            if df_side.empty:
                continue
            for _, r in df_side.iterrows():
                strike = float(r['strike'])
                bid = float(r['bid']) if not pd.isna(r['bid']) else np.nan
                ask = float(r['ask']) if not pd.isna(r['ask']) else np.nan
                lastPrice = float(r['lastPrice']) if not pd.isna(r['lastPrice']) else np.nan
                mid = np.nan
                if not np.isnan(bid) and not np.isnan(ask):
                    mid = 0.5*(bid + ask)
                elif not np.isnan(lastPrice):
                    mid = lastPrice
                implied_vol = float(r['impliedVol']) if ('impliedVol' in r and not pd.isna(r['impliedVol'])) else np.nan
                rows.append({
                    'date': datetime.today().date(),
                    'expiry': pd.to_datetime(exp).date(),
                    'option_type': opt_type,
                    'strike': strike,
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'lastPrice': lastPrice,
                    'impliedVol_market': implied_vol,
                    'volume': r.get('volume', np.nan),
                    'openInterest': r.get('openInterest', np.nan),
                    'underlying_spot': spot,
                    'dividendYield': div_yield,
                    'source': 'yfinance'
                })
    df_opts = pd.DataFrame(rows)
    return df_opts

def enrich_and_compute_iv(options_df, r=RISK_FREE_RATE, q=DIVIDEND_YIELD):
    """Add T, forward approximation and compute implied vol for rows with mid price.
    Returns cleaned dataframe."""
    df = options_df.copy()
    # Use the latest underlying close (align by date where possible)
    # Here we have snapshot date = today; for historic snapshots you'd join on the snapshot date.
    # for each row compute T in year fraction, use underlying_spot present in options_df
    df['T'] = df.apply(lambda r0: max((r0['expiration'] - r0['date']).days / 365.0, 0.0), axis=1)
    df = df[df['T'] > 0.0001]
    # dividend yield
    df['q'] = q
    # Forward approx: F = S * exp((r - q) * T)
    df['forward'] = df.apply(lambda r0: r0['Spot'] * math.exp((r - r0['q']) * r0['T']), axis=1)
    # log-moneyness
    df['moneyness'] = df['strike'] / df['Spot']
    df['log_moneyness'] = np.log(df['strike'] / df['Spot']).astype(float)
    # compute implied vol where price_for_iv is present and T>0 (or compute approx at T==0)
    ivs = []
    for _, row in df.iterrows():
        price = row['mid']
        S = row['Spot']
        K = row['strike']
        T = row['T']
        qv = row['q'] if not pd.isna(row['q']) else 0.0
        ot = "C" if row["call_put"].lower().startswith("c") else "P"
        if pd.isna(price) or T < 0:
            ivs.append(np.nan)
            continue
        # If price is NaN, skip
        try:
            iv = implied_vol_from_price(ot, float(price), float(S), float(K), float(T), r, float(qv))
        except Exception:
            iv = np.nan
        ivs.append(iv)
    df['impliedVol_calc'] = ivs
    # df['impliedVol_calc'] = df['impliedVol_calc'].ffill()
    df = df.dropna()
    # total implied variance w = sigma^2 * T
    df['total_var'] = df['impliedVol_calc']**2 * df['T']
    # filter out fully NaN rows
    return df

def compute_realized_vol(under_df, lookback=REALIZED_LOOKBACK_DAYS):
    """Compute rolling realized vol (annualized) using log returns."""
    df = under_df.copy()
    # ensure log_return is in numeric
    df['log_return'] = df['log_return'].astype(float)
    # rolling std of log returns (sample), annualize
    df['rv_30d'] = df['log_return'].rolling(window=lookback, min_periods=1).std() * math.sqrt(252)
    # also compute simple historical vol over window (std of returns)
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

        combined_df = options_df.merge(
            under_df,
            on='date',
            how='left',
        )
        
        # compute realized vol
        rv_df = compute_realized_vol(under_df, lookback=REALIZED_LOOKBACK_DAYS)
        rv_csv = os.path.join(PROC_DIR, f"realized_vol_{ticker}.csv")
        rv_df.to_csv(rv_csv, index=False)
        print("Saved realized vol to", rv_csv)

        # Enrich & compute IVs
        opts_enriched = enrich_and_compute_iv(combined_df, r=RISK_FREE_RATE)
        proc_opts = os.path.join(PROC_DIR, f"options_cleaned_{ticker}.parquet")
        opts_enriched.to_parquet(proc_opts, index=False)
        print("Saved processed options to", proc_opts)

        # historical IV data
        iv_hist_df = fetch_iv_historical_data(ticker, START_DATE, END_DATE)
        dfs = [iv_hist_df, rv_df[['date', 'Spot', 'rv_30d', 'hv_30d_pct']].rename(columns={'rv_30d':'realizedVol', 'hv_30d_pct':'historicalVol'}), opts_enriched[['date', 'call_put', 'tenor', 'strike', 'impliedVol_calc', 'impliedVol', 'moneyness']]]
        vols_data = reduce(lambda  left, right: pd.merge(left, right, on='date', how='left'), dfs)
        vols_data['iv_err'] = vols_data['impliedVol_calc'] - vols_data['impliedVol']
        vols_data = vols_data.dropna()
        vols_data_csv = os.path.join(PROC_DIR, f"vol_data_{ticker}.csv")
        vols_data.to_csv(vols_data_csv, index=False)
        print("Saved processed vol data to", vols_data_csv)

if __name__ == "__main__":
    main()