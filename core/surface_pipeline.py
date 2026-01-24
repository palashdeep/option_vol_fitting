import numpy as np
from core.iv_inference import choose_iv_for_row
from core.arbitrage.butterfly import repair_convexity_local
from core.arbitrage.vertical import enforce_vertical_arbitrage_on_iv_grid
from core.arbitrage.black_scholes import call_from_put
import pandas as pd

def infer_implied_vols(df, S, r, q):
    """
    Infer implied vol per option using:
    1. market mid if liquid
    2. parity reconstruction
    3. dataset fallback
    """
    ivs, sources, flags = [], [], []

    for _, row in df.iterrows():
        iv, src, flag = choose_iv_for_row(row, df, r, q)
        ivs.append(iv)
        sources.append(src)
        flags.append(flag)
    
    df = df.copy()
    df["iv_chosen"] = ivs
    df["iv_source"] = sources
    df["iv_flag"] = flags
    
    return df.dropna(subset=["iv_chosen"])

def enforce_static_arbitrage(df, S, T, r, q):
    """
    Enforce static arbitrage:
    - butterfly (convexity)
    - vertical (monotonicity)
    """
    F = S * np.exp(r * T)

    calls = df[(df["call_put"] == "Call") & (df["strike"] >= F)]
    puts  = df[(df["call_put"] == "Put")  & (df["strike"] <  F)]

    puts = puts.copy()
    puts["price_call"] = call_from_put(
        puts["mid"].values, S, puts["strike"].values, T, r, q
    )

    calls = calls.copy()
    calls["price_call"] = calls["mid"].values

    df_calls = pd.concat([calls, puts], axis=0)
    df_calls = df_calls.sort_values("log_moneyness")

    k = np.log(df_calls["strike"].values / S)
    iv = df_calls["iv_chosen"].values
    w = iv**2 * T

    w_fixed, _ = repair_convexity_local(k, w) # Butterfly
    iv_fixed = np.sqrt(np.maximum(w_fixed / T, 1e-12))

    K = df_calls["strike"].values
    iv_vfixed, _, _ = enforce_vertical_arbitrage_on_iv_grid(iv_fixed, K, S, T, r, option_type='C') # Vertical

    df_calls = df_calls.copy()
    df_calls["iv_static_free"] = iv_vfixed

    return df_calls

def build_surface_single_expiry(df, S, T, r, q):
    """Build arbitrage-free IV surface for single expiry"""
    
    df = infer_implied_vols(df, S, r, q)
    df = enforce_static_arbitrage(df, S, T, r, q)

    return df