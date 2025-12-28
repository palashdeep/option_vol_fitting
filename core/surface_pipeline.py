import numpy as np
from iv_inference import choose_iv_for_row
from arbitrage.butterfly import repair_convexity_local
from arbitrage.vertical import enforce_vertical_arbitrage_on_iv_grid

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

def enforce_static_arbitrage(df, S, r, T):
    """
    Enforce static arbitrage:
    - butterfly (convexity)
    - vertical (monotonicity)
    """
    df = df.sort_values("log_moneyness")

    k = np.log(df["strike"].values / S)
    iv = df["iv_chosen"].values
    w = iv**2 * T

    w_fixed, _ = repair_convexity_local(k, w) # Butterfly
    iv_fixed = np.sqrt(np.maximum(w_fixed / T, 1e-12))

    K = df["strike"].values
    iv_vfixed, _, _ = enforce_vertical_arbitrage_on_iv_grid(iv_fixed, K, S, T, r, option_type='C') # Vertical

    df = df.copy()
    df["iv_static_free"] = iv_vfixed
    
    return df

def build_surface_single_expiry(df, S, r, q, T):
    """Build arbitrage-free IV surface for single expiry"""
    df = infer_implied_vols(df, S, r, q)
    df = enforce_static_arbitrage(df, S, r, T)

    return df