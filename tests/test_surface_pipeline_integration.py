import numpy as np
import pandas as pd
from core.arbitrage.black_scholes import bs_price
from core.surface_pipeline import build_surface_single_expiry

def synthetic_option_surface_with_violations():
    """
    Construct a synthetic single-expiry option surface
    with deliberate static arbitrage violations
    """
    S = 100.0
    T = 0.5
    r = 0.01
    q = 0.0

    strikes = np.array([80, 90, 100, 110, 120])

    # Deliberately non-convex implied vol smile
    implied_vol = np.array([0.30, 0.22, 0.15, 0.23, 0.32])

    prices = []
    for K, iv in zip(strikes, implied_vol):
        price = bs_price(
            option_type="C",
            S=S,
            K=K,
            T=T,
            r=r,
            q=q,
            sigma=iv
        )
        prices.append(price)

    df = pd.DataFrame({
        "date": pd.Timestamp("2024-01-01"),
        "expiration": pd.Timestamp("2024-07-01"),
        "call_put": "Call",
        "strike": strikes,
        "bid": np.array(prices) * 0.99,
        "ask": np.array(prices) * 1.01,
        "impliedVol": implied_vol,
        "T": T,
        "log_moneyness": np.log(strikes / S)
    })

    return df

def test_surface_pipeline_runs():
    df = synthetic_option_surface_with_violations()
    surface = build_surface_single_expiry(df, S=100, T=0.5, r=0.01, q=0.0)

    assert "iv_static_free" in surface.columns
    assert np.all(surface["iv_static_free"] > 0)