import math
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from scripts.bootstrap import *
from models.svi import fit_svi
from models.sabr import calibrate_sabr
from models.spline import fit_spline_total_variance
from core.viz.utils import get_spot
from core.surface_pipeline import build_surface_single_expiry
from core.arbitrage.calendar import enforce_calendar_arbitrage

def fit_models(df, S, T, r, q):
    """Fit surface representations to arbitrage-free IVs"""
    k = np.log(df["strike"].values / S)
    iv = df["iv_arbitrage_free"].values

    results = {}

    spline = fit_spline_total_variance(k, iv, T) # Spline
    results["spline"] = spline

    try:
        svi_params, _ = fit_svi(k, iv, T)   # SVI
        results["svi"] = svi_params
    except Exception:
        results["svi"] = None

    try:
        F = S * np.exp((r-q) * T)
        sabr_params, _ = calibrate_sabr(F, df["strike"].values, T, iv)
        results["sabr"] = sabr_params
    except Exception:
        results["sabr"] = None

    return results

def enforce_calendar_across_expiries(surfaces_by_expiry, k_grid):
    """Enforce calendar arbitrage across expiries"""

    surfaces_by_expiry = sorted(surfaces_by_expiry, key=lambda x: x['T'])
    w_matrix = np.array([(s["iv_static_free"] ** 2) * s['T'] for s in surfaces_by_expiry])

    w_fixed, changed = enforce_calendar_arbitrage(k_grid, w_matrix)

    out = []
    for i, s in enumerate(surfaces_by_expiry):
        iv_fixed = np.sqrt(np.maximum(w_fixed[i] / s['T'], 1e-12))
        out.append({**s, "iv_arbitrage_free": iv_fixed})

    return out, changed

def run_surface_fit(df, close_series, k_grid, r=0.01, q=0.0):
    """Run complete surface fit + model fitting"""
    surfaces = []
    model_fits = []

    for expiry, df_exp in df.groupby("expiration"):
        S = get_spot(close_series, df_exp["date"].iloc[0])
        T = df_exp["T"].iloc[0]

        surface = build_surface_single_expiry(df_exp, S, r, q, T)
        
        k = np.log(surface["strike"].values / S)
        surfaces.append({
            "expiry": expiry,
            "T": T,
            "S": S,
            "df": surface,
            "k": k
        })
    
    surfaces_final, changed = enforce_calendar_across_expiries(surfaces, k_grid)

    for i, s in enumerate(surfaces_final):
        df_exp, T, S = s["df"].copy(), s["T"], s["S"]
        model_fit = fit_models(df_exp, S, T, r, q)
        model_fit["expiry"] = T
        model_fits.append(model_fit)

    return model_fits