import numpy as np
from scipy.interpolate import UnivariateSpline

def fit_spline_total_variance(log_moneyness, market_iv, T, weights=None, s=None):
    """fit spline to total variance w = iv^2 * T"""
    tv = (market_iv ** 2) * T
    spline = UnivariateSpline(log_moneyness, tv, w=weights, s=s)
    return spline

def iv_from_spline(spline, k_grid, T):
    """Get IV using spline"""
    w = spline(k_grid)
    w = np.maximum(w, 1e-12)
    iv = np.sqrt(w / T)
    return iv