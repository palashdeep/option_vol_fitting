import numpy as np
from scipy.optimize import least_squares
# ---------------------------
# 4) SVI parameterization & calibration
#    w(k) = a + b*(rho*(k - m) + sqrt((k-m)^2 + sigma^2))
#    constraints: b>=0, sigma>0, |rho|<1
# ---------------------------
def svi_total_variance(params, k):
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def svi_residuals(params, k, w_market, weights=None):
    a, b, rho, m, sigma = params
    # enforce simple bounds within residuals: if violate, penalize
    if b < 0 or sigma <= 0 or abs(rho) >= 0.9999:
        # large penalty vector
        return 1e6 * np.ones_like(k)
    w_model = svi_total_variance(params, k)
    if weights is None:
        return w_model - w_market
    else:
        return (w_model - w_market) * weights

def fit_svi(k, market_iv, T, weights=None, init=None):
    # market_iv: array, T scalar
    w_market = (market_iv**2) * T
    # initial guess heuristics
    if init is None:
        a0 = np.median(w_market) * 0.5
        b0 = max(0.01, (np.max(w_market)-np.min(w_market)) / 2.0)
        rho0 = 0.0
        m0 = 0.0
        sigma0 = 0.1
        init = [a0, b0, rho0, m0, sigma0]
    bounds_lower = [-1.0, 1e-8, -0.999, -5.0, 1e-6]
    bounds_upper = [5.0, 5.0, 0.999, 5.0, 5.0]
    res = least_squares(svi_residuals, x0=init, bounds=(bounds_lower, bounds_upper),
                        args=(k, w_market, weights), xtol=1e-10, ftol=1e-10)
    return res.x, res