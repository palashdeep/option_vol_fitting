import numpy as np
from scipy.interpolate import UnivariateSpline
# ---------------------------
# Spline on total variance
#    (fit total variance w(k) = iv^2 * T)
# ---------------------------
def fit_spline_total_variance(log_moneyness, market_iv, T, weights=None, s=None):
    # input arrays: log_moneyness k, market_iv (annual), T scalar
    # fit spline to total variance w = iv^2 * T
    tv = (market_iv ** 2) * T
    if weights is not None:
        w = weights
    else:
        # default vega-based weight may be better; here equal weights
        w = None
    # choose smoothing s: if None use small smoothing proportional to variance
    spline = UnivariateSpline(log_moneyness, tv, w=w, s=s)
    return spline

def iv_from_spline(spline, k_grid, T):
    w = spline(k_grid)
    w = np.maximum(w, 1e-12)
    iv = np.sqrt(w / T)
    return iv