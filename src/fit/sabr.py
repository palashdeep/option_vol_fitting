import math
import numpy as np
from scipy.optimize import least_squares
# ---------------------------
# 5) SABR Hagan closed-form (lognormal formulation)
# ---------------------------
def sabr_hagan_iv(F, K, T, alpha, beta, rho, nu):
    # Robust implementation of Hagan's SABR approximation returning implied vol (lognormal)
    # Handles ATM separately (Taylor expansion)
    eps = 1e-12
    if F <= 0 or K <= 0:
        return np.nan
    one_minus_beta = 1.0 - beta
    if abs(F - K) < 1e-14:
        # ATM expansion
        FK_pow = F ** (one_minus_beta)
        term1 = alpha / FK_pow
        term2 = 1.0 + ( ( (one_minus_beta**2) / 24.0) * (alpha**2) / (FK_pow**2)
                       + (rho * beta * nu * alpha) / (4.0 * FK_pow)
                       + (nu**2 * (2.0 - 3.0 * rho**2) / 24.0) ) * T
        return term1 * term2
    logFK = math.log(F / K)
    FK = F * K
    z = (nu / alpha) * (FK ** (one_minus_beta / 2.0)) * logFK
    # avoid division by zero in x(z)
    denom_xz_num = math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho
    denom_xz_den = 1.0 - rho
    if denom_xz_num <= 0 or denom_xz_den <= 0:
        return np.nan
    x_z = math.log(denom_xz_num / denom_xz_den)
    # prefactor
    denom = (FK ** (one_minus_beta / 2.0)) * (1.0 + (one_minus_beta**2 / 24.0) * (logFK**2) + (one_minus_beta**4 / 1920.0) * (logFK**4))
    pre = (alpha / denom) * (z / x_z)
    # correction factor
    termA = (one_minus_beta**2 / 24.0) * (alpha**2) / (FK ** (one_minus_beta))
    termB = (rho * beta * nu * alpha) / (4.0 * (FK ** (one_minus_beta / 2.0)))
    termC = ((2.0 - 3.0 * rho**2) * nu**2 / 24.0)
    corr = 1.0 + (termA + termB + termC) * T
    return pre * corr

def sabr_residuals(x, F, K_arr, T, market_iv, weights=None, beta=0.5):
    alpha, rho, nu = x
    beta_local = beta
    model = np.array([sabr_hagan_iv(F, K, T, alpha, beta_local, rho, nu) for K in K_arr])
    res = model - market_iv
    if weights is not None:
        return res * weights
    return res

def calibrate_sabr(F, K_arr, T, market_iv, weights=None, beta=0.5, x0=None):
    # x0 guess: alpha ~ ATM_iv, rho ~ 0, nu ~ 0.3
    if x0 is None:
        # pick ATM index nearest forward
        idx = np.argmin(np.abs(K_arr - F))
        alpha0 = max(1e-6, market_iv[idx])
        x0 = [alpha0, 0.0, 0.3]
    lb = [1e-8, -0.999, 1e-8]
    ub = [5.0, 0.999, 5.0]
    res = least_squares(sabr_residuals, x0, bounds=(lb, ub), args=(F, K_arr, T, market_iv, weights, beta),
                        xtol=1e-12, ftol=1e-12, max_nfev=5000)
    return res.x, res