"""
Heston pricer (COS method) + calibration example.

Requirements:
    pip install numpy scipy

Functions:
- heston_cf(u, params, S, r, q, T): characteristic function of log-spot.
- cos_price_call(S, K_vec, r, q, T, heston_params, N=256, L=10): price calls via COS.
- implied_vol_from_price (Brent root).
- calibrate_heston(S, Ks, market_iv, r, q, T, hist_price_tol...): calibrator returning params.

Notes:
- heston_params = (kappa, theta, sigma, rho, v0)
- All vol numbers are on annualized basis.
- COS truncation interval uses cumulant-based L (default L=10).
"""

import numpy as np
import math
from scipy.optimize import brentq, differential_evolution, least_squares
from black_scholes import implied_vol_from_price

# -------------------------
# Heston characteristic function (log-spot)
# using formulation that is numerically stable for COS
# params: kappa, theta, sigma, rho, v0
# -------------------------
def heston_cf(u, params, S, r, q, T):
    """
    Characteristic function of log(S_T) under Heston:
    phi(u) = exp( C(T,u) + D(T,u)*v0 + i u * x0 )
    where x0 = ln(S)
    """
    kappa, theta, sigma, rho, v0 = params
    x0 = math.log(S)
    # complex unit
    i = 1j
    # parameters
    a = kappa * theta
    # d and g definitions (Heston original)
    # note: u is possibly vector; handle numpy arrays
    u = np.atleast_1d(u).astype(complex)
    # compute terms
    alpha = - (u * u + i * u)  # using characteristic of log-price? but standard derivation uses different sign; use standard Heston form
    # We'll implement standard Heston CF following widely-used form:
    # see e.g. Lord et al. or Gatheral: phi(u) = exp(C + D v0 + i u x0)
    # where:
    # d = sqrt((rho*sigma*i*u - kappa)^2 + sigma^2*(i*u + u^2))
    # g = (kappa - rho*sigma*i*u - d) / (kappa - rho*sigma*i*u + d)
    # C = i*u*(ln(F)) + (a/sigma^2)* ( (kappa - rho*sigma*i*u - d)*T - 2*log((1 - g*exp(-d*T)) / (1 - g)) )
    # D = (kappa - rho*sigma*i*u - d) / sigma^2 * (1 - exp(-d*T)) / (1 - g*exp(-d*T))
    # But careful with sign conventions. We'll use the proven implementation below.
    # Implement with vectorization:
    iu = i * u
    # intermediate
    b = kappa - rho * sigma * iu
    # d
    disc = b * b + (iu + u * u) * (sigma * sigma)
    d = np.sqrt(disc)
    # avoid sign ambiguity: choose branch with positive real part for d
    # g
    g = (b - d) / (b + d)
    # exponential terms
    exp_minus_dT = np.exp(-d * T)
    # C and D
    # A term (C)
    term1 = (b - d) * T - 2.0 * np.log((1 - g * exp_minus_dT) / (1 - g))
    C = (a / (sigma * sigma)) * term1 + iu * (x0 + (r - q) * T)
    D = (b - d) / (sigma * sigma) * (1 - exp_minus_dT) / (1 - g * exp_minus_dT)
    # phi
    phi = np.exp(C + D * v0)
    # return same shape as u
    if phi.size == 1:
        return phi[0]
    return phi

# -------------------------
# COS method for European call prices
# reference: Fang & Oosterlee (2008)
# -------------------------
def cos_price_call(S, K_vec, r, q, T, params, N=256, L=10):
    """
    COS pricing for European call options under Heston.
    - S: spot
    - K_vec: array-like strikes
    - r, q: rates
    - T: time to maturity (years)
    - params: Heston params (kappa, theta, sigma, rho, v0)
    - N: number of COS terms (power of two recommended)
    - L: truncation size multiplier (typical 8..12)
    Returns: numpy array of call prices (same length as K_vec)
    """
    # Ensure numpy arrays
    Ks = np.array(K_vec, dtype=float)
    x0 = math.log(S)
    # characteristic function grid
    u = np.arange(N) * 1.0  # u_k = k
    # cumulants for truncation range (approx)
    # first cumulant c1 = E[log(S_T)] ≈ ln(S) + (r - q) * T + (1/kappa)*(1 - exp(-kappa*T))*(theta - v0) ??? 
    # Simpler practical approach: use moment-matching approx:
    # use Heston cumulants approximations (see literature). For robustness use:
    kappa, theta, sigma, rho, v0 = params
    # c1:
    c1 = x0 + (r - q) * T + (kappa * theta - kappa * v0) * ( (T - (1 - math.exp(-kappa*T))/kappa) / (kappa) ) if kappa != 0 else x0 + (r-q)*T
    # but to avoid complexity, use common approximation for truncation (see Fang&Oosterlee): use log-spot mean and variance.
    # We'll compute approximate variance c2 via integrating variance process: Var(log S_T) ≈ ?
    # Simpler robust fallback: set large range using S and L
    # Use range [a,b] = [c1 - L*sqrt(c2), c1 + L*sqrt(c2)] with c2 roughly = max(0.1, theta*T*2)
    c2 = max(1e-6, theta * T * 2.0 + v0 * T * 0.1)  # crude but safe
    a = c1 - L * math.sqrt(c2)
    b = c1 + L * math.sqrt(c2)
    # Prepare k grid and U
    k = np.arange(N)
    u_k = k * math.pi / (b - a)  # frequency grid used in cosine expansion
    # Evaluate characteristic function at u_k (shifted by -i to handle e^{i u x})
    # Cos method expects phi(u) of log-price; we pass u = u_k
    cf_vals = heston_cf(u_k, params, S, r, q, T)  # vector of length N
    # Compute Fourier-cosine coefficients of payoff (call)
    # For call payoff f(x) = max(exp(x) - K, 0) transform to coefficients (see Fang & Oosterlee)
    # Define chi and psi functions used to compute coefficients
    def chi(k, a, b, c, d):
        # chi_k(a,b) = integral_c^d e^x cos(k*pi*(x-a)/(b-a)) dx
        # closed form:
        kp = k * math.pi / (b - a)
        res = np.zeros_like(k, dtype=float) if isinstance(k, np.ndarray) else 0.0
        # vectorized:
        if np.isscalar(k):
            kp = float(kp)
            res = (math.exp(d) * (kp * math.cos(kp*(d - a)) + math.sin(kp*(d - a))) - 
                   math.exp(c) * (kp * math.cos(kp*(c - a)) + math.sin(kp*(c - a)))) / (1 + kp*kp)
            return res
        else:
            kp = k * math.pi / (b - a)
            res = (np.exp(d) * (kp * np.cos(kp*(d - a)) + np.sin(kp*(d - a))) - 
                   np.exp(c) * (kp * np.cos(kp*(c - a)) + np.sin(kp*(c - a)))) / (1.0 + kp*kp)
            return res

    def psi(k, a, b, c, d):
        kp = k * math.pi / (b - a)
        # integral of cos = (sin term)/kp except k=0 case
        if np.isscalar(k):
            if abs(kp) < 1e-12:
                return d - c
            return (math.sin(kp*(d - a)) - math.sin(kp*(c - a))) / kp
        else:
            kp_arr = kp
            res = np.empty_like(kp_arr, dtype=float)
            zero_mask = np.abs(kp_arr) < 1e-12
            res[zero_mask] = d - c
            res[~zero_mask] = (np.sin(kp_arr[~zero_mask]*(d - a)) - np.sin(kp_arr[~zero_mask]*(c - a))) / kp_arr[~zero_mask]
            return res

    # For each strike K, compute coefficients V_k
    prices = np.zeros_like(Ks, dtype=float)
    factor = 2.0 / (b - a)
    for idx, K in enumerate(Ks):
        c = math.log(K)
        d = b  # in call transform, upper limit is b
        # compute coefficients Uk
        # note: for call, integration from c to b (where exp(x) - K positive)
        # see Fang & Oosterlee (2008) eqns
        chi_k = chi(k, a, b, c, d)  # vector length N
        psi_k = psi(k, a, b, c, d)
        Vk = 2.0 / (b - a) * (chi_k - K * psi_k)
        # adjust k=0 term
        Vk[0] *= 0.5
        # multiplication with real part of cf*exp(-i u a)
        exp_minus_iu_a = np.exp(-1j * u_k * a)
        # phi(u_k) * exp(-i*u_k*a)
        temp = cf_vals * exp_minus_iu_a
        # take real part
        temp_real = np.real(temp)
        # price in log-space; final price = exp(-rT) * sum_{k} Re( phi(u_k) * exp(-i u_k a) ) * Vk
        price = math.exp(-r * T) * np.sum(temp_real * Vk)
        prices[idx] = price
    return prices

# -------------------------
# Helper: convert Heston prices -> implied vols
# -------------------------
def heston_iv_surface_from_params(S, Ks, r, q, T, params, N=256, L=10):
    # produce model IVs for vector of Ks
    prices = cos_price_call(S, Ks, r, q, T, params, N=N, L=L)
    ivs = np.array([implied_vol_from_price(p, S, K, T, r, option_type='C') for p, K in zip(prices, Ks)])
    return ivs, prices

# -------------------------
# Calibration wrapper
# -------------------------
def calibrate_heston(S, Ks, market_iv, r, q, T,
                     bounds = ((1e-3, 0.0001, 1e-3, -0.99, 1e-4), (10.0, 5.0, 5.0, 0.99, 5.0)),
                     popsize=15, maxiter=30, Ncos=256, L=10, verbose=False):
    """
    Calibrate Heston params to market implied vol surface for a single expiry T.
    Procedure:
      1) minimize objective on implied vol differences (vega-weighted or simple) using differential_evolution (global)
      2) refine with least_squares local optimizer
    Returns: params (kappa, theta, sigma, rho, v0), diagnostics
    """
    Ks = np.array(Ks)
    market_iv = np.array(market_iv)

    # objective: RMSE on IV (can weight by vega or 1)
    def obj_de(x):
        kappa, theta, sigma, rho, v0 = x
        # enforce positivity constraints already by bounds
        params = (kappa, theta, sigma, rho, v0)
        try:
            model_iv, _ = heston_iv_surface_from_params(S, Ks, r, q, T, params, N=Ncos, L=L)
        except Exception:
            return 1e6
        # compute error; ignore NaNs in model
        ok = ~np.isnan(model_iv) & ~np.isnan(market_iv)
        if ok.sum() == 0:
            return 1e6
        # e.g. weight by 1 / iv^2 to not overweight OTM large ivs
        w = np.ones_like(model_iv)
        err = (model_iv[ok] - market_iv[ok]) * w[ok]
        return np.sqrt(np.mean(err**2))

    # differential evolution global search
    if verbose:
        print("Starting differential evolution search...")
    result_de = differential_evolution(obj_de, bounds=list(zip(bounds[0], bounds[1])), maxiter=maxiter, popsize=popsize, polish=True)
    x0 = result_de.x
    if verbose:
        print("DE finished. x0:", x0)

    # refine with least squares on implied vol residuals
    def resid_ls(x):
        params = tuple(x)
        model_iv, _ = heston_iv_surface_from_params(S, Ks, r, q, T, params, N=Ncos, L=L)
        # compute residuals (market - model)
        res = (model_iv - market_iv)
        # mask NaN
        res[np.isnan(res)] = 0.0
        return res

    if verbose:
        print("Refining with least squares...")
    ls_bounds = (bounds[0], bounds[1])
    res_ls = least_squares(resid_ls, x0, bounds=ls_bounds, ftol=1e-8, xtol=1e-8, max_nfev=200)
    params_opt = tuple(res_ls.x)
    # produce final model IV and prices
    model_iv_opt, prices_opt = heston_iv_surface_from_params(S, Ks, r, q, T, params_opt, N=Ncos, L=L)
    # metrics
    ok = ~np.isnan(model_iv_opt) & ~np.isnan(market_iv)
    rmse = np.nan
    if ok.sum() > 0:
        rmse = math.sqrt(np.mean((model_iv_opt[ok] - market_iv[ok])**2))
    return {
        'params': params_opt,
        'res_de': result_de,
        'res_ls': res_ls,
        'model_iv': model_iv_opt,
        'model_price': prices_opt,
        'rmse': rmse
    }