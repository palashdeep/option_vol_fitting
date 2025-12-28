import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(S, K, T, r, q=0, sigma=0.05, option_type='C'):
    """Black-Scholes price (European) with continuous dividend yield q.
    option_type: 'C' or 'P'
    S: spot
    K: strike
    T: time to expiry in years
    r: risk-free rate (annual, continuous)
    q: dividend yield (annual, continuous)
    sigma: vol (annual)
    """
    if T <= 0:
        if option_type == 'C':
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
    if sigma <= 0:
        if option_type == 'C':
            return math.exp(-q*T)*S - math.exp(-r*T)*K if S*math.exp(-q*T) > K*math.exp(-r*T) else 0.0
        else:
            return math.exp(-r*T)*K - math.exp(-q*T)*S if K*math.exp(-r*T) > S*math.exp(-q*T) else 0.0

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'C':
        return math.exp(-q*T) * S * norm.cdf(d1) - math.exp(-r*T) * K * norm.cdf(d2)
    else:
        return math.exp(-r*T) * K * norm.cdf(-d2) - math.exp(-q*T) * S * norm.cdf(-d1)
    
def bs_vega(S, K, T, r, sigma):
    """Vega calculation"""
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol_from_price(price, S, K, T, r, q=0.0, sigma_bounds=(1e-6, 5.0), option_type='C'):
    """Invert Black-Scholes price -> implied vol using root-finding (Brent). Returns np.nan if no solution."""
    intrinsic = max(0.0, (S * math.exp(-q*T) - K * math.exp(-r*T)) if option_type == 'C' else (K * math.exp(-r*T) - S * math.exp(-q*T)))
    
    if price < intrinsic - 1e-12 or price <= intrinsic + 1e-3:
        return np.nan
    
    try:
        def objective(sigma):
            return bs_price(S, K, T, r, q, sigma, option_type) - price
        
        a, b = sigma_bounds
        fa, fb = objective(a), objective(b)
        
        if fa * fb > 0:
            for b_try in [10.0, 20.0]:
                fb = objective(b_try)
                if fa * fb <= 0:
                    b = b_try
                    break
            else:
                return np.nan
        
        vol = brentq(objective, a, b, xtol=1e-8, maxiter=200)
    
        return vol
    
    except Exception:
        return np.nan
    
def put_from_call(call_price, S, K, T, r, q):
    """Reconstruct put price from call price using put-call parity"""
    
    return call_price - S * math.exp(-q * T) + K * math.exp(-r * T)


def call_from_put(put_price, S, K, T, r, q):
    """Reconstruct call price from put price using put-call parity"""
    
    return put_price + S * math.exp(-q * T) - K * math.exp(-r * T)
