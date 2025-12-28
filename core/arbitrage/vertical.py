import numpy as np
import math
from black_scholes import bs_price, implied_vol_from_price

def _pava_nonincreasing(y):
    """Pool Adjacent Violators Algorithm (PAVA) for monotone non-increasing sequences"""   
    
    y_neg = [-val for val in y] # transform to non-decreasing problem by negation
    n = len(y_neg)
    sums = []
    counts = []
    
    for i in range(n):
        sums.append(y_neg[i])
        counts.append(1)
        while len(sums) >= 2 and (sums[-2] / counts[-2]) > (sums[-1] / counts[-1]):
            s2 = sums.pop()
            c2 = counts.pop()
            s1 = sums.pop()
            c1 = counts.pop()
            sums.append(s1 + s2)
            counts.append(c1 + c2)
    
    y_adj_neg = []
    for s, c in zip(sums, counts):
        avg = s / c
        y_adj_neg.extend([avg] * c)

    y_adj = [ -v for v in y_adj_neg ]
    return np.array(y_adj, dtype=float)

def _check_vertical_prices(K, call_prices, tol=1e-12):
    """Return violation indices where price increases with strike"""
    bad = []
    for i in range(len(K)-1):
        if call_prices[i] < call_prices[i+1] - tol:
            bad.append(i)
    
    return bad

def _repair_vertical_by_pava(K, call_prices):
    """Apply PAVA to produce non-increasing call prices and return repaired prices + count of changes"""
    
    violations = len(_check_vertical_prices(K, call_prices))
    repaired = _pava_nonincreasing(call_prices) # PAVA preserves index order
    changed_count = int(np.sum(np.abs(repaired - call_prices) > 1e-12))
    
    return repaired, violations, changed_count

def enforce_vertical_arbitrage_on_iv_grid(iv_grid, K_grid, S, T, r, option_type='C'):
    """
    Fixes for vertical arbitrage on the given IV grid
    iv_grid: array of model IVs aligned with K_grid
    K_grid: strikes ascending
    """
    prices = np.array([bs_price(S, K, T, r, sigma=sigma, option_type=option_type) if (not np.isnan(sigma)) else np.nan for K, sigma in zip(K_grid, iv_grid)])
    nan_mask = np.isnan(prices)
    
    if np.any(nan_mask):
        prices[nan_mask] = np.maximum(0.0, S - K_grid[nan_mask] * math.exp(-r * T)) # replace NaNs with small intrinsic price (conservative)
    
    bad_idx = _check_vertical_prices(K_grid, prices)
    flags = {'vertical_violations': len(bad_idx), 'vertical_fixed': False}
    
    if len(bad_idx) == 0:
        return iv_grid.copy(), prices.copy(), flags

    repaired_prices, violations, changed_count = _repair_vertical_by_pava(K_grid, prices)
    flags['vertical_violations'] = violations
    flags['vertical_fixed'] = True if changed_count > 0 else False

    iv_fixed = np.empty_like(iv_grid, dtype=float)
    for i, (K, p) in enumerate(zip(K_grid, repaired_prices)):
        try:
            iv = implied_vol_from_price(p, S, K, T, r, option_type=option_type)
            if np.isnan(iv):
                # fallback: small iv if price ~ intrinsic else use previous iv
                intrinsic = max(0.0, S - K * math.exp(-r * T))
                if p <= intrinsic + 1e-12:
                    iv = 1e-6
                else:
                    iv = implied_vol_from_price(min(p, S), S, K, T, r, option_type=option_type)
            iv_fixed[i] = iv if not np.isnan(iv) else 1e-6
        except Exception:
            iv_fixed[i] = 1e-6
    
    return iv_fixed, repaired_prices, flags