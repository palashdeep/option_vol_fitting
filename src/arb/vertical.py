import numpy as np
import math
from black_scholes import bs_price, implied_vol_from_price

def pava_nonincreasing(y):
    """
    Pool Adjacent Violators Algorithm (PAVA) for monotone non-increasing sequences.
    Returns y_adj which is non-increasing and is L2 projection of y onto that set.
    Implementation transforms to non-decreasing problem by negation.
    """
    # transform to non-decreasing problem by negation
    y_neg = [-val for val in y]
    n = len(y_neg)
    # levels (blocks) as lists of (cum_sum, cum_count)
    sums = []
    counts = []
    for i in range(n):
        # start new block with y_neg[i]
        sums.append(y_neg[i])
        counts.append(1)
        # merge backwards while violation (block average decreases)
        while len(sums) >= 2 and (sums[-2] / counts[-2]) > (sums[-1] / counts[-1]):
            # merge last two
            s2 = sums.pop()
            c2 = counts.pop()
            s1 = sums.pop()
            c1 = counts.pop()
            sums.append(s1 + s2)
            counts.append(c1 + c2)
    # now expand blocks back to sequence
    y_adj_neg = []
    for s, c in zip(sums, counts):
        avg = s / c
        y_adj_neg.extend([avg] * c)
    # revert negation
    y_adj = [ -v for v in y_adj_neg ]
    return np.array(y_adj, dtype=float)

def check_vertical_arbitrage_prices(K, call_prices, tol=1e-12):
    """
    Return indices where price increases with strike (call_price[i] < call_price[i+1] - tol).
    Expects K sorted ascending. Returns list of violating indices (i where violation between i and i+1).
    """
    bad = []
    for i in range(len(K)-1):
        if call_prices[i] < call_prices[i+1] - tol:
            bad.append(i)
    return bad

def repair_vertical_by_pava(K, call_prices):
    """
    Apply PAVA to produce non-increasing call prices and return repaired prices + count of changes.
    """
    # ensure K sorted ascending; PAVA will preserve index order
    repaired = pava_nonincreasing(call_prices)
    # count violations corrected
    violations = len(check_vertical_arbitrage_prices(K, call_prices))
    changed_count = int(np.sum(np.abs(repaired - call_prices) > 1e-12))
    return repaired, violations, changed_count

# -----------------------------
# Helper: convert model IV grid -> call prices -> repair -> back to IV
# -----------------------------
def enforce_vertical_arbitrage_on_iv_grid(iv_grid, K_grid, S, T, r, option_type='C'):
    """
    iv_grid: array of model IVs aligned with K_grid (same length)
    K_grid: strikes ascending
    Returns: iv_fixed (same shape), price_fixed, flags dict
    """
    # compute model prices from iv_grid
    prices = np.array([bs_price(S, K, T, r, sigma, option_type) if (not np.isnan(sigma)) else np.nan
                       for K, sigma in zip(K_grid, iv_grid)])
    # If NaNs present in prices (bad iv), fall back to forward filling or drop - here we replace nan by 0 to be conservative
    nan_mask = np.isnan(prices)
    if np.any(nan_mask):
        # replace NaNs with small intrinsic price (conservative)
        prices[nan_mask] = np.maximum(0.0, S - K_grid[nan_mask] * math.exp(-r * T))
    bad_idx = check_vertical_arbitrage_prices(K_grid, prices)
    flags = {'vertical_violations': len(bad_idx), 'vertical_fixed': False}
    if len(bad_idx) == 0:
        return iv_grid.copy(), prices.copy(), flags

    # repair using PAVA
    repaired_prices, violations, changed_count = repair_vertical_by_pava(K_grid, prices)
    flags['vertical_violations'] = violations
    flags['vertical_fixed'] = True if changed_count > 0 else False

    # invert repaired prices back to ivs (use implied_vol_from_price per strike)
    iv_fixed = np.empty_like(iv_grid, dtype=float)
    for i, (K, p) in enumerate(zip(K_grid, repaired_prices)):
        # keep numeric safety: price cannot exceed forward intrinsic upper bound; implied solver may fail
        try:
            iv = implied_vol_from_price(p, S, K, T, r, option_type=option_type)
            if np.isnan(iv):
                # fallback: small iv if price ~ intrinsic else use previous iv
                intrinsic = max(0.0, S - K * math.exp(-r * T))
                if p <= intrinsic + 1e-12:
                    iv = 1e-6
                else:
                    # try to bracket differently
                    iv = implied_vol_from_price(min(p, S), S, K, T, r, option_type=option_type)
            iv_fixed[i] = iv if not np.isnan(iv) else 1e-6
        except Exception:
            iv_fixed[i] = 1e-6
    return iv_fixed, repaired_prices, flags