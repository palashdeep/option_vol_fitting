import numpy as np

def check_vertical_arbitrage(prices, K):
    """
    Quick check: for calls: prices must be non-increasing with strike.
    Returns indices where monotonicity violated (i.e. price[i] < price[i+1] should hold).
    """
    bad = []
    for i in range(len(K)-1):
        if prices[i] < prices[i+1] - 1e-12:
            bad.append(i)
    return bad

def enforce_calendar_arbitrage(k_grid, w_grid_by_T):
    """
    w_grid_by_T: 2D array shape (n_T, n_k) where rows are ordered by increasing expiry.
    For each k index j, enforce non-decreasing w over T by taking cumulative max along axis 0.
    Returns adjusted w_grid and boolean mask of where increases occurred.
    """
    w = np.array(w_grid_by_T, dtype=float)
    # require rows sorted by increasing T before calling
    w_adj = np.maximum.accumulate(w, axis=0)
    changed = np.any(w_adj > w + 1e-12)
    return w_adj, changed