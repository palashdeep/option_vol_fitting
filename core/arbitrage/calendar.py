import numpy as np

def enforce_calendar_arbitrage(k_grid, w_grid_by_T):
    """For each k index j, enforce non-decreasing w over T by taking cumulative max along axis 0"""
    if not np.all(np.diff(np.arange(w_grid_by_T.shape[0])) >= 0):
        raise ValueError("Rows must be ordered by increasing maturity")
    
    w = np.array(w_grid_by_T, dtype=float)
    w_adj = np.maximum.accumulate(w, axis=0)
    changed = np.any(w_adj > w + 1e-12)
    
    return w_adj, changed
