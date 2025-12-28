import numpy as np

def second_difference_w(k, w):
    """Discrete second derivative of w(k) on non-uniform grid k"""
    n = len(k)
    sec = np.full(n, np.nan)
    for i in range(1, n-1):
        h1 = k[i] - k[i-1]
        h2 = k[i+1] - k[i]
        if h1 <= 0 or h2 <= 0:
            continue
        # second derivative approx for non-uniform spacing
        sec[i] = 2.0 * ( (w[i+1] - w[i]) / (h2*(h1+h2)) - (w[i] - w[i-1]) / (h1*(h1+h2)) )
    return sec

def check_butterfly_arbitrage(k, w, tol=1e-12):
    """Return indices where second derivative of w(k) is < -tol (i.e. convexity violated)"""
    sec = second_difference_w(k, w)
    bad_idx = np.where(sec < -tol)[0]
    return bad_idx, sec

def repair_convexity_local(k, w, max_iter=50, tol=1e-12):
    """
    Iteratively repair local convexity violations by replacing violating point with
    neighbor-average (or local convex projection)
    """
    w = w.copy().astype(float)
    n = len(w)
    changed = False
    
    for _ in range(max_iter):
        bad_idx, _ = check_butterfly_arbitrage(k, w, tol=tol)
        if len(bad_idx) == 0:
            break
        
        changed = True
        for i in bad_idx:
            if 0 < i < n-1:
                new_val = 0.5 * (w[i-1] + w[i+1])   # Local projection heuristic
                new_val = max(new_val, 1e-12)
                w[i] = new_val
    
    bad_idx_final, _ = check_butterfly_arbitrage(k, w, tol=tol)
    
    if len(bad_idx_final) > 0:
        try:
            from scipy.interpolate import UnivariateSpline
            spl = UnivariateSpline(k, w, k=3, s=1e-6 * n)
            w_smooth = spl(k)
            w = np.maximum(w_smooth, 1e-12)
            changed = True        
        except Exception:
            pass
    
    return w, changed
