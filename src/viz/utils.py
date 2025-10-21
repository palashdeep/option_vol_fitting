import math
import numpy as np
import pandas as pd

# ---------------------------
# Black-Scholes helper functions
# ---------------------------
def intrinsic_price(S, K, T, r, q, option_type):
    if option_type == 'C':
        return max(0.0, S * math.exp(-q*T) - K * math.exp(-r*T))
    else:
        return max(0.0, K * math.exp(-r*T) - S * math.exp(-q*T))

def is_quote_liquid(bid, ask, mid, S, K, T, r, q,
                    max_spread_pct=0.20, min_extrinsic_frac=0.001):
    """
    Returns (bool, reason_str). Liquid if:
      - bid/ask present and mid between them,
      - spread <= max_spread_pct * mid,
      - extrinsic = mid - intrinsic >= min_extrinsic_frac * S
    """
    if np.isnan(mid) or np.isnan(bid) or np.isnan(ask):
        return False, "missing_quote"
    if not (bid <= mid <= ask + 1e-12):
        return False, "mid_outside_bidask"
    spread = ask - bid
    if mid <= 0:
        return False, "mid_nonpositive"
    if spread / max(mid, 1e-12) > max_spread_pct:
        return False, f"wide_spread:{spread/mid:.2f}"
    # extrinsic w.r.t intrinsic (assume call intrinsic; caller should compute correct option_type)
    intrinsic_c = max(0.0, S * math.exp(-q*T) - K * math.exp(-r*T))
    extrinsic = mid - intrinsic_c
    if extrinsic < min_extrinsic_frac * S:
        return False, f"low_extrinsic:{extrinsic:.6f}"
    return True, "liquid"
    
def year_frac_days(start_date, end_date):
    return (end_date - start_date).days / 365.0

def forward_price_from_spot(S, r, q, T):
    return S * np.exp((r - q) * T)

def rmse(a, b):
    ok = (~np.isnan(a)) & (~np.isnan(b))
    if ok.sum() == 0:
        return np.nan
    return math.sqrt(np.mean((a[ok] - b[ok])**2))

def rmse_vega(a, b, vega):
    ok = (~np.isnan(a)) & (~np.isnan(b)) & (vega > 0)
    if ok.sum() == 0:
        return np.nan
    w = vega[ok] / np.sum(vega[ok])
    return math.sqrt(np.sum(((a[ok] - b[ok])**2) * w))

def assign_tenor_series(dates, expiries, target_months=None):
    """
    Assign tenor labels (e.g., '1M', '3M', '6M', '1Y', '2Y', ... up to '10Y') to pairs of dates and expiries.
    - dates, expiries: pandas Series of dtype datetime64[ns]
    - target_months: list of integer month targets to snap to (defaults to common buckets up to 10 years)
    Returns a DataFrame with days_to_expiry, months_to_expiry (rounded), and tenor (nearest bucket).
    """
    if target_months is None:
        # Default standard buckets (in months): 1M,2M,3M,6M,9M,1Y,2Y..10Y
        target_months = [1, 2, 3, 6, 9] + [12 * y for y in range(1, 11)]
    
    # Ensure datetime dtype
    dates = pd.to_datetime(dates)
    expiries = pd.to_datetime(expiries)
    
    # Compute days and convert to months (using average month length)
    days = (expiries - dates).dt.days
    # Use average month length: 30.436875 days (365.2425 / 12)
    months = (days / 30.436875).round().astype('Int64')  # allow for NA
    
    def nearest_bucket(m):
        if pd.isna(m) or m < 0:
            return pd.NA
        # find nearest value in target_months
        diffs = [abs(m - t) for t in target_months]
        best = target_months[diffs.index(min(diffs))]
        # format label
        if best < 12:
            return f"{int(best)}M"
        else:
            years = best // 12
            return f"{int(years)}Y"
    
    tenor = months.apply(nearest_bucket)
    
    return pd.DataFrame({
        'date': dates.dt.date,
        'expiration': expiries.dt.date,
        'days_to_expiry': days,
        'months_rounded': months,
        'tenor': tenor
    })