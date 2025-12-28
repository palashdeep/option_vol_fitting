import math
import numpy as np
import pandas as pd

def get_spot(close_series, snap_date):
    """Return sopt price aligned to snapshot date"""
    ts = close_series.index.get_loc(snap_date, method="nearest")
    return float(close_series.iloc[ts])

def intrinsic_price(S, K, T, r, q, option_type):
    if option_type == 'C':
        return max(0.0, S * math.exp(-q*T) - K * math.exp(-r*T))
    else:
        return max(0.0, K * math.exp(-r*T) - S * math.exp(-q*T))

def is_quote_liquid(bid, ask, mid, S, K, T, r, q,
                    max_spread_pct=0.20, min_extrinsic_frac=0.001):
    """Checks if quote is liquid and provides a reason if not"""
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
    """Year fraction for given range"""
    return (end_date - start_date).days / 365.0

def forward_price_from_spot(S, r, q, T):
    """Get forward price"""
    return S * np.exp((r - q) * T)

def assign_tenor_series(dates, expiries, target_months=None):
    """Assign tenor labels (e.g., '1M', '3M', '6M', '1Y', '2Y', ... up to '10Y') to pairs of dates and expiries"""
    if target_months is None:
        target_months = [1, 2, 3, 6, 9] + [12 * y for y in range(1, 11)]
    
    dates = pd.to_datetime(dates)
    expiries = pd.to_datetime(expiries)
    
    days = (expiries - dates).dt.days
    months = (days / 30.436875).round().astype('Int64')  # Use average month length: 30.436875 days (365.2425 / 12)
    
    def nearest_bucket(m):
        if pd.isna(m) or m < 0:
            return pd.NA
        
        diffs = [abs(m - t) for t in target_months]
        best = target_months[diffs.index(min(diffs))]

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