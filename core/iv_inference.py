import math
import numpy as np
import pandas as pd
from viz.utils import is_quote_liquid, intrinsic_price
from arbitrage.black_scholes import implied_vol_from_price, put_from_call

def choose_iv_for_row(row, df_snapshot, r=0.01, q=0.0, max_spread_pct=0.20, min_extrinsic_frac=0.001):
    """Decide which iv to use for each row"""
    K = float(row['strike'])
    T = float(row.get('T', 30.0/365.0))
    S = float(row.get('Spot', np.nan))
    if np.isnan(S):
        S = float(df_snapshot['strike'].median()) # If S missing, use median strike
    
    opt_type = 'C' if str(row['call_put']).lower().startswith('c') else 'P'
    bid = row.get('bid', np.nan)
    ask = row.get('ask', np.nan)
    mid = row.get('mid', np.nan)
    ds_iv = row.get('impliedVol', np.nan)

    # 1) try market mid directly if liquid
    liquid, reason = is_quote_liquid(bid, ask, mid, S, K, T, r, q, max_spread_pct=max_spread_pct, min_extrinsic_frac=min_extrinsic_frac)
    if liquid:
        iv_market = implied_vol_from_price(mid, S, K, T, r, q, option_type=opt_type)
        if not np.isnan(iv_market):
            return iv_market, "market_mid", "ok"
        # inversion failed (e.g., price out of BS bounds) -> mark and continue
        parity_reason = "inv_failed_market"
    else:
        parity_reason = reason

    # 2) parity fallback if this is a put and call market looks liquid
    if opt_type == 'P':
        call_row = df_snapshot[(df_snapshot['strike'] == K) & (df_snapshot['call_put'].str.lower().str.startswith('c'))]
        if not call_row.empty:
            call_row = call_row.iloc[0]
            call_bid = call_row.get('bid', np.nan)
            call_ask = call_row.get('ask', np.nan)
            call_mid = call_row.get('mid', np.nan)
            call_liquid, call_reason = is_quote_liquid(call_bid, call_ask, call_mid, S, K, T, r, q, max_spread_pct=max_spread_pct, min_extrinsic_frac=min_extrinsic_frac)
            
            if call_liquid and not np.isnan(call_mid):
                c_price = float(call_mid)
                p_via_call = put_from_call(c_price, S, K, T, r, q)
                intrinsic_put = intrinsic_price(S, K, T, r, q, 'P')
                
                if p_via_call + 1e-12 >= intrinsic_put: # Sanity check
                    iv_put_via_call = implied_vol_from_price(p_via_call, S, K, T, r, q, option_type='P')
                    if not np.isnan(iv_put_via_call):
                        return iv_put_via_call, "parity_from_call_market", f"parity_ok_call_liquid:{call_reason}"
                    else:
                        # parity inversion failed
                        parity_reason = "parity_inv_failed"
                else:
                    parity_reason = "parity_negative_price"
            else:
                parity_reason = f"call_not_liquid:{call_reason}"
        else:
            parity_reason = "no_call_row"

    # 3) fallback to dataset impliedVol
    if not pd.isnull(ds_iv):
        return float(ds_iv), "dataset", f"fallback_due_to:{parity_reason}"
    
    # 4) Lastly try returning the inverted mid even if not liquid (but mark it)
    iv_try = implied_vol_from_price(mid, S, K, T, r, q, option_type=opt_type) if not np.isnan(mid) else np.nan
    if not np.isnan(iv_try):
        return iv_try, "market_mid_unreliable", f"used_unreliable_quote:{parity_reason}"
    
    # No vol found
    return np.nan, "none", f"no_iv_found:{parity_reason}"