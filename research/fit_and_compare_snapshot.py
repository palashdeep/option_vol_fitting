import math
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from scripts.bootstrap import *
from models.svi import svi_total_variance, fit_svi
from models.sabr import sabr_hagan_iv, calibrate_sabr
from models.spline import fit_spline_total_variance, iv_from_spline
from models.heston import calibrate_heston, heston_iv_surface_from_params
from core.arbitrage.butterfly import check_butterfly_arbitrage, repair_convexity_local
from core.arbitrage.calendar import enforce_calendar_arbitrage
from core.arbitrage.vertical import enforce_vertical_arbitrage_on_iv_grid
from core.viz.utils import year_frac_days, forward_price_from_spot, rmse, rmse_vega, is_quote_liquid, intrinsic_price
from core.arbitrage.black_scholes import implied_vol_from_price, bs_vega

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

# ---------------------------
# End-to-end per-snapshot pipeline
# ---------------------------

def fit_and_compare_snapshot(df_snap, vol_hist_df, close_series, r=0.01, q=0.0, beta_sabr=0.5,
                             grid_points=80, do_heston=True, heston_opts=None,
                             max_spread_pct=0.20, min_extrinsic_frac=0.001):
    """
    Revised fit_and_compare_snapshot:
    - Uses choose_iv_for_row logic to pick iv per option row (market mid, parity-from-call, or dataset fallback)
    - Fits Spline / SVI / SABR / Heston per expiry and stores model grids
    - Enforces calendar arbitrage across expiries by resampling total variance to common k_grid
      and calling enforce_calendar_arbitrage (user provided)
    - Recomputes metrics and repairs after calendar enforcement
    """
    # ----- PASS 1: per-expiry fit & store results for calendar enforcement -----
    # We'll collect per-expiry dictionaries and then enforce calendar arbitrage across expiries
    expiry_infos = []  # list of dicts for each expiry
    snapshot_date = None

    # iterate expiries and compute iv_chosen etc
    for expiry, sub in df_snap.groupby('expiration'):
        sub = sub.copy()
        sub = sub.dropna(subset=['bid', 'ask', 'strike'])
        if sub.empty:
            continue
        sub['mid'] = (sub['bid'] + sub['ask']) / 2.0
        sub['T'] = sub.apply(lambda rrow: (pd.to_datetime(rrow['expiration']) - pd.to_datetime(rrow['date'])).days / 365.0, axis=1)
        sub = sub[sub['T'] > 1/365.0]
        if sub.shape[0] < 6:
            continue
        if snapshot_date is None:
            snapshot_date = sub['date'].iloc[0]
        try:
            S = float(close_series.loc[close_series.index.get_loc(pd.to_datetime(snapshot_date), method='nearest')])
        except Exception:
            S = float(sub['strike'].median())
        T0 = float(sub['T'].iloc[0])
        F_est = S * math.exp((r - q) * T0)

        # annotate Spot and T into sub so choose_iv_for_row can use them
        sub['Spot'] = S
        sub['T'] = T0

        # pick iv per row using market/parity/dataset fallback
        ivs = []
        iv_sources = []
        iv_flags = []
        for _, row in sub.iterrows():
            iv_val, src, flag = choose_iv_for_row(row, sub, r=r, q=q)
            ivs.append(iv_val)
            iv_sources.append(src)
            iv_flags.append(flag)
        sub['iv_chosen'] = ivs
        sub['iv_source'] = iv_sources
        sub['iv_flag'] = iv_flags

        # drop rows where iv_chosen is NaN
        sub = sub.dropna(subset=['iv_chosen'])
        if sub.shape[0] < 6:
            continue

        sub['log_moneyness'] = np.log(sub['strike'] / S)
        sub['vega'] = sub.apply(lambda rrow: bs_vega(S, rrow['strike'], rrow['T'], r, rrow['iv_chosen']), axis=1)
        sub = sub[sub['vega'] > 1e-8]
        if sub.shape[0] < 6:
            continue

        sub = sub.sort_values('log_moneyness')
        k_arr = sub['log_moneyness'].values
        K_arr = sub['strike'].values
        market_iv = sub['iv_chosen'].values
        vega_arr = sub['vega'].values
        weights = vega_arr / (np.sum(vega_arr) + 1e-12)

        # fit models (Spline, SVI, SABR, Heston) and produce iv grids per expiry
        # Spline
        spline = fit_spline_total_variance(k_arr, market_iv, T0, s=None, weights=None)
        # store a per-expiry grid (we will later resample to common_k_grid)
        k_grid = np.linspace(k_arr.min() - 0.05, k_arr.max() + 0.05, grid_points)
        iv_spline_grid = iv_from_spline(spline, k_grid, T0)
        iv_spline_at_K = iv_from_spline(spline, np.log(K_arr / S), T0)

        # SVI
        svi_params = None; iv_svi_grid = None; iv_svi_at_K = None
        try:
            svi_params, svi_res = fit_svi(k_arr, market_iv, T0, weights=weights, init=None)
            w_svi_grid = svi_total_variance(svi_params, k_grid)
            iv_svi_grid = np.sqrt(np.maximum(w_svi_grid, 1e-12) / T0)
            iv_svi_at_K = np.sqrt(np.maximum(svi_total_variance(svi_params, np.log(K_arr / S)), 1e-12) / T0)
        except Exception:
            svi_params = None

        # SABR
        sabr_params = None; iv_sabr_grid = None; iv_sabr_at_K = None
        try:
            sabr_params, sabr_res = calibrate_sabr(F_est, K_arr, T0, market_iv, weights=weights, beta=beta_sabr)
            alpha, rho, nu = sabr_params
            iv_sabr_grid = np.array([sabr_hagan_iv(F_est, K, T0, alpha, beta_sabr, rho, nu) for K in (np.exp(k_grid) * S)])
            iv_sabr_at_K = np.array([sabr_hagan_iv(F_est, K, T0, alpha, beta_sabr, rho, nu) for K in K_arr])
        except Exception:
            sabr_params = None

        # Heston (optional)
        iv_heston_grid = None; iv_heston_at_K = None; heston_info = None
        if do_heston:
            try:
                if heston_opts is None:
                    heston_opts = {}
                heston_opts_local = {
                    'bounds': heston_opts.get('bounds', ((1e-3, 1e-4, 1e-3, -0.99, 1e-4),(10.0,5.0,5.0,0.99,5.0)) if heston_opts else ((1e-3, 1e-4, 1e-3, -0.99, 1e-4),(10.0,5.0,5.0,0.99,5.0))),
                    'popsize': heston_opts.get('popsize', 6) if heston_opts else 6,
                    'maxiter': heston_opts.get('maxiter', 6) if heston_opts else 6,
                    'Ncos': heston_opts.get('Ncos', 128) if heston_opts else 128,
                    'L': heston_opts.get('L', 10) if heston_opts else 10,
                }
                hest_res = calibrate_heston(S, K_arr, market_iv, r, q, T0,
                                            bounds=heston_opts_local['bounds'],
                                            popsize=heston_opts_local['popsize'],
                                            maxiter=heston_opts_local['maxiter'],
                                            Ncos=heston_opts_local['Ncos'],
                                            L=heston_opts_local['L'],
                                            verbose=False)
                heston_info = hest_res
                iv_heston_at_K = hest_res['model_iv']
                Ks_grid = np.exp(k_grid) * S
                iv_heston_grid, _ = heston_iv_surface_from_params(S, Ks_grid, r, q, T0, hest_res['params'], N=heston_opts_local['Ncos'], L=heston_opts_local['L'])
            except Exception:
                heston_info = None

        # hv_current / rv_used
        hv_row = vol_hist_df[(vol_hist_df['date'] == sub['date'].iloc[0]) & (vol_hist_df['act_symbol'] == sub['act_symbol'].iloc[0])]
        hv_current = None
        if not hv_row.empty and 'hv_current' in hv_row.columns:
            hv_current = float(hv_row['hv_current'].iloc[0]) if not pd.isna(hv_row['hv_current'].iloc[0]) else None
        days_horizon = max(1, int(round(T0 * 252)))
        rv_forward = realized_vol_forward(close_series, sub['date'].iloc[0], days_horizon)
        rv_backward = realized_vol_backward(close_series, sub['date'].iloc[0], days_horizon)
        rv_used = hv_current if hv_current is not None else (rv_forward if not np.isnan(rv_forward) else rv_backward)

        # store expiry info for later calendar enforcement
        expiry_infos.append({
            'expiry': expiry,
            'T': T0,
            'S': S,
            'k_arr': k_arr,
            'K_arr': K_arr,
            'market_iv': market_iv,
            'vega_arr': vega_arr,
            'weights': weights,
            'k_grid': k_grid,
            'iv_spline_grid': iv_spline_grid,
            'iv_svi_grid': iv_svi_grid,
            'iv_sabr_grid': iv_sabr_grid,
            'iv_heston_grid': iv_heston_grid,
            'iv_spline_at_K': iv_spline_at_K,
            'iv_svi_at_K': iv_svi_at_K,
            'iv_sabr_at_K': iv_sabr_at_K,
            'iv_heston_at_K': iv_heston_at_K,
            'svi_params': svi_params,
            'sabr_params': sabr_params,
            'heston_info': heston_info,
            'hv_current': rv_used,
            'sub_df': sub  # keep annotated sub frame for flags / audit
        })

    # if no expiry info, return None
    if len(expiry_infos) == 0:
        return None

    # ----- Calendar arbitrage enforcement across expiries (if more than 1 expiry) -----
    # Build common k grid covering min(k_min) .. max(k_max) across expiries
    k_min = min(info['k_grid'].min() for info in expiry_infos)
    k_max = max(info['k_grid'].max() for info in expiry_infos)
    common_k_grid = np.linspace(k_min, k_max, grid_points)

    # helper to resample iv_grid -> total variance on common grid
    def resample_w_on_common(iv_grid, k_grid, T):
        if iv_grid is None:
            return None
        # interp iv onto common k grid (use nearest extrapolation)
        iv_common = np.interp(common_k_grid, k_grid, iv_grid, left=iv_grid[0], right=iv_grid[-1])
        w_common = (iv_common ** 2) * T
        return w_common

    # For each model build w_matrix (rows = expiries sorted by T ascending)
    expiry_infos_sorted = sorted(expiry_infos, key=lambda x: x['T'])
    T_array = np.array([info['T'] for info in expiry_infos_sorted])
    n_exp = len(expiry_infos_sorted)
    n_k = len(common_k_grid)

    # function to build, enforce and convert back to iv grids per expiry
    def enforce_calendar_for_model(model_key):
        # collect resampled w rows
        w_rows = []
        have = []
        for info in expiry_infos_sorted:
            ivg = info.get(model_key)
            if ivg is None:
                w_rows.append(None)
                have.append(False)
            else:
                w_rows.append(resample_w_on_common(ivg, info['k_grid'], info['T']))
                have.append(True)
        if not any(have):
            return None  # model not present
        # build matrix for rows that exist; for missing rows we fill with nearest available (conservative)
        W = np.vstack([row if row is not None else np.nan * np.ones(n_k) for row in w_rows])
        # fill NaNs by nearest row (forward/backward fill)
        for j in range(n_k):
            col = W[:, j]
            # forward/back fill
            mask = np.isnan(col)
            if np.any(mask):
                # replace leading NaNs with first non-NaN
                if np.all(mask):
                    # if entire column NaN (shouldn't happen), set small value
                    W[:, j] = 1e-12
                else:
                    # forward fill
                    idxs = np.where(~mask)[0]
                    first = idxs.min(); last = idxs.max()
                    col[:first] = col[first]
                    col[last+1:] = col[last]
                    # linear interp for interior NaNs
                    nans = np.where(mask)[0]
                    notnans = np.where(~mask)[0]
                    W[:, j] = np.interp(np.arange(len(col)), notnans, col[notnans])
        # call user's enforce_calendar_arbitrage (assumes rows sorted by increasing T)
        W_adj, cal_changed = enforce_calendar_arbitrage(common_k_grid, W)
        # convert back to iv rows per expiry: iv = sqrt(w/T)
        iv_rows_adj = []
        for i_row in range(W_adj.shape[0]):
            T_i = expiry_infos_sorted[i_row]['T']
            iv_rows_adj.append(np.sqrt(np.maximum(W_adj[i_row, :] / T_i, 1e-12)))
        # map iv_rows_adj back to each expiry's original k_grid (interpolate)
        iv_grids_per_expiry = []
        for idx_info, info in enumerate(expiry_infos_sorted):
            iv_common = iv_rows_adj[idx_info]
            # interpolate iv_common (common_k_grid) back to info['k_grid']
            iv_on_orig = np.interp(info['k_grid'], common_k_grid, iv_common, left=iv_common[0], right=iv_common[-1])
            iv_grids_per_expiry.append(iv_on_orig)
        # return list in same order as expiry_infos_sorted
        return iv_grids_per_expiry

    # enforce calendar for each model
    spline_iv_grids_adj = enforce_calendar_for_model('iv_spline_grid')
    svi_iv_grids_adj = enforce_calendar_for_model('iv_svi_grid')
    sabr_iv_grids_adj = enforce_calendar_for_model('iv_sabr_grid')
    heston_iv_grids_adj = enforce_calendar_for_model('iv_heston_grid')

    # ----- PASS 2: recompute metrics & repairs using calendar-adjusted grids -----
    out_results = []
    for idx, info in enumerate(expiry_infos_sorted):
        expiry = info['expiry']; T0 = info['T']; S = info['S']
        k_arr = info['k_arr']; K_arr = info['K_arr']; market_iv = info['market_iv']; vega_arr = info['vega_arr']
        weights = info['weights']; sub = info['sub_df']; rv_used = info['hv_current']

        # pick adjusted iv grids if available, otherwise fall back to original
        # note: iv_*_grid are defined on info['k_grid']
        k_grid_orig = info['k_grid']
        iv_spline_grid = (spline_iv_grids_adj[idx] if spline_iv_grids_adj is not None else info['iv_spline_grid'])
        iv_svi_grid = (svi_iv_grids_adj[idx] if svi_iv_grids_adj is not None else info['iv_svi_grid'])
        iv_sabr_grid = (sabr_iv_grids_adj[idx] if sabr_iv_grids_adj is not None else info['iv_sabr_grid'])
        iv_heston_grid = (heston_iv_grids_adj[idx] if heston_iv_grids_adj is not None else info['iv_heston_grid'])

        # recompute w grids and repair convexity + vertical arbitrage per model
        # Spline
        w_spline = (iv_spline_grid ** 2) * T0
        bad_idx_spline, _ = check_butterfly_arbitrage(k_grid_orig, w_spline)
        w_spline_fixed, spline_conv_changed = repair_convexity_local(k_grid_orig, w_spline)
        iv_spline_fixed = np.sqrt(np.maximum(w_spline_fixed / T0, 1e-12))
        K_grid = np.exp(k_grid_orig) * S
        iv_spline_vfixed, price_spline_fixed, flags_spline_vert = enforce_vertical_arbitrage_on_iv_grid(iv_spline_fixed, K_grid, S, T0, r, option_type='C')

        # SVI
        if iv_svi_grid is not None:
            w_svi = (iv_svi_grid ** 2) * T0
            bad_idx_svi, _ = check_butterfly_arbitrage(k_grid_orig, w_svi)
            w_svi_fixed, svi_conv_changed = repair_convexity_local(k_grid_orig, w_svi)
            iv_svi_fixed = np.sqrt(np.maximum(w_svi_fixed / T0, 1e-12))
            iv_svi_vfixed, price_svi_fixed, flags_svi_vert = enforce_vertical_arbitrage_on_iv_grid(iv_svi_fixed, K_grid, S, T0, r, option_type='C')
        else:
            bad_idx_svi = []; svi_conv_changed = False
            iv_svi_fixed = None; iv_svi_vfixed = None; flags_svi_vert = {'vertical_violations':0,'vertical_fixed':False}

        # SABR
        if iv_sabr_grid is not None:
            w_sabr = (iv_sabr_grid ** 2) * T0
            bad_idx_sabr, _ = check_butterfly_arbitrage(k_grid_orig, w_sabr)
            w_sabr_fixed, sabr_conv_changed = repair_convexity_local(k_grid_orig, w_sabr)
            iv_sabr_fixed = np.sqrt(np.maximum(w_sabr_fixed / T0, 1e-12))
            iv_sabr_vfixed, price_sabr_fixed, flags_sabr_vert = enforce_vertical_arbitrage_on_iv_grid(iv_sabr_fixed, K_grid, S, T0, r, option_type='C')
        else:
            bad_idx_sabr = []; sabr_conv_changed = False
            iv_sabr_fixed = None; iv_sabr_vfixed = None; flags_sabr_vert = {'vertical_violations':0,'vertical_fixed':False}

        # Heston
        if iv_heston_grid is not None:
            w_heston = (iv_heston_grid ** 2) * T0
            bad_idx_heston, _ = check_butterfly_arbitrage(k_grid_orig, w_heston)
            w_heston_fixed, heston_conv_changed = repair_convexity_local(k_grid_orig, w_heston)
            iv_heston_fixed = np.sqrt(np.maximum(w_heston_fixed / T0, 1e-12))
            iv_heston_vfixed, price_heston_fixed, flags_heston_vert = enforce_vertical_arbitrage_on_iv_grid(iv_heston_fixed, K_grid, S, T0, r, option_type='C')
        else:
            bad_idx_heston = []; heston_conv_changed = False
            iv_heston_fixed = None; iv_heston_vfixed = None; flags_heston_vert = {'vertical_violations':0,'vertical_fixed':False}

        # interpolate model IVs to market strikes K_arr to compute metrics
        # spline at K
        from scipy.interpolate import UnivariateSpline
        spline_for_interp = UnivariateSpline(k_grid_orig, (iv_spline_fixed**2)*T0, s=0)
        iv_spline_at_K_fixed = np.sqrt(np.maximum(spline_for_interp(np.log(K_arr / S)) / T0, 1e-12))

        iv_svi_at_K_fixed = None
        if iv_svi_fixed is not None:
            iv_svi_at_K_fixed = np.interp(np.log(K_arr / S), k_grid_orig, iv_svi_fixed)

        iv_sabr_at_K_fixed = None
        if iv_sabr_fixed is not None:
            iv_sabr_at_K_fixed = np.interp(np.log(K_arr / S), k_grid_orig, iv_sabr_fixed)

        iv_heston_at_K_fixed = None
        if iv_heston_fixed is not None:
            iv_heston_at_K_fixed = np.interp(np.log(K_arr / S), k_grid_orig, iv_heston_fixed)

        metrics = {
            'rmse_spline': rmse(iv_spline_at_K_fixed, market_iv),
            'rmse_spline_vega': rmse_vega(iv_spline_at_K_fixed, market_iv, vega_arr),
            'rmse_svi': rmse(iv_svi_at_K_fixed, market_iv) if iv_svi_at_K_fixed is not None else np.nan,
            'rmse_svi_vega': rmse_vega(iv_svi_at_K_fixed, market_iv, vega_arr) if iv_svi_at_K_fixed is not None else np.nan,
            'rmse_sabr': rmse(iv_sabr_at_K_fixed, market_iv) if iv_sabr_at_K_fixed is not None else np.nan,
            'rmse_sabr_vega': rmse_vega(iv_sabr_at_K_fixed, market_iv, vega_arr) if iv_sabr_at_K_fixed is not None else np.nan,
            'rmse_heston': rmse(iv_heston_at_K_fixed, market_iv) if iv_heston_at_K_fixed is not None else np.nan
        }

        # ATM and vega-weighted IVs
        idx_atm = np.argmin(np.abs(K_arr - S))
        atm_iv = market_iv[idx_atm]
        vega_weighted_iv = np.sum(market_iv * vega_arr) / (np.sum(vega_arr) + 1e-12)
        iv_hv_diff_atm = (atm_iv - rv_used) if rv_used is not None else np.nan
        iv_hv_diff_vega = (vega_weighted_iv - rv_used) if rv_used is not None else np.nan

        # prepare outputs
        out = {
            'date': snapshot_date,
            'act_symbol': info['sub_df']['act_symbol'].iloc[0],
            'expiry': expiry,
            'spots_est': S,
            'T': T0,
            'days_horizon': max(1, int(round(T0 * 252))),
            'hv_current': rv_used,
            'atm_iv': atm_iv,
            'vega_weighted_iv': vega_weighted_iv,
            'svi_params': info['svi_params'].tolist() if info['svi_params'] is not None else None,
            'sabr_params': info['sabr_params'].tolist() if info['sabr_params'] is not None else None,
            'metrics': metrics,
            'iv_hv_diff_atm': iv_hv_diff_atm,
            'iv_hv_diff_vega': iv_hv_diff_vega,
            'k_grid': k_grid_orig,
            'K_grid': K_grid,
            'iv_spline_grid': iv_spline_fixed,
            'iv_svi_grid': iv_svi_fixed,
            'iv_sabr_grid': iv_sabr_fixed,
            'iv_heston_grid': iv_heston_fixed,
            'iv_spline_at_K': iv_spline_at_K_fixed,
            'iv_svi_at_K': iv_svi_at_K_fixed,
            'iv_sabr_at_K': iv_sabr_at_K_fixed,
            'iv_heston_at_K': iv_heston_at_K_fixed,
            'market_K': K_arr,
            'market_iv': market_iv,
            'sub_df': sub
        }

        # flat row
        flat = {
            'date': snapshot_date,
            'act_symbol': info['sub_df']['act_symbol'].iloc[0],
            'expiry': expiry,
            'spot': S,
            'T': T0,
            'days_horizon': max(1, int(round(T0 * 252))),
            'hv_current': rv_used,
            'atm_iv': atm_iv,
            'vega_weighted_iv': vega_weighted_iv,
            'iv_hv_diff_atm': iv_hv_diff_atm,
            'iv_hv_diff_vega': iv_hv_diff_vega,
            # SVI params
            'svi_a': info['svi_params'][0] if info['svi_params'] is not None else np.nan,
            'svi_b': info['svi_params'][1] if info['svi_params'] is not None else np.nan,
            'svi_rho': info['svi_params'][2] if info['svi_params'] is not None else np.nan,
            'svi_m': info['svi_params'][3] if info['svi_params'] is not None else np.nan,
            'svi_sigma': info['svi_params'][4] if info['svi_params'] is not None else np.nan,
            # SABR params
            'sabr_alpha': info['sabr_params'][0] if info['sabr_params'] is not None else np.nan,
            'sabr_rho': info['sabr_params'][1] if info['sabr_params'] is not None else np.nan,
            'sabr_nu': info['sabr_params'][2] if info['sabr_params'] is not None else np.nan,
            # Heston params (if any)
            'heston_kappa': info['heston_info']['params'][0] if info['heston_info'] is not None else np.nan,
            'heston_theta': info['heston_info']['params'][1] if info['heston_info'] is not None else np.nan,
            'heston_sigma': info['heston_info']['params'][2] if info['heston_info'] is not None else np.nan,
            'heston_rho': info['heston_info']['params'][3] if info['heston_info'] is not None else np.nan,
            'heston_v0': info['heston_info']['params'][4] if info['heston_info'] is not None else np.nan,
            # RMSEs
            'rmse_spline': metrics['rmse_spline'],
            'rmse_spline_vega': metrics['rmse_spline_vega'],
            'rmse_svi': metrics['rmse_svi'],
            'rmse_svi_vega': metrics['rmse_svi_vega'],
            'rmse_sabr': metrics['rmse_sabr'],
            'rmse_sabr_vega': metrics['rmse_sabr_vega'],
            'rmse_heston': metrics['rmse_heston'],
            # arbitration flags
            'butterfly_violations_spline': len(bad_idx_spline),
            'butterfly_convex_changed_spline': spline_conv_changed,
            'vertical_violations_spline': flags_spline_vert.get('vertical_violations', 0),
            'vertical_fixed_spline': flags_spline_vert.get('vertical_fixed', False),
            'butterfly_violations_svi': len(bad_idx_svi) if bad_idx_svi is not None else 0,
            'butterfly_convex_changed_svi': svi_conv_changed if 'svi_conv_changed' in locals() else False,
            'vertical_violations_svi': flags_svi_vert.get('vertical_violations', 0),
            'vertical_fixed_svi': flags_svi_vert.get('vertical_fixed', False),
            'butterfly_violations_sabr': len(bad_idx_sabr) if bad_idx_sabr is not None else 0,
            'butterfly_convex_changed_sabr': sabr_conv_changed if 'sabr_conv_changed' in locals() else False,
            'vertical_violations_sabr': flags_sabr_vert.get('vertical_violations', 0),
            'vertical_fixed_sabr': flags_sabr_vert.get('vertical_fixed', False),
            'butterfly_violations_heston': len(bad_idx_heston) if bad_idx_heston is not None else 0,
            'butterfly_convex_changed_heston': heston_conv_changed if 'heston_conv_changed' in locals() else False,
            'vertical_violations_heston': flags_heston_vert.get('vertical_violations', 0),
            'vertical_fixed_heston': flags_heston_vert.get('vertical_fixed', False),
            # counts of iv_source types for audit
            'n_market_iv_rows': int((info['sub_df']['iv_source'] == 'market_mid').sum()),
            'n_parity_iv_rows': int((info['sub_df']['iv_source'] == 'parity_from_call_market').sum()),
            'n_dataset_iv_rows': int((info['sub_df']['iv_source'] == 'dataset').sum())
        }

        out_results.append(flat)

    # combine and return
    if len(out_results) == 0:
        return None
    return pd.DataFrame(out_results)