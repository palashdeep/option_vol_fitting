import matplotlib.pyplot as plt
import numpy as np

def plot_iv_vol(df, symbol='', column_calc='impliedVol_calc', column_vol='impliedVol'):
    """Plot computed vs existing IVs"""
    plt.scatter(df[column_vol], df[column_calc], alpha=0.5)
    plt.plot([0,5],[0,5],'r--')
    plt.xlabel("Existing IV (from DB)")
    plt.ylabel("Computed IV")
    plt.title(f"Implied Vol Comparison - {symbol}")
    plt.show()

def plot_smile(df, expiry, symbol='', column_strike='strike', column_calc='impliedVol_calc', column_vol='impliedVol', column_rv='realizedVol'):
    """Plot diff vols for a given expiry"""
    sub = df[df["tenor"] == expiry]
    print(f"Plotting smile for {symbol} expiry {expiry}, {len(sub)} points")
    plt.figure(figsize=(8,5))
    plt.plot(sub[column_strike], sub[column_calc], "o-", label="Implied (calc)")
    plt.plot(sub[column_strike], sub[column_vol], "x-", label="IV (DB)")
    plt.axhline(sub[column_rv].mean(), color='r', linestyle='--', label="Realized 21d")
    plt.title(f"Vol Smile - {symbol}, {expiry}")
    plt.xlabel("Strike")
    plt.ylabel("Volatility")
    plt.legend()
    plt.show()

def plot_iv_error(df, symbol='', moneyness_col='moneyness', iv_err_col='iv_err'):
    """Plot IV error vs moneyness"""
    plt.scatter(df[moneyness_col], df[iv_err_col])
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Strike / Forward Moneyness")
    plt.ylabel("IV_calc - IV_data")
    plt.title("Implied Vol Error vs Moneyness")
    plt.show()

def plot_smiles_comparison(result):
    df = result['df']
    K = result['market_K']
    market_iv = result['market_iv']
    plt.figure(figsize=(9,5))
    plt.plot(K, market_iv, 'o', label='market IV (used)')
    if result['iv_spline_at_K'] is not None:
        plt.plot(result['K_grid'], result['iv_spline_grid'], '-', label='Spline (total var)')
    if result['iv_svi_grid'] is not None:
        plt.plot(result['K_grid'], result['iv_svi_grid'], '--', label='SVI')
    if result['iv_sabr_grid'] is not None:
        plt.plot(result['K_grid'], result['iv_sabr_grid'], ':', label='SABR (Hagan)')
    if result['iv_heston_grid'] is not None:
        plt.plot(result['K_grid'], result['iv_heston_grid'], '-.', label='Heston')
    # realized
    rv = result['rv_used']
    if not np.isnan(rv):
        plt.axhline(rv, color='k', linestyle='-.', label=f'Realized ({result["days_horizon"]}d) = {rv:.2%}')
    plt.xlabel('Strike')
    plt.ylabel('Implied vol')
    plt.title(f"Vol Smile — {result['date'].date()} exp {result['expiry'].date()}")
    plt.legend()
    plt.show()

def plot_residuals(result):
    K = result['market_K']
    market_iv = result['market_iv']
    plt.figure(figsize=(9,4))
    if result['iv_spline_at_K'] is not None:
        plt.scatter(K, result['iv_spline_at_K'] - market_iv, label='spline-residual')
    if result['iv_svi_at_K'] is not None:
        plt.scatter(K, result['iv_svi_at_K'] - market_iv, label='svi-residual')
    if result['iv_sabr_at_K'] is not None:
        plt.scatter(K, result['iv_sabr_at_K'] - market_iv, label='sabr-residual')
    if result['iv_heston_at_K'] is not None:
        plt.scatter(K, result['iv_heston_at_K'] - market_iv, label='heston-residual')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Strike')
    plt.ylabel('IV residual (model - market)')
    plt.legend()
    plt.show()
