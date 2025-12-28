import numpy as np
# ---------------------------
# Realized vol (forward if available, else backward)
#    using daily close series. Annualized using 252 trading days.
# ---------------------------
def realized_vol_forward(close_series, start_date, days):
    # close_series: pandas Series indexed by date
    # we want returns from start_date+1 to start_date+days (forward realized)
    start_idx = close_series.index.get_loc(start_date, method='nearest')
    end_idx = start_idx + days
    if end_idx >= len(close_series):
        return np.nan  # not enough forward data
    prices = close_series.iloc[start_idx: end_idx + 1].values
    rets = np.diff(np.log(prices))
    if len(rets) <= 1:
        return np.nan
    rv = np.sqrt(np.sum(rets**2) * (252.0 / len(rets)))
    return rv

def realized_vol_backward(close_series, end_date, days):
    # compute realized vol using previous 'days' prior to end_date (exclusive)
    end_idx = close_series.index.get_loc(end_date, method='nearest')
    start_idx = end_idx - days
    if start_idx < 0:
        return np.nan
    prices = close_series.iloc[start_idx: end_idx + 1].values
    rets = np.diff(np.log(prices))
    if len(rets) <= 1:
        return np.nan
    rv = np.sqrt(np.sum(rets**2) * (252.0 / len(rets)))
    return rv