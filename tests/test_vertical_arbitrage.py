import numpy as np
from core.arbitrage.vertical import enforce_vertical_arbitrage_on_iv_grid

def test_vertical_spread_monotonicity():
    
    S = 100.0
    T = 0.5
    r = 0.05

    strikes = np.array([105, 110, 115, 120])
    iv_grid = np.array([0.30, 0.15, 0.25, 0.20])

    iv_fixed, repaired_prices, flags = enforce_vertical_arbitrage_on_iv_grid(iv_grid, strikes, S, T, r, option_type='C')

    diffs = np.diff(repaired_prices)
    assert np.all(diffs <= 1e-12)
    assert flags["vertical_fixed"] is True
