import numpy as np

from core.arbitrage.vertical import enforce_vertical_arbitrage

def test_vertical_spread_monotonicity():
    strikes = np.array([90, 100, 110, 120])
    call_prices = np.array([15.0, 14.0, 14.5, 13.0])

    fixed_prices, flags = enforce_vertical_arbitrage(call_prices, strikes)

    diffs = np.diff(fixed_prices)
    assert np.all(diffs <= 1e-12)
    assert["vertical_fixed"] is True
