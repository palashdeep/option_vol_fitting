import numpy as np

from core.arbitrage.calendar import enforce_calendar_arbitrage

def test_calendar_monotonic_totoal_varaince():
    W = np.array([
        [0.020, 0.030, 0.040],
        [0.018, 0.028, 0.038],
        [0.025, 0.035, 0.045],
    ])  # T x k

    k_grid = np.array([-0.1, 0.0, 0.1])

    W_fixed, changed = enforce_calendar_arbitrage(k_grid, W)

    assert np.all(np.diff(W_fixed) >= -1e-12)
    assert changed is True