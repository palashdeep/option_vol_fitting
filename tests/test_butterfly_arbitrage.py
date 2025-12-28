import numpy as np

from core.arbitrage.butterfly import (
    check_butterfly_arbitrage,
    repair_convexity_local
)

def test_butterfly_convexity_repair():
    k = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    w = np.array([0.04, 0.035, 0.02, 0.035, 0.04])

    bad_idx, _ = check_butterfly_arbitrage(k, w)
    assert len(bad_idx) > 0

    w_fixed, changed = repair_convexity_local(k, w)

    bad_idx_fixed, _ = check_butterfly_arbitrage(k, w_fixed)

    assert len(bad_idx_fixed) == 0
    assert changed is True