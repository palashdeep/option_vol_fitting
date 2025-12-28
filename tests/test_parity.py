import math
import numpy as np

from core.arbitrage.black_scholes import put_from_call, call_from_put

def test_put_call_parity_identity():
    S = 100.0
    K = 105.0
    r = 0.01
    q = 0.02
    T = 0.5

    call_price = 4.25
    put_price = put_from_call(call_price, S, K, r, q, T)

    lhs = call_price - put_price
    rhs = S * math.exp(-q*T) - K * math.exp(-r*T)

    assert abs(lhs - rhs) < 1e-10

def test_call_reconstruction_round_trip():
    S = 100.0
    K = 965.0
    r = 0.01
    q = 0.0
    T = 1.0

    call_price = 7.8
    put_price = put_from_call(call_price, S, K, r, q, T)
    call_reconstructed = call_from_put(put_price, S, K, r, q, T)

    assert abs(call_price - call_reconstructed) < 1e-10