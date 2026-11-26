"""The shift registers behind the feedthrough scramblers (rpps.sequence.lfsr)"""
import numpy as np
import pytest

import rpps as rp
from rpps.sequence.lfsr import fibonacci, galois_left

REGISTERS = [fibonacci, galois_left]

def register(cls):
    return cls(np.array([1, 0, 0, 0], bool), np.array([4, 3])) # x^4 + x^3 + 1: primitive

def output(reg, n):
    return [int(reg.get_bit()) for _ in range(n)]

@pytest.mark.parametrize("cls", REGISTERS)
def test_a_primitive_polynomial_gives_a_maximal_length_sequence(cls):
    s = output(register(cls), 45)
    assert s[:15] == s[15:30] == s[30:] and sum(s[:15]) == 8
    assert all(s[:15] != s[p:p + 15] for p in (1, 3, 5)) # no shorter period

@pytest.mark.parametrize("cls", REGISTERS)
def test_reset_returns_to_the_seed_and_copies_are_independent(cls):
    reg = register(cls)
    first = output(reg, 20)
    copy, fresh = reg.clone(), reg.fresh()
    assert output(fresh, 20) == first
    assert output(copy, 10) == output(reg, 10) # the clone goes on from where it was taken
    output(copy, 3)
    assert copy.lfsr.tolist() != reg.lfsr.tolist()
    reg.reset()
    assert output(reg, 20) == first

def feedthrough(register, poly=(7, 4)):
    return rp._config.load_obj("scramble", "t", {"base": "scramble", "type": "feedthrough", "register": register,
                                                 "poly": list(poly), "seed": [1, 0, 1, 1, 0, 0, 1]})

@pytest.mark.parametrize("register", ["fibonacci", "galois_left",
    pytest.param("galois_right", marks=pytest.mark.xfail(strict=True, reason="galois_right shifts its feedback out of the "
                 "register before it reaches the output: no delays, so the data is unscrambled after the first register length"))])
def test_every_register_form_scrambles_with_the_polynomials_delays(register, rng):
    s = feedthrough(register)
    data = rng.integers(0, 2, 400).astype(bool)
    y = np.asarray(s.encode(data)).astype(bool)
    assert s.delays() == [4, 7]
    assert np.mean(y[7:] != data[7:]) > 0.3
    assert np.array_equal(np.asarray(feedthrough(register).decode(y)).astype(bool), data)
