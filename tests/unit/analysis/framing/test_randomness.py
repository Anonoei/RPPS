"""rpps.analysis.framing.randomness: NIST SP 800-22 tests (each against the standard's worked examples) and the verdict"""
import bz2

import numpy as np
import pytest

from rpps.analysis.framing import payload as PL
from rpps.analysis.framing import randomness as R
from rpps.math.prob import igamc

def bits(s):
    return np.array([int(c) for c in s], np.uint8)

# NIST SP 800-22 rev. 1a, section 2: the 100-bit example sequence of sections 2.1.8 - 2.13.8
EX100 = bits("1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000")

@pytest.mark.parametrize("got, want", [
    (lambda: R.frequency(bits("1011010101")), 0.527089),            # 2.1.4
    (lambda: R.frequency(EX100), 0.109599),                          # 2.1.8
    (lambda: R.block_frequency(bits("0110011010"), 3), 0.801252),   # 2.2.4
    (lambda: R.block_frequency(EX100, 10), 0.706438),               # 2.2.8
    (lambda: R.runs(bits("1001101011")), 0.147232),                 # 2.3.4
    (lambda: R.runs(EX100), 0.500798),                               # 2.3.8
    (lambda: R.serial(bits("0011011101"), 3)[0], 0.808792),         # 2.11.6
    (lambda: R.serial(bits("0011011101"), 3)[1], 0.670320),         # 2.11.6
    (lambda: R.approximate_entropy(bits("0100110101"), 3), 0.261961),  # 2.12.4
    (lambda: R.approximate_entropy(EX100, 2), 0.235301),            # 2.12.8
    (lambda: R.cusum(bits("1011010111")), 0.4116588),               # 2.13.4
    (lambda: R.cusum(EX100), 0.219194),                              # 2.13.8 forward
    (lambda: R.cusum(EX100, backward=True), 0.114866),              # 2.13.8 backward
])
def test_the_standards_worked_examples(got, want):
    assert got() == pytest.approx(want, abs=2e-6)

def test_igamc_is_the_chi_square_tail():
    assert igamc(1.5, 0.5) == pytest.approx(0.801252, abs=1e-6) and igamc(2, 0.8) == pytest.approx(0.808792, abs=1e-6)
    assert igamc(1, 3.0) == pytest.approx(np.exp(-3.0), rel=1e-12) # Q(1, x) = e^-x
    assert igamc(127.5, 127.5) == pytest.approx(0.4882, abs=1e-4) and igamc(5, 0.0) == 1.0

def test_random_bytes_pass_and_compressed_ones_do_not():
    r = R.test(np.random.default_rng(3).integers(0, 2, 400_000).astype(np.uint8))
    assert r.passed and r.verdict == "random" and r.entropy > 7.99
    rng = np.random.default_rng(5)
    words = ["".join(chr(97 + c) for c in rng.integers(0, 26, rng.integers(2, 9))) for _ in range(4000)]
    text = " ".join(words[i] for i in rng.zipf(1.2, 200000) % 4000).encode() # Text-like: a large vocabulary, Zipf
    c = R.test(np.unpackbits(np.frombuffer(bz2.compress(text), np.uint8))[:400_000])
    assert not c.passed and c.verdict == "high entropy" # near-uniform bytes, but not a cipher's output
    assert R.test(np.zeros(500, np.uint8)) is None # too short to say

def test_a_random_looking_payload_says_whether_it_passes():
    v = PL.verdict(np.random.default_rng(4).integers(0, 2, 200_000).astype(np.uint8))
    assert v.kind == "random-looking" and v.randomness.passed and "NIST SP 800-22" in v.why
