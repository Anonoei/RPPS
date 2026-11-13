"""LTE turbo code (3GPP TS 36.212 5.1.3.2), rate matching (5.1.4.1) and CRC24A (5.1.1), from the rpps definitions

    python projects/lte/lte_turbo.py --selftest

block/turbo/lte: two 8-state RSC codes (g0 = 1 + D^2 + D^3, g1 = 1 + D + D^3), the QPP interleaver of Table 5.1.3-3 (all 188 sizes), trellis
termination; three streams d0 (systematic), d1, d2 (parities) of K + 4 bits with the tail bits of 5.1.3.2.2; max-log-MAP decoding that uses the tails.
block/ratematch/lte_turbo: sub-block interleaving, circular buffer, redundancy versions.
"""
import numpy as np

import rpps as rp
import sys

_CRC24A = rp.link.load("crc", "crc24_lte_a")._enc

def crc24a(bits):
    """gCRC24A (36.212 5.1.1), zero start, MSB first: returns 24 bits"""
    return [int(b) for b in _CRC24A.to_bits(_CRC24A.compute_bits(bits))]

def _turbo(K):
    return rp.code.load("block", "turbo/lte", K=K)._enc


def _rm(K, E, rv):
    return rp.code.load("block", "ratematch/lte_turbo", K=K, E=E, rv=rv)._enc


def turbo_encode(c):
    """d0, d1, d2 of length K + 4 (tail bits included)"""
    c = np.asarray(c, np.uint8)
    return _turbo(len(c)).encode(c).reshape(3, len(c) + 4).astype(np.uint8)


def rate_match(d, E, rv):
    K = d.shape[1] - 4
    return _rm(K, E, rv).encode(np.asarray(d).reshape(-1)).astype(np.uint8)


def dematch(llr, K, rv, d=None):
    """Add E soft bits (positive = 0) onto the (3, K+4) streams; pass d to accumulate several transmissions (HARQ combining)"""
    acc = _rm(K, len(llr), rv).decode(np.asarray(llr, float)).reshape(3, K + 4)
    return acc if d is None else d + acc


def turbo_decode(d, K, iters=8, scale=0.75):
    """d: (3, K+4) soft streams from dematch. Returns hard bits of length K and the final LLRs."""
    t = _turbo(K)
    t.iters, t.scale = iters, scale
    bits = t.decode(np.asarray(d, float).reshape(-1))
    return bits.astype(np.uint8), t.llr

def selftest():
    rng = np.random.default_rng(3)
    ok = n = 0
    for K, E, rv, sigma in ((248, 1000, 0, 1.0), (248, 700, 2, 0.8), (248, 1200, 3, 1.1), (200, 900, 1, 0.9)):
        for t in range(6):
            bits = rng.integers(0, 2, K - 24).astype(np.uint8)
            c = np.concatenate((bits, np.array(crc24a(bits), np.uint8)))
            tx = rate_match(turbo_encode(c), E, rv)
            llr = (1 - 2.0*tx) + rng.normal(0, sigma, E)
            dec, _ = turbo_decode(dematch(llr, K, rv), K)
            n += 1; ok += np.array_equal(dec, c)
    print(f"selftest: {ok}/{n} turbo blocks recovered (K 200-248, rate 0.2-0.35, noise sigma 0.8-1.1)")
    # CRC property: a message plus its CRC has zero remainder
    m = rng.integers(0, 2, 100).astype(np.uint8)
    full = np.concatenate((m, np.array(crc24a(m), np.uint8)))
    print("crc24a zero remainder:", crc24a(full) == [0]*24)
    return ok == n

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
