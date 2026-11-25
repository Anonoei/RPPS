"""The bit kernels (rpps.math.bits: distance, unstuff) give the Python reference's output (rpps._native off and on),
and what stands on them (HDLC deframing, FrameSync's word search) gives the same frames"""
import numpy as np
import pytest

import rpps as rp
from rpps import _accel
from rpps.math import bits

pytestmark = pytest.mark.skipif(not _accel.available(), reason="rpps._native is not built")

class Counting:
    def __init__(self, mod):
        self.mod, self.calls = mod, 0
    def __getattr__(self, name):
        self.calls += 1
        return getattr(self.mod, name)

def both(fn):
    saved = _accel.native
    try:
        _accel.native = None
        ref = fn()
        spy = Counting(saved)
        _accel.native = spy
        got = fn()
        assert spy.calls > 0, "the native kernel never ran"
    finally:
        _accel.native = saved
    return ref, got

@pytest.mark.parametrize("L", [1, 7, 8, 32, 63, 64])
def test_distance_at_every_position(L):
    rng = np.random.default_rng(L)
    b = rng.integers(0, 2, 5000).astype(np.uint8)
    w = rng.integers(0, 2, L).astype(np.uint8)
    b[100:100 + L] = w
    b[900:900 + L] = 1 - w
    ref, got = both(lambda: bits.distance(b, w))
    assert np.array_equal(ref, got) and len(got) == len(b) - L + 1
    assert got[100] == 0 and got[900] == L
    assert 100 in bits.find(b, w) and 900 in bits.find(b, 1 - w)

def test_distance_of_a_word_longer_than_the_bits_or_of_none():
    assert len(bits.distance(np.ones(5, np.uint8), np.ones(8, np.uint8))) == 0
    assert len(bits.distance(np.ones(5, np.uint8), np.ones(0, np.uint8))) == 0

def test_unstuff_spans_good_aborted_and_ragged():
    rng = np.random.default_rng(0)
    h = rp.link.load("frame/hdlc").encoder
    frames = [bytes(rng.integers(0, 256, n, dtype=np.uint8)) for n in (4, 31, 200)] + [b"\xff"*40]
    b = h.encode(frames).astype(np.uint8)
    starts = np.nonzero(bits.distance(b, h.flag_bits) == 0)[0]
    a, e = starts[:-1] + 8, starts[1:]
    real = np.nonzero(e - a > 16)[0] # (the frames; the rest are idle flags side by side)
    b2 = b.copy()
    b2[a[real[1]] + 20:a[real[1]] + 28] = 1 # (eight ones inside the second frame: an abort)
    a = np.append(a, a[real[0]]) # (the first frame cut 3 bits short: ragged)
    e = np.append(e, e[real[0]] - 3)
    for x in (b, b2):
        (s0, r0), (s1, r1) = both(lambda: bits.unstuff(x, a, e, 5))
        assert np.array_equal(s0, s1) and r0 == r1
        assert s1[-1] == 2 and (x is b or s1[real[1]] == 1)
    assert sum(len(r) > 0 for r in r1) == len(real) - 1 # (every frame but the aborted one gives its bytes)

@pytest.mark.parametrize("name", ["hdlc", "ax25"])
def test_hdlc_frames_the_same_either_way(name):
    rng = np.random.default_rng(1)
    h = rp.link.load(f"frame/{name}").encoder
    frames = [bytes(rng.integers(0, 256, int(rng.integers(4, 80)), dtype=np.uint8)) for _ in range(30)]
    b = h.encode(frames).astype(np.uint8)
    b[rng.integers(0, len(b), 6)] ^= 1 # (a few errors: bad FCSs, aborts)
    noise = rng.integers(0, 2, 20000).astype(np.uint8)
    x = np.concatenate((noise, b, noise))
    (f0, a0), (f1, a1) = both(lambda: h.frames_at(x))
    assert f0 == f1 and a0 == a1
    assert sum(ok for _, _, ok, _ in f1) >= 20
