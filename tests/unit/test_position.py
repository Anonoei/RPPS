"""Absolute stream positions (ARCHITECTURE.md, Sync methods rule 1, and Migration step 2)"""
from types import SimpleNamespace

import numpy as np
import pytest

import rpps as rp

class Probe(rp.Stage):
    """Records the input position of every block it sees, and passes the samples through"""
    def __init__(self):
        self.seen = []
    def new_state(self, meta):
        return SimpleNamespace()
    def step(self, samples, st, meta):
        self.seen.append((meta.pos, len(samples)))
        return samples

def blocks(x, cuts, *stages, **rates):
    stream, outs = None, []
    i = 0
    for b in cuts:
        d = rp.Data(x[i:i + b], stream=stream, **rates)
        stream = d.stream
        for st in stages:
            d = d + st
        outs.append(d)
        i += b
    return outs

def test_source_blocks_carry_their_position():
    x = np.zeros(100, np.complex64)
    out = blocks(x, [10, 0, 37, 53], Fs=1e3)
    assert [o.meta.pos for o in out] == [0, 10, 10, 47]
    assert out[-1].stream.pos == 100

def test_a_new_stream_starts_at_zero():
    assert rp.Data(np.zeros(5, np.complex64)).meta.pos == 0

@pytest.mark.parametrize("cuts", [[100], [33, 33, 34], [1]*7 + [93], [0, 50, 0, 50]])
def test_stage_outputs_are_positioned_independently_of_the_cut(cuts):
    x = np.arange(100).astype(np.complex64)
    out = blocks(x, cuts, rp.signal.sample.down.Decimate(4), Fs=8.0, sps=8)
    pos = [o.meta.pos for o in out]
    lens = [len(o) for o in out]
    assert pos == list(np.concatenate(([0], np.cumsum(lens)[:-1]))) # each block starts where the last ended
    for o in out: # and the position is the decimated index of the input it came from
        if len(o):
            assert o.data[0].real == 4*o.meta.pos

def test_a_stage_sees_its_inputs_position():
    probe = Probe()
    blocks(np.zeros(60, np.complex64), [20, 5, 35], probe, Fs=1.0)
    assert probe.seen == [(0, 20), (20, 5), (25, 35)]

def test_positions_after_a_rate_change():
    probe = Probe()
    blocks(np.zeros(40, np.complex64), [13, 27], rp.signal.sample.up.Zeros(4), probe, Fs=1.0, sps=1)
    assert probe.seen == [(0, 52), (52, 108)]

def test_time_of_items():
    d = rp.Data(np.zeros(10, np.complex64), Fs=1e3, t0=100.0)
    d2 = rp.Data(np.zeros(10, np.complex64), Fs=1e3, stream=d.stream)
    assert d.meta.time_of(0) == pytest.approx(100.0) and d2.meta.time_of(3) == pytest.approx(100.013)
    dec = d2 + rp.signal.sample.down.Decimate(2)
    assert dec.meta.pos == 5 # the stage first ran on the second block: its outputs start at that block, at half the rate
    assert dec.meta.time_of(0) == pytest.approx(100.010)
    assert rp.Data(np.zeros(4, np.complex64)).meta.time_of() is None

def test_a_stage_added_mid_stream_starts_at_its_input():
    probe = Probe()
    d = rp.Data(np.zeros(30, np.complex64), Fs=1.0)
    d2 = rp.Data(np.zeros(30, np.complex64), stream=d.stream)
    out = d2 + rp.signal.sample.down.Decimate(3) + probe
    assert out.meta.pos == 10 and probe.seen == [(10, 10)]

def test_restart_starts_positions_over():
    d = rp.Data(np.zeros(10, np.complex64))
    d + rp.signal.sample.down.Decimate(2)
    d.stream.restart()
    assert d.stream.pos == 0 and d.stream.out_pos == {}
