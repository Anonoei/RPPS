"""Blocks.threaded / pipelined: the same output as one thread, errors and early exits handled, rates kept"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import rpps as rp
from rpps.base import Stage

def shaped(n_sym=4000, sps=8, seed=0):
    rng = np.random.default_rng(seed)
    sym = rng.choice([-1, 1], n_sym) + 1j*rng.choice([-1, 1], n_sym)
    sig = np.convolve(np.repeat(sym, sps), np.ones(sps)/sps, mode="same")
    sig = sig*np.exp(2j*np.pi*1e-4*np.arange(len(sig)))
    return (0.3*sig + 0.02*(rng.standard_normal(len(sig)) + 1j*rng.standard_normal(len(sig)))).astype(np.complex64)

@pytest.fixture
def capture(tmp_path):
    x = shaped()
    path = tmp_path / "a.iq"
    x.tofile(path)
    return x, path

def chain():
    return rp.sync.AGC(1.0, 0.01) + rp.sync.time.MM(alpha=0.1) + rp.sync.Costas.from_bandwidth(0.02, order=4)

def blocks_of(path):
    return rp.serial.File("cf64", path, Fs=8e3, sps=8)(1000)

@pytest.mark.parametrize("how", ["threaded source", "pipelined", "both ends"])
def test_threads_give_the_same_stream(capture, how):
    x, path = capture
    ref = (blocks_of(path) + chain()).collect()
    if how == "threaded source":
        got = (blocks_of(path).threaded() + chain()).collect()
    elif how == "pipelined":
        got = blocks_of(path).pipelined(chain()).collect()
    else:
        got = (blocks_of(path).threaded(depth=2) + chain()).threaded(depth=2).collect()
    assert len(got) == len(ref) > 3000
    assert np.array_equal(got, ref)

class Boom(Stage):
    def __init__(self, at):
        self.at = at
    def new_state(self, meta):
        return SimpleNamespace(n=0)
    def step(self, samples, st, meta):
        st.n += 1
        if st.n == self.at:
            raise RuntimeError("boom")
        return samples

def test_an_error_in_a_stage_thread_reaches_the_reader(capture):
    _, path = capture
    with pytest.raises(RuntimeError, match="boom"):
        blocks_of(path).pipelined(rp.sync.AGC() + Boom(3)).collect()
    # and every thread of the pipeline stopped (not left to the garbage collector)
    assert not [t for t in threading.enumerate() if t.name == "rpps-blocks"]

def test_leaving_the_loop_stops_the_threads_and_closes_the_source():
    closed = threading.Event()
    def source():
        stream = rp.Data(np.zeros(1, np.complex64), Fs=1e3).stream
        try:
            while True: # endless, like a radio
                yield rp.Data(np.ones(256, np.complex64), Fs=1e3, stream=stream)
        finally:
            closed.set()
    before = len([t for t in threading.enumerate() if t.name != "rpps-blocks"]) # threads of other tests may still be exiting
    for i, _ in enumerate(rp.serial.source.Blocks(source()).pipelined(rp.sync.AGC() + rp.sync.AGC())):
        if i == 20:
            break
    assert closed.wait(2.0)
    deadline = time.time() + 2.0
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.01)
    assert threading.active_count() <= before

class Calibrate(Stage):
    """Learns sps = 4 on the first block, slowly, so blocks pile up behind it"""
    def new_state(self, meta):
        return SimpleNamespace(done=False)
    def step(self, samples, st, meta):
        if not st.done:
            time.sleep(0.05)
            meta.calibrate_sps(4.0)
            st.done = True
        return samples

class Record(Stage):
    def __init__(self):
        self.sps = []
    def step(self, samples, st, meta):
        self.sps.append(meta.sps)
        return samples

def test_an_sps_learned_downstream_reaches_blocks_already_in_flight(tmp_path):
    path = tmp_path / "a.iq"
    np.ones(20000, np.complex64).tofile(path)
    rec = Record()
    src = rp.serial.File("cf64", path, Fs=8e3, sps=2.0)
    list(src(500).pipelined(rp.sample.down.Decimate(2) + Calibrate() + rec, depth=8))
    assert len(rec.sps) == 40 and all(s == pytest.approx(4.0) for s in rec.sps)
    assert src.stream.src_sps == pytest.approx(8.0)

def test_a_dropping_source_counts_what_it_discards():
    def source():
        stream = rp.Data(np.zeros(1, np.complex64), Fs=1e3).stream
        for _ in range(200):
            yield rp.Data(np.ones(64, np.complex64), Fs=1e3, stream=stream)
    got, stream = 0, None
    for b in rp.serial.source.Blocks(source()).threaded(depth=1, drop=True):
        stream = b.stream
        time.sleep(0.002) # a slow reader
        got += 1
    assert got + stream.dropped == 200 and stream.dropped > 0

def test_load_reports_each_stage_against_real_time(capture):
    _, path = capture
    src = rp.serial.File("cf64", path, Fs=8e3, sps=8)
    agc, mm = rp.sync.AGC(), rp.sync.time.MM()
    (src(1000) + (agc + mm)).collect()
    assert src.stream.air == pytest.approx(len(capture[0])/8e3)
    load = src.stream.load()
    assert set(load) == {agc, mm} and all(0 < v < 1 for v in load.values())
