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
    return rp.sync.AGC(1.0) + rp.sync.time.MM(alpha=0.1) + rp.sync.Costas.from_bandwidth(0.02, order=4)

def blocks_of(path):
    return rp.source.File("cf32", path, Fs=8e3, sps=8)(1000)

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
    for i, _ in enumerate(rp.source.source.Blocks(source()).pipelined(rp.sync.AGC() + rp.sync.AGC())):
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
    src = rp.source.File("cf32", path, Fs=8e3, sps=2.0)
    list(src(500).pipelined(rp.signal.sample.down.Decimate(2) + Calibrate() + rec, depth=8))
    assert len(rec.sps) == 40 and all(s == pytest.approx(4.0) for s in rec.sps)
    assert src.stream.src_sps == pytest.approx(8.0)

def test_a_dropping_source_counts_what_it_discards():
    def source():
        stream = rp.Data(np.zeros(1, np.complex64), Fs=1e3).stream
        for _ in range(200):
            yield rp.Data(np.ones(64, np.complex64), Fs=1e3, stream=stream)
    got, stream, dec = 0, None, rp.signal.sample.down.Decimate(2)
    for b in rp.source.source.Blocks(source()).threaded(depth=1, drop=True) + dec:
        stream = b.stream
        assert b.meta.pos == stream.in_pos[dec]//2 - 32 # after a gap too: the decimator counts the dropped blocks
        time.sleep(0.002) # a slow reader
        got += 1
    assert 64*got + stream.dropped == 64*200 and stream.dropped > 0 # in source samples

@pytest.mark.filterwarnings("ignore:Simulated") # its watchdog, when this machine is slow to read
def test_a_radio_overrun_moves_the_positions_on():
    """Samples a radio lost are counted in meta.pos at the source and through every stage, and in stream.dropped"""
    pysdrlib = pytest.importorskip("pysdrlib")
    sdr = pysdrlib.device("sim", realtime=False)
    src = rp.source.Device(sdr, cf=100e6, Fs=1e6)
    dec = rp.signal.sample.down.Decimate(4)
    last, jumps, after = None, [], None
    for i, b in enumerate(src(4096) + dec):
        if i == 2:
            sdr.drop(10000)
        if last is not None and b.meta.pos != last:
            jumps.append(b.meta.pos - last)
        last = b.meta.pos + len(b)
        # The decimator's positions follow the source's (each gap rounded to its rate once)
        assert abs(b.meta.pos - (src.stream.pos - 4096)/4) <= len(jumps)
        if after is not None or i > 400: # One block past the drop: a gap inside a block shows at the next one
            break
        if b.stream.pos - 4096*(i + 1) >= 10000: # (`dropped` can also count what was lost before the first block)
            after = i
    # A loaded machine can overrun the sim on its own (its watchdog): every gap is counted, however many there are
    lost = src.stream.pos - 4096*(i + 1) # what the source's positions moved on by
    assert src.stream.dropped >= lost >= 10000 and min(jumps) > 0
    assert abs(4*sum(jumps) - lost) <= 4*len(jumps)
    assert src.stream.t0 is not None and b.meta.time_of(0) == pytest.approx(src.stream.t0 + (src.stream.pos - 4096)/1e6)

def test_load_reports_each_stage_against_real_time(capture):
    _, path = capture
    src = rp.source.File("cf32", path, Fs=8e3, sps=8)
    agc, mm = rp.sync.AGC(), rp.sync.time.MM()
    (src(1000) + (agc + mm)).collect()
    assert src.stream.air == pytest.approx(len(capture[0])/8e3)
    load = src.stream.load()
    assert set(load) == {agc, mm} and all(0 < v < 1 for v in load.values())

def test_stages_behind_acquire_use_the_detection_their_block_went_through_with(tmp_path):
    """Pipelined, Acquire runs ahead and refines for later blocks: the stages after it must use each block's own detection
    (meta.det) and sps, so the output is the iterated one exactly, whatever the threads' timing"""
    from test_lock import psk
    x = (psk(80000, 4.0, tau=1.3, cfo=2e-3, esn0_db=18)*0.3).astype(np.complex64)
    path = tmp_path / "q.iq"
    x.tofile(path)
    make = lambda: rp.sync.Acquire() + rp.sync.time.Mat() + rp.sync.phase.FFCarrier()
    ref = np.concatenate([o.data for o in rp.source.File("cf32", path, Fs=1e6, sps=4)(2048) + make()])
    for _ in range(4):
        got = np.concatenate([o.data for o in rp.source.File("cf32", path, Fs=1e6, sps=4)(2048).pipelined(make())])
        assert np.array_equal(got, ref)
