"""The load of a stream: each operator's time over the signal's (1.0 keeps up exactly), stages and pipes alike, own time
only, over a recent window, and as one number per thread (rpps.base.meta.Stream.load, realtime)"""
import threading
import time

import numpy as np
import pytest

import rpps as rp
from rpps.base.meta import Stream, timed

class Slow(rp.base.Pipe):
    """A pipe that takes `t` seconds to decode, then decodes through `inner`"""
    def __init__(self, t, inner=None):
        self.t, self.inner = t, inner
    def decode(self, data):
        time.sleep(self.t)
        return data / self.inner if self.inner is not None else data

def block(n=1000, Fs=1e3):
    return rp.Data(np.zeros(n, bool), Fs=Fs)

def test_a_decoder_is_counted():
    """Pipes (`/`, `*`: decoders, the heavy part) count, not only stages"""
    code = rp.code.load("block/hamming/7_4")
    d = rp.Data(np.zeros(7000, bool), Fs=1e3)
    d.stream.add_air(1.0)
    d / code
    load = d.stream.load()
    assert code in load and load[code] > 0

def test_what_runs_inside_counts_once():
    """The outer pipe's time is its own: far under the inner's, which a count of both would be over. The times are far
    apart as a sleep on a loaded runner overshoots by milliseconds"""
    inner = Slow(0.2)
    outer = Slow(0.02, inner)
    d = block()
    d.stream.add_air(1.0)
    d / outer
    b = d.stream.busy
    assert 0.015 < b[outer] < 0.15 and b[inner] > 0.19, b

def test_realtime_is_the_busiest_thread():
    st = Stream(1e3)
    st.add_air(1.0)
    run = lambda op, t: timed(st, op, time.sleep, t)
    run("a", 0.1); run("b", 0.1)                               # this thread: 0.2
    th = threading.Thread(target=run, args=("c", 0.3)); th.start(); th.join()   # another: 0.3
    assert st.busy["c"] > 0.29 and st.realtime() == pytest.approx(st.busy["c"]) # (what the sleeps took, not what was asked)
    st.thread_of["c"] = "pool"
    st.lanes["pool"] = 3                                        # three workers share it: 0.1 of their capacity
    assert st.realtime() == pytest.approx(st.busy["a"] + st.busy["b"])

def test_window_sees_a_stage_that_slowed_down():
    st = Stream(1e3)
    for i in range(200): # 2 s of signal; the op takes half real time, then twice it
        st.busy["op"] = st.busy.get("op", 0.0) + (0.005 if i < 100 else 0.02)
        st.add_air(0.01)
    assert st.load()["op"] == pytest.approx(1.25, rel=0.02)
    assert st.load(window=0.5)["op"] == pytest.approx(2.0, rel=0.06)
    assert st.realtime(window=0.5) == pytest.approx(2.0, rel=0.06)

def test_nothing_before_signal():
    st = Stream(1e3)
    assert st.load() == {} and st.realtime() == 0.0

def test_a_source_counts_its_signal_and_stages_their_time():
    src = rp.source.Memory(np.zeros(40000, np.complex64), Fs=1e6)
    st = None
    for d in src(4000) + rp.signal.DCBlock():
        st = d.stream
    assert st.air == pytest.approx(0.04) and len(st.load()) == 1 and st.realtime() > 0
