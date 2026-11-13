"""Throughput of FractionalPolyphase (20 MS/s -> 10.7622 MS/s, with the channel moved to baseband), against what it replaces

    python tests/resample_bench.py [seconds of air] [block]

Native kernel (threaded), the Python reference (RPPS_NO_NATIVE), and, when projects/atsc/atsc_front.py is still there, its ctypes helper.
Each is run once to warm up and the second run is timed; the figure is CPU-seconds per second of air, wall and with the threads summed.
"""
import os
import sys
import time

import numpy as np

import rpps as rp
from rpps import _accel

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 20
FS = 20e6
RATIO = 4.5e6*684/286/FS
n = int(SECONDS*FS)
x = (np.random.default_rng(0).standard_normal(n) + 1j*np.random.default_rng(1).standard_normal(n)).astype(np.complex64)

def stage_run():
    stage = rp.signal.sample.FractionalPolyphase(RATIO, cutoff=3.7e6, shift=-0.5e6)
    stream, out = None, 0
    for a in range(0, n, BLOCK):
        d = rp.Data(x[a:a + BLOCK], Fs=FS, sps=2, cf=604e6, stream=stream)
        stream = d.stream
        out += len(np.asarray((d + stage).data))
    return out

def timed(name, fn):
    fn()
    t0, c0 = time.perf_counter(), time.process_time()
    out = fn()
    w, c = time.perf_counter() - t0, time.process_time() - c0
    print(f"{name:28s} wall {w/SECONDS:7.3f} s/s   cpu {c/SECONDS:7.3f} s/s   ({out} samples out)")

timed("FractionalPolyphase native", stage_run)
saved, _accel.native = _accel.native, None
timed("FractionalPolyphase python", stage_run)
_accel.native = saved
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "projects", "atsc"))
    import atsc_front
    def front_run():
        f = atsc_front.Front(604e6, 604.5e6, FS/(RATIO*FS))
        return sum(len(f.process(x[a:a + BLOCK])) for a in range(0, n, BLOCK))
    timed("atsc_front.Front (ctypes)", front_run)
except ImportError:
    pass
