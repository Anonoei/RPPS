"""Throughput of FractionalPolyphase (20 MS/s -> 10.7622 MS/s, with the channel moved to baseband), against what it replaces

    python tools/resample_bench.py [seconds of air] [block]
    python tools/resample_bench.py 2 --update-baseline   # keep this machine's figures in .benchmarks/ (--check compares)

Native kernel (threaded), the Python reference (RPPS_NO_NATIVE), and, when projects/atsc/atsc_front.py is still there, its ctypes helper.
Each is run once to warm up and the second run is timed; the figure is CPU-seconds per second of air, wall and with the threads summed.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

import rpps as rp
from rpps import _accel

import _bench

FS = 20e6
RATIO = 4.5e6*684/286/FS
SECONDS, BLOCK, n, x = 2.0, 1 << 20, 0, None

def setup(seconds, block):
    global SECONDS, BLOCK, n, x
    SECONDS, BLOCK = seconds, block
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

def timed(name, fn, rec):
    fn()
    t0, c0 = time.perf_counter(), time.process_time()
    out = fn()
    w, c = time.perf_counter() - t0, time.process_time() - c0
    print(f"{name:28s} wall {w/SECONDS:7.3f} s/s   cpu {c/SECONDS:7.3f} s/s   ({out} samples out)")
    rec.add(f"{name} wall", w/SECONDS, "s/s", "lower")
    rec.add(f"{name} cpu", c/SECONDS, "s/s", "lower")

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seconds", type=float, nargs="?", default=2.0, help="seconds of air")
    ap.add_argument("block", type=int, nargs="?", default=1 << 20)
    _bench.add_args(ap)
    a = ap.parse_args(argv)
    setup(a.seconds, a.block)
    rec = _bench.Record("resample")
    timed("FractionalPolyphase native", stage_run, rec)
    saved, _accel.native = _accel.native, None
    timed("FractionalPolyphase python", stage_run, rec)
    _accel.native = saved
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "projects" / "atsc"))
        import atsc_front
        def front_run():
            f = atsc_front.Front(604e6, 604.5e6, FS/(RATIO*FS))
            return sum(len(f.process(x[a:a + BLOCK])) for a in range(0, n, BLOCK))
        timed("atsc_front.Front (ctypes)", front_run, rec)
    except ImportError:
        pass
    return _bench.finish(a, rec)

if __name__ == "__main__":
    sys.exit(main())
