"""Frames per second of the DVB-S2 profile: the serial runner against the parallel runner, and where a frame's time goes

    python tools/parallel_bench.py [frames] [max_workers] [block]
    python tools/parallel_bench.py 16 8 --update-baseline   # keep this machine's rates in .benchmarks/ (--check compares)
    python tools/parallel_bench.py 64 4 --device             # the region on the GPU (rpps._accel.device(): CuPy, CUDA or HIP)
    python tools/parallel_bench.py 64 4 --device --timing mat   # and the matched filter and timing too (MatTrack)

block: samples per block fed to the receiver (default: the whole stream as one block). In blocks the serial front of one
block overlaps the region of the one before, as it does on a live source.

A synthetic stream of normal frames (QPSK 1/2 and 8PSK 3/4, pilots on), received from raw samples. Each runner is run once
to warm up (sequences, code tables) and the second run is timed. Only what comes after the frame boundary is parallel, so
the speed-up is bounded by the serial front (timing, carrier, frame sync).
"""
import argparse
import sys
import time
from collections import defaultdict

import numpy as np

import rpps as rp

import _bench

N = W = BLOCK = 0
BATCH = 16
XP = None # the region's array module (--device), else numpy
p = o = x = data = None

def setup(frames_n, workers, block, timing="known", device=False):
    """The profile and its transmitted stream, as the module's globals the runners read. timing: the receiver's method
    (known, oerder_meyr, mat); device: a `mat` front on the GPU too"""
    global N, W, BLOCK, p, o, x, data
    N, W, BLOCK = frames_n, workers, block
    p = rp.profile.load("dvbs2/normal", sync={"timing": {"method": timing, **({"device": True} if device and timing == "mat" else {})}})
    t = rp.profile.table("dvbs2/modcod")
    frames = [{"modcod": (4, 12)[i % 2], "pilots": 1} for i in range(N)]
    dfl = [t.row({**f, "short": 0})["kbch"] - 80 for f in frames]
    data = np.random.default_rng(1).integers(0, 2, sum(dfl)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=frames)
    x = (np.concatenate((o.data, np.zeros(200, np.complex64)))*np.exp(0.7j)).astype(np.complex64)

def run(rx):
    cuts = list(range(0, len(x), BLOCK or len(x)))
    t0 = time.perf_counter()
    outs, st = [], None
    for k, a in enumerate(cuts):
        d = rp.Data(x[a:a + (BLOCK or len(x))], Fs=o.Fs, sps=4, delay=o.delay, stream=st, flush=k == len(cuts) - 1)
        st = d.stream
        outs.append(np.asarray(rx(d).data).astype(bool))
    dt = time.perf_counter() - t0
    assert np.array_equal(np.concatenate(outs), data)
    return dt

def timed(make):
    """Warm up on a first runner, then time a fresh one (its operators are new; the tables and sequences are cached)"""
    rx = make()
    run(rx)
    close = getattr(rx, "close", None)
    if close:
        close()
    rx = make()
    dt = run(rx)
    if getattr(rx, "close", None):
        rx.close()
    return dt

def region_split():
    """Seconds per kernel of the region (inline, warmed up): what each BatchPlan kept"""
    make = lambda: p.rx(parallel=rp.parallel.Parallel("inline", xp=XP, max_batch=BATCH))
    run(make())
    rx = make()
    total = run(rx)
    spent = defaultdict(float)
    for plan in rx.sched.executor.worker.plans.values():
        for k, v in plan.spent.items():
            spent[k] += v
    return total, dict(spent)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("frames", type=int, nargs="?", default=16)
    ap.add_argument("max_workers", type=int, nargs="?", default=8)
    ap.add_argument("block", type=int, nargs="?", default=0, help="samples a block (0: the whole stream)")
    ap.add_argument("--device", action="store_true", help="the region's kernels on the GPU (thread and inline runs)")
    ap.add_argument("--batch", type=int, default=16, help="frames a batch at most (Parallel max_batch)")
    ap.add_argument("--timing", default="known", choices=["known", "oerder_meyr", "mat"],
                    help="the receiver's timing (mat: the matched filter fused in; on the GPU with --device)")
    _bench.add_args(ap)
    a = ap.parse_args(argv)
    setup(a.frames, a.max_workers, a.block, a.timing, a.device)
    global XP, BATCH
    BATCH = a.batch
    if a.device:
        XP, backend = rp._accel.device()
        if backend is None:
            ap.error("--device: no usable GPU (rpps._accel.device() gives numpy)")
    rec = _bench.Record("parallel")
    base = timed(p.rx)
    print(f"{N} frames, native={rp._accel.available()}, timing {a.timing}, region on {getattr(XP, '__name__', 'numpy')}, batch {BATCH}")
    print(f"serial        {base:6.2f} s  {N/base:6.2f} frames/s")
    rec.add("serial", N/base, "frames/s")
    w = 1
    while w <= W:
        dt = timed(lambda: p.rx(parallel=rp.parallel.Parallel("thread", workers=w, xp=XP, max_batch=BATCH)))
        print(f"thread x{w:<2}    {dt:6.2f} s  {N/dt:6.2f} frames/s  x{base/dt:.2f}")
        rec.add(f"thread x{w}", N/dt, "frames/s")
        w *= 2

    for w in (2, 4, 8):
        if w <= W and XP is None: # (a device does not cross a fork)
            dt = timed(lambda: p.rx(parallel=rp.parallel.Parallel("process", workers=w, max_batch=BATCH)))
            print(f"process x{w:<2}   {dt:6.2f} s  {N/dt:6.2f} frames/s  x{base/dt:.2f}")
            rec.add(f"process x{w}", N/dt, "frames/s")

    total, spent = region_split()
    reg = sum(spent.values())
    print(f"\nregion {reg:.2f} s of {total:.2f} s ({100*reg/total:.0f}%), per kernel:")
    for k, v in sorted(spent.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s} {v:6.3f} s  {100*v/reg:5.1f}%  {1000*v/N:6.2f} ms/frame")
        rec.add(f"region {k}", 1000*v/N, "ms/frame", "lower")
    return _bench.finish(a, rec)

if __name__ == "__main__":
    sys.exit(main())
