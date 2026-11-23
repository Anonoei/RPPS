"""A parallel region's kernels on the host and on the GPU: milliseconds a frame, by batch size

    python tools/device_bench.py                    # DVB-S2 normal frames, batches of 1, 4, 16, 64
    python tools/device_bench.py --batches 16 64 --repeat 5
    python tools/device_bench.py --update-baseline  # keep this machine's numbers in .benchmarks/ (--check compares)

Host: the kernel on numpy (the C++ kernels when built, on all cores). Device: the same kernel on `rpps._accel.device()`
(CuPy on CUDA or ROCm/HIP), with the batch uploaded before and downloaded after, as a plan does; each time is the best of
--repeat runs after a warm-up (compiles, constants on the device), synchronized. Kernels: the LDPC decoder (`LdpcDecode`,
rates 1/2 and 3/4, Es/N0 a little above threshold so it iterates), BCH (`BchDecode`, clean codewords and 5
errors each) and the demapper (`Demap`, 8PSK)
"""
import argparse
import sys
import time
from types import SimpleNamespace

import numpy as np

import rpps as rp
from rpps.base.data import Digital, Mod
from rpps.base.kernel import Batch, Form, ToDevice, to_host
from rpps.parallel.kernel import kernel_of
from rpps.profile.steps import DECODE

import _bench

def ldpc_case(name, rows, noise, seed=1):
    code = rp.code.load(f"block/ldpc/{name}")
    rng = np.random.default_rng(seed)
    X = []
    for _ in range(rows):
        msg = rng.integers(0, 2, code.e_num).astype(bool)
        cw = np.asarray((rp.Data(msg, flush=True) * code).data).astype(bool)
        X.append((2.0/noise**2*((1 - 2.0*cw) + noise*rng.normal(size=len(cw)))).astype(np.float32))
    return code, np.array(X), Form(Mod, rp.Category.SYMBOL, rp.dtype.LLR, rp.Rep.CHANNEL, 1e6, 1, 0.0, 0.0)

def bch_case(name, rows, flips, seed=3):
    """Codewords with `flips` bit errors each (what an LDPC decoder leaves is mostly none: flips=0 is the common case)"""
    code = rp.code.load(f"block/bch/{name}")
    rng = np.random.default_rng(seed)
    X = []
    for _ in range(rows):
        cw = np.asarray((rp.Data(rng.integers(0, 2, code.e_num).astype(bool), flush=True) * code).data).astype(bool)
        cw[rng.choice(len(cw), flips, replace=False)] ^= True
        X.append(cw)
    return code, np.array(X), Form(Digital, rp.Category.CODE, rp.dtype.BITS, rp.Rep.CHANNEL, 1e6, 1, 0.0, 0.0)

def demap_case(rows, seed=2):
    c = rp.symbol.load("dvbs2_8psk")
    rng = np.random.default_rng(seed)
    pts = np.asarray(c.points)
    X = pts[rng.integers(0, len(pts), (rows, 21600))] + 0.1*(rng.normal(size=(rows, 21600)) + 1j*rng.normal(size=(rows, 21600)))
    return c, X.astype(np.complex64), Form(Mod, rp.Category.SYMBOL, rp.dtype.SYMBOLS, rp.Rep.SYMBOLS, 1e6, 1, 0.0, 0.0)

def best(fn, repeat):
    fn() # warm up
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)

def timing(op, X, f, xp, repeat):
    k = kernel_of(SimpleNamespace(id="b", rx=[(DECODE, op)], tx=[], controls={}), {})
    mk = lambda: Batch(np.array(X), f, [0]*len(X), X.shape[1])
    host = best(lambda: k.run(np, mk()), repeat)
    dev = None
    if xp is not None:
        def on_dev():
            b = k.run(xp, ToDevice().run(xp, mk()))
            b.x = to_host(b.x)
            return b
        ref, got = k.run(np, mk()), on_dev()
        same = np.mean(ref.x == got.x) if ref.x.dtype == bool else None
        if same is not None and same < 0.999:
            print(f"  WARNING: {k.name}: device bits agree with the host on only {100*same:.2f}%")
        dev = best(on_dev, repeat)
    return k.name, host, dev

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--repeat", type=int, default=3)
    _bench.add_args(ap)
    a = ap.parse_args(argv)
    xp, backend = rp._accel.device()
    xp = xp if backend else None
    gpu = xp.cuda.runtime.getDeviceProperties(0)["name"].decode() if xp is not None else "none"
    print(f"device: {backend or 'none'} ({gpu}), native={rp._accel.available()}")
    rec = _bench.Record("device")
    cases = [("ldpc 1/2", lambda n: ldpc_case("dvbs2_64800_1_2", n, 0.85)),
             ("ldpc 3/4", lambda n: ldpc_case("dvbs2_64800_3_4", n, 0.6)),
             ("bch 1/2", lambda n: bch_case("dvbs2_1_2", n, 0)),
             ("bch 1/2 +5", lambda n: bch_case("dvbs2_1_2", n, 5)),
             ("demap 8psk", demap_case)]
    print(f"{'kernel':12s} {'batch':>5s} {'host ms/fr':>11s} {'device ms/fr':>13s} {'speed-up':>9s}")
    for label, make in cases:
        for n in a.batches:
            op, X, f = make(n)
            name, host, dev = timing(op, X, f, xp, a.repeat)
            h = 1000*host/n
            rec.add(f"{label} x{n} host", h, "ms/frame", "lower")
            if dev is None:
                print(f"{label:12s} {n:5d} {h:11.2f} {'-':>13s}")
                continue
            d = 1000*dev/n
            rec.add(f"{label} x{n} device", d, "ms/frame", "lower")
            print(f"{label:12s} {n:5d} {h:11.2f} {d:13.2f} {host/dev:8.2f}x")
    return _bench.finish(a, rec)

if __name__ == "__main__":
    sys.exit(main())
