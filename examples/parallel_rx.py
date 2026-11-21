"""Use more than one core: a profile's frame-parallel region

    python examples/parallel_rx.py [frames] [workers]

Once frame sync has found a frame and its header has set the parameters, the rest of the work on that frame (demap,
deinterleave, LDPC, BCH, descramble) has no state shared with the next one. A profile marks where that region starts,
and `p.rx(parallel=rp.parallel.Parallel(kind, workers=n))` runs it on frames in batches: "inline" (one thread, batched),
"thread" (the C++ kernels release the GIL) or "process". Everything before the region runs as usual. The output is
the serial runner's, bit for bit, for any block cut, worker count and batch size.

Here: DVB-S2 normal frames (QPSK 1/2 and 8PSK 3/4 alternating, pilots on, each frame's own MODCOD from its header),
received serially and in parallel. Each runner is run once to warm up the shared tables and the second run is timed,
as tests/parallel_bench.py does in more detail. The `__main__` guard is needed: process workers import this file.
"""
import sys
import time

import numpy as np

import rpps as rp

def main(frames=8, workers=4):
    # Timing known: the transmitter's. The carrier and frame sync still run (README.dev.md "Parallelism")
    p = rp.profile.load("dvbs2/normal", sync={"timing": {"method": "known"}})
    table = rp.profile.table("dvbs2/modcod")
    fields = [{"modcod": (4, 12)[i % 2], "pilots": 1} for i in range(frames)] # one header per frame
    sizes = [table.row({**f, "short": 0})["kbch"] - 80 for f in fields] # data bits each frame carries
    data = np.random.default_rng(1).integers(0, 2, sum(sizes)).astype(bool)
    wave = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=fields)
    x = (np.concatenate((wave.data, np.zeros(200, np.complex64)))*np.exp(0.7j)).astype(np.complex64) # a phase offset

    def receive(rx):
        t = time.perf_counter()
        out = rx(rp.Data(x, Fs=wave.Fs, sps=wave.sps, delay=wave.delay, flush=True))
        return np.asarray(out.data).astype(bool), time.perf_counter() - t

    def timed(make):
        receive(make()) # warm up
        rx = make()
        try:
            got, dt = receive(rx)
        finally:
            if hasattr(rx, "close"): # a parallel runner owns its workers
                rx.close()
        assert np.array_equal(got, data), "the output differs from what was sent"
        return dt

    print(f"{frames} DVB-S2 normal frames, {len(data)} data bits, native kernels: {rp._accel.available()}")
    base = timed(p.rx)
    print(f"  serial        {base:6.2f} s")
    for kind in ("thread", "process"):
        dt = timed(lambda: p.rx(parallel=rp.parallel.Parallel(kind, workers=workers)))
        print(f"  {kind:7s} x{workers}    {dt:6.2f} s  ({base/dt:.1f}x)")

if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
