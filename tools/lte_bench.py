"""Real-time factor of the LTE downlink profile: one thread against `pipelined` (the trunk and the branches a thread each)

    python tools/lte_bench.py                    # 50 RB at 15.36 MS/s, 40 frames (0.4 s of signal)
    python tools/lte_bench.py --n-rb 25 --frames 100
    python tools/lte_bench.py --update-baseline  # keep this machine's numbers in .benchmarks/ (--check compares)

A synthetic downlink (rpps.grid.lte.tx: two ports, SIB1 every 20 ms, 15 dB SNR, a 3.1 kHz offset) received by
`lte/downlink` in blocks of `--block` samples. Each way is run once to warm up and the next run is timed. The factor is
the receiver's time over the signal's: under 1.0 keeps up. Each step's own load (busy time over air time) shows where
the time goes; `pipelined` runs at about the slower of the trunk and the branches.
"""
import argparse
import sys
import time

import numpy as np

import rpps as rp
from rpps.grid import lte

import _bench

FS = {6: 1.92e6, 15: 3.84e6, 25: 7.68e6, 50: 15.36e6, 75: 23.04e6, 100: 30.72e6}

def signal(n_rb, frames):
    sib = np.random.default_rng(1).integers(0, 2, 224).astype(np.uint8)
    return lte.tx.channel(lte.tx.downlink(pci=77, ports=2, n_rb=n_rb, frames=frames, sib=sib), cfo=3.1e3, fs=FS[n_rb],
                          delay=1234, snr_db=15.0, seed=1)

def run(x, n_rb, block, pipelined):
    """(seconds, {step: load}, SIB1 transport blocks decoded) of one receive"""
    rx = rp.profile.load("lte/downlink", n_rb=n_rb).rx()
    src = rp.source.Memory(x, Fs=FS[n_rb])
    blocks = src(block)
    t = time.perf_counter()
    outs = list(blocks.pipelined(rx) if pipelined else blocks + rx)
    dt = time.perf_counter() - t
    air = src.stream.air
    load = {f"trunk {k}": v for k, v in src.stream.load().items()}
    for name in outs[-1]:
        load.update({f"{name} {k}": v/air for k, v in outs[-1][name].stream.busy.items()})
    sibs = sum(1 for m in rx.take_messages() if m.kind == "pdsch" and m.crc_ok)
    return dt, load, sibs

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-rb", type=int, default=50, choices=sorted(FS))
    ap.add_argument("--frames", type=int, default=40, help="10 ms radio frames of signal")
    ap.add_argument("--block", type=int, default=65536, help="samples a block")
    _bench.add_args(ap)
    a = ap.parse_args(argv)
    x = signal(a.n_rb, a.frames)
    air = len(x)/FS[a.n_rb]
    rec = _bench.Record("lte")
    print(f"lte/downlink {a.n_rb} RB at {FS[a.n_rb]/1e6:.2f} MS/s, {air:.2f} s of signal, native={rp._accel.available()}")
    for name, pipe in (("one thread", False), ("pipelined", True)):
        run(x, a.n_rb, a.block, pipe) # (warm up: tables, the decoder's buffers)
        dt, load, sibs = run(x, a.n_rb, a.block, pipe)
        print(f"{name:11s} {dt:6.2f} s  {dt/air:5.2f}x real time  ({sibs} transport blocks)")
        rec.add(f"{name} real time", dt/air, "x", "lower")
        if not pipe:
            for k, v in sorted(load.items(), key=lambda kv: -kv[1]):
                print(f"   {v:6.3f}  {k}")
    return _bench.finish(a, rec)

if __name__ == "__main__":
    sys.exit(main())
