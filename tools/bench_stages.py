"""Throughput of RPPS stages, Python loops vs the C++ kernels (rpps._native), and of a chain threaded vs not

    python tools/bench_stages.py              # every stage, 1 M samples
    python tools/bench_stages.py -n 4000000 --block 65536
    python tools/bench_stages.py --blocks     # block-rate sync stages, filters, TX shaping: best of 5 at two block sizes
    python tools/bench_stages.py --update-baseline   # keep this machine's rates in .benchmarks/ (--check compares)
"""
import argparse
import sys
import time

import numpy as np

import rpps as rp
from rpps import _accel

import _bench

STAGES = {
    "AGC": (lambda: rp.sync.AGC(1.0), dict(Fs=1e6)),
    "Costas (QPSK)": (lambda: rp.sync.Costas.from_bandwidth(0.02, order=4), dict(Fs=1e6, sps=1)),
    "MM (sps 8)": (lambda: rp.sync.time.MM(), dict(Fs=8e6, sps=8)),
    "IIR (butter 6)": (lambda: rp.signal.filters.IIR(rp.signal.filters.iir.design("butter", 6, "lowpass", 0.1)), dict(Fs=1e6)),
    "DCBlock": (lambda: rp.signal.DCBlock(), dict(Fs=1e6)),
    "CMA (11 taps)": (lambda: rp.sync.equalize.CMA(11, 0.002), dict(Fs=1e6, sps=1)),
}

def llr_of(bits, rng, ebn0):
    s = 1.0 - 2.0*np.asarray(bits, dtype=np.float64)
    sigma = 10**(-ebn0/20)
    return 2*(s + sigma*rng.standard_normal(len(s)))/sigma**2

def decoders(rng):
    """(name, decode(), information bits per call) for each decoder, set up on a noisy codeword stream"""
    out = []
    c = rp.code.load("conv/k7_133_171/1_2")
    x = llr_of(c.encoder.encode(rng.integers(0, 2, 20000).astype(bool)), rng, 3.0)
    out.append(("Viterbi K7 1/2", lambda: c.decoder.decode(x), 20000))
    t = rp.code.load("block/trellis/p25_1_2").decoder
    tx = np.array([llr_of(t.encode(m), rng, 3.0) for m in rng.integers(0, 2, (64, 192)).astype(bool)])
    out.append(("fsm P25 1/2 (64 frames)", lambda: t.decode_batch(tx), 64*192))
    tb = rp.code.load("block/turbo/lte", K=6144)
    tbx = llr_of(tb.encoder.encode(rng.integers(0, 2, 6144).astype(bool)), rng, 0.8)
    out.append(("turbo LTE K=6144", lambda: tb.decoder.decode(tbx), 6144))
    ld = rp.code.load("block/ldpc/dvbs2_64800_1_2")
    ldx = llr_of(np.zeros(ld.decoder.den, dtype=bool), rng, 1.5) # the all-zero codeword (encoding needs dense GF(2) elimination)
    out.append(("LDPC DVB-S2 64800 1/2", lambda: ld.decoder.decode(ldx), ld.decoder.num))
    r = rp.code.load("block/rs/204_188")
    cw = r.encoder.encode(rng.integers(0, 256, 188*200), symbols=True).reshape(200, 204).copy()
    for w in cw:
        w[rng.choice(204, 6, replace=False)] ^= rng.integers(1, 256, 6)
    out.append(("RS(204,188) 200 words", lambda: r.decoder.decode(cw.reshape(-1), symbols=True), 200*188*8))
    b = rp.code.load("block/bch/dvbs2_1_2")
    bw = b.encoder.encode(rng.integers(0, 2, b.decoder.num*4).astype(bool)).reshape(4, -1).copy()
    for w in bw:
        w[rng.choice(len(w), 10, replace=False)] ^= True
    out.append(("BCH DVB-S2 1/2, 4 words", lambda: b.decoder.decode(bw.reshape(-1)), 4*b.decoder.num))
    return out

def timed(fn, native):
    saved = _accel.native
    _accel.native = saved if native else None
    try:
        t = time.perf_counter()
        fn()
        return time.perf_counter() - t
    finally:
        _accel.native = saved

def rate(make, x, block, rates, native):
    """Samples per second through a fresh stage, x cut into blocks of `block`"""
    saved = _accel.native
    _accel.native = saved if native else None
    try:
        stage, stream = make(), None
        t = time.perf_counter()
        for i in range(0, len(x), block):
            d = rp.Data(x[i:i+block], stream=stream, **rates)
            stream = d.stream
            d + stage
        return len(x)/(time.perf_counter() - t)
    finally:
        _accel.native = saved

def test_signals(n, sps=4, frame=1000, seed=1):
    """QPSK at `sps` samples per symbol (RRC 0.35, a small carrier offset, noise) and its symbols with a known preamble
    every `frame` symbols. Made with numpy alone, so the stages under test don't make their own input"""
    rng = np.random.default_rng(seed)
    nsym = n//sps
    syms = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, nsym) + 1))
    pre = np.exp(1j*np.pi/4*(2*np.random.default_rng(7).integers(0, 4, 64) + 1))
    for k in range(0, nsym - len(pre), frame):
        syms[k:k+len(pre)] = pre
    up = np.zeros(nsym*sps, dtype=np.complex128)
    up[::sps] = syms
    h = rp.signal.filters.ShapingRRC(None, sps, 0.35).pulse
    x = np.convolve(up, h/np.sqrt(np.sum(h*h)), "same")
    x *= np.exp(2j*np.pi*1e-4*np.arange(len(x)))
    x += 0.1*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    s = syms*np.exp(2j*np.pi*2e-4*np.arange(nsym)) + 0.1*(rng.standard_normal(nsym) + 1j*rng.standard_normal(nsym))
    return x.astype(np.complex64), s.astype(np.complex64), pre

def block_stages(pre, sps=4):
    """(name, make, input 'x' or 's', rates): the block-rate stages, filters and shifts, the TX shaping"""
    Fs = 4e6
    xr, sr = dict(Fs=Fs, sps=sps), dict(Fs=Fs/sps, sps=1)
    F = rp.signal.filters
    return [
        ("FIR 129", lambda: F.low_pass(129, 0.5e6, Fs), "x", xr),
        ("FIR 1025", lambda: F.low_pass(1025, 0.5e6, Fs), "x", xr),
        ("ShiftFreq", lambda: F.ShiftFreq(10e3), "x", xr),
        ("OerderMeyr", lambda: rp.sync.time.OerderMeyr(sps), "x", xr),
        ("MPower", lambda: rp.sync.phase.MPower(4), "s", sr),
        ("FrameSync", lambda: rp.sync.FrameSync(pre), "s", sr),
        ("BlockLS", lambda: rp.sync.equalize.BlockLS(rp.symbol.load("QPSK")), "s", sr),
        ("TX Zeros+RRC sps4", lambda: rp.signal.sample.up.Zeros(4) + F.ShapingRRC(65, 4), "s", sr),
        ("TX Zeros+RRC sps8", lambda: rp.signal.sample.up.Zeros(8) + F.ShapingRRC(129, 8), "s", sr),
        ("TX Interpolate sps4", lambda: rp.signal.sample.up.Interpolate(4, F.ShapingRRC(65, 4)), "s", sr),
        ("TX Interpolate sps8", lambda: rp.signal.sample.up.Interpolate(8, F.ShapingRRC(129, 8)), "s", sr),
    ]

def best_rate(make, x, block, rates, repeat):
    """Best of `repeat` runs (one process: a shared machine's load makes runs across processes incomparable), items/s"""
    best = 0.0
    for _ in range(repeat):
        stage, stream = make(), None
        t = time.perf_counter()
        for i in range(0, len(x), block):
            d = rp.Data(x[i:i+block], stream=stream, **rates)
            stream = d.stream
            d + stage
        best = max(best, len(x)/(time.perf_counter() - t))
    return best

def chain():
    """A receiver after the channel filter: matched filter, AGC, timing, carrier (sps 8 in, symbols out)"""
    taps = rp.signal.filters.designer.designer(129, 0.6e6, fs=8e6)
    return (rp.signal.filters.filters.ConvFilter(taps) + rp.sync.AGC(1.0) + rp.sync.time.MM(alpha=0.1)
            + rp.sync.Costas.from_bandwidth(0.02, order=4))

def chain_rate(x, block, how):
    """Samples per second through the chain: one thread, or each stage in its own"""
    def source():
        stream = None
        for i in range(0, len(x), block):
            d = rp.Data(x[i:i+block], stream=stream, Fs=8e6, sps=8)
            stream = d.stream
            yield d
    blocks = rp.source.source.Blocks(source())
    t = time.perf_counter()
    blocks = blocks + chain() if how == "one thread" else blocks.pipelined(chain())
    for _ in blocks:
        pass
    return len(x)/(time.perf_counter() - t)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=1_000_000, help="samples per stage")
    ap.add_argument("--block", type=int, default=16384, help="block length")
    ap.add_argument("--python-n", type=int, default=100_000, help="samples for the (slow) Python loops")
    ap.add_argument("--no-decoders", action="store_true", help="skip the FEC decoders")
    ap.add_argument("--blocks", action="store_true", help="only the block-rate stages, filters and TX shaping, best of --repeat")
    ap.add_argument("--repeat", type=int, default=5, help="runs per block-rate stage (the best is kept)")
    _bench.add_args(ap)
    a = ap.parse_args(argv)
    rec = _bench.Record("stages_blocks" if a.blocks else "stages")
    if a.blocks:
        x, s, pre = test_signals(a.n)
        print(f"{'block stage':20s} " + " ".join(f"{f'MS/s @{b}':>12s}" for b in (4096, 65536)))
        for name, make, which, rates in block_stages(pre):
            data = x if which == "x" else s
            r = [best_rate(make, data, b, rates, a.repeat) for b in (4096, 65536)]
            print(f"{name:20s} " + " ".join(f"{v/1e6:12.2f}" for v in r))
            for b, v in zip((4096, 65536), r):
                rec.add(f"{name} @{b}", v/1e6, "MS/s")
        return _bench.finish(a, rec)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(a.n) + 1j*rng.standard_normal(a.n)).astype(np.complex64)
    print(f"{'stage':16s} {'python MS/s':>12s} {'native MS/s':>12s} {'speedup':>8s}")
    for name, (make, rates) in STAGES.items():
        py = rate(make, x[:a.python_n], a.block, rates, native=False)
        nat = rate(make, x, a.block, rates, native=True) if _accel.available() else float("nan")
        print(f"{name:16s} {py/1e6:12.3f} {nat/1e6:12.2f} {nat/py:7.0f}x")
        rec.add(f"{name} python", py/1e6, "MS/s")
        rec.add(f"{name} native", None if np.isnan(nat) else nat/1e6, "MS/s")
    if not _accel.available():
        print("rpps._native is not built: python tools/builder.py -c")
        return _bench.finish(a, rec)
    print(f"\nchain (FIR 129 + AGC + MM + Costas), blocks of {a.block}")
    for how in ("one thread", "pipelined"):
        r = rec.add(f"chain {how}", chain_rate(x, a.block, how)/1e6, "MS/s")
        print(f"  {how:12s} {r:6.2f} MS/s")
    if a.no_decoders:
        return _bench.finish(a, rec)
    print(f"\n{'decoder':26s} {'python Mb/s':>12s} {'native Mb/s':>12s} {'speedup':>8s}")
    for name, fn, nbits in decoders(rng):
        py, nat = timed(fn, False), timed(fn, True)
        print(f"{name:26s} {nbits/py/1e6:12.4f} {nbits/nat/1e6:12.3f} {py/nat:7.0f}x")
        rec.add(f"{name} python", nbits/py/1e6, "Mb/s")
        rec.add(f"{name} native", nbits/nat/1e6, "Mb/s")
    return _bench.finish(a, rec)

if __name__ == "__main__":
    sys.exit(main())
