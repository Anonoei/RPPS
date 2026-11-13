"""Throughput of RPPS stages, Python loops vs the C++ kernels (rpps._native), and of a chain threaded vs not

    python tests/bench_stages.py              # every stage, 1 M samples
    python tests/bench_stages.py -n 4000000 --block 65536
"""
import argparse
import time

import numpy as np

import rpps as rp
from rpps import _accel

STAGES = {
    "AGC": (lambda: rp.sync.AGC(1.0, 0.01), dict(Fs=1e6)),
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
    c = rp.code.load("conv", "k7_133_171/1_2")
    x = llr_of(c._enc.encode(rng.integers(0, 2, 20000).astype(bool)), rng, 3.0)
    out.append(("Viterbi K7 1/2", lambda: c._dec.decode(x), 20000))
    t = rp.code.load("block", "trellis/p25_1_2")._dec
    tx = np.array([llr_of(t.encode(m), rng, 3.0) for m in rng.integers(0, 2, (64, 192)).astype(bool)])
    out.append(("fsm P25 1/2 (64 frames)", lambda: t.decode_batch(tx), 64*192))
    tb = rp.code.load("block", "turbo/lte", K=6144)
    tbx = llr_of(tb._enc.encode(rng.integers(0, 2, 6144).astype(bool)), rng, 0.8)
    out.append(("turbo LTE K=6144", lambda: tb._dec.decode(tbx), 6144))
    ld = rp.code.load("block", "ldpc/dvbs2_64800_1_2")
    ldx = llr_of(np.zeros(ld._dec.den, dtype=bool), rng, 1.5) # the all-zero codeword (encoding needs dense GF(2) elimination)
    out.append(("LDPC DVB-S2 64800 1/2", lambda: ld._dec.decode(ldx), ld._dec.num))
    r = rp.code.load("block", "rs/204_188")
    cw = r._enc.encode(rng.integers(0, 256, 188*200), symbols=True).reshape(200, 204).copy()
    for w in cw:
        w[rng.choice(204, 6, replace=False)] ^= rng.integers(1, 256, 6)
    out.append(("RS(204,188) 200 words", lambda: r._dec.decode(cw.reshape(-1), symbols=True), 200*188*8))
    b = rp.code.load("block", "bch/dvbs2_1_2")
    bw = b._enc.encode(rng.integers(0, 2, b._dec.num*4).astype(bool)).reshape(4, -1).copy()
    for w in bw:
        w[rng.choice(len(w), 10, replace=False)] ^= True
    out.append(("BCH DVB-S2 1/2, 4 words", lambda: b._dec.decode(bw.reshape(-1)), 4*b._dec.num))
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

def chain():
    """A receiver after the channel filter: matched filter, AGC, timing, carrier (sps 8 in, symbols out)"""
    taps = rp.signal.filters.designer.designer(129, 0.6e6, fs=8e6)
    return (rp.signal.filters.filters.ConvFilter(taps) + rp.sync.AGC(1.0, 0.001) + rp.sync.time.MM(alpha=0.1)
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

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=1_000_000, help="samples per stage")
    ap.add_argument("--block", type=int, default=16384, help="block length")
    ap.add_argument("--python-n", type=int, default=100_000, help="samples for the (slow) Python loops")
    ap.add_argument("--no-decoders", action="store_true", help="skip the FEC decoders")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(a.n) + 1j*rng.standard_normal(a.n)).astype(np.complex64)
    print(f"{'stage':16s} {'python MS/s':>12s} {'native MS/s':>12s} {'speedup':>8s}")
    for name, (make, rates) in STAGES.items():
        py = rate(make, x[:a.python_n], a.block, rates, native=False)
        nat = rate(make, x, a.block, rates, native=True) if _accel.available() else float("nan")
        print(f"{name:16s} {py/1e6:12.3f} {nat/1e6:12.2f} {nat/py:7.0f}x")
    if not _accel.available():
        print("rpps._native is not built: python setup.py build_ext -i")
        return
    print(f"\nchain (FIR 129 + AGC + MM + Costas), blocks of {a.block}")
    for how in ("one thread", "pipelined"):
        print(f"  {how:12s} {chain_rate(x, a.block, how)/1e6:6.2f} MS/s")
    if a.no_decoders:
        return
    print(f"\n{'decoder':26s} {'python Mb/s':>12s} {'native Mb/s':>12s} {'speedup':>8s}")
    for name, fn, nbits in decoders(rng):
        py, nat = timed(fn, False), timed(fn, True)
        print(f"{name:26s} {nbits/py/1e6:12.4f} {nbits/nat/1e6:12.3f} {py/nat:7.0f}x")

if __name__ == "__main__":
    main()
