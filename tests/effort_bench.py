"""Effort levels (rp.set_effort) measured: the lowest Es/N0 each chain decodes at, and how fast, per level

    python tests/effort_bench.py                      # every chain, FER 1e-1, 20 frame errors a point
    python tests/effort_bench.py --chains polar_dci tpc --fer 0.01 --errors 50
    python tests/effort_bench.py --quick              # fewer errors, coarser search: a few minutes
    python tests/effort_bench.py --markdown           # table rows for rpps._effort as well

A chain: random messages, the code's encoder, the constellation, AWGN at Es/N0 (per symbol, unit mean power), the
constellation's LLRs with the exact noise variance (max-log or exact as the level says), the decoder. Each SNR point
reuses the same messages and noise shapes (scaled), so a lower SNR never decodes more by luck. The threshold is the
lowest Es/N0 (to --step dB) where the frame error rate is at most --fer, found by bisection; a point stops at --errors
frame errors. Eb/N0 = Es/N0 - 10 log10(bits per symbol x rate). Speed: information bits a second, one thread,
demapping and decoding, best of 3 over one batch, at the threshold (where `max` iterates most) and 1 dB above it.
The timing rows (OerderMeyr) are the EVM floor of a noiseless QPSK stream at 2 and 4 samples a symbol, and its symbols a
second.
"""
import argparse
import time

import numpy as np

import rpps as rp
from rpps import _accel, _effort

# name: (code ref, load params, constellation, words a batch, starting bracket of Es/N0 (dB))
CHAINS = {
    "ldpc_64800_1_2_qpsk": ("block/ldpc/dvbs2_64800_1_2", {}, "dvbs2_qpsk", 4, (0.0, 3.0)),
    "ldpc_16200_3_5_8psk": ("block/ldpc/dvbs2_16200_3_5", {}, "dvbs2_8psk", 8, (4.0, 8.0)),
    "turbo_lte_6144_qpsk": ("block/turbo/lte", {"K": 6144}, "qpsk", 2, (-3.0, 1.0)),
    "turbo_lte_40_qpsk": ("block/turbo/lte", {"K": 40}, "qpsk", 64, (-1.0, 4.0)),
    "polar_dci_qpsk": ("block/polar/nr_dci", {"K": 63, "E": 432}, "qpsk", 64, (-6.0, 0.0)),
    "polar_pbch_qpsk": ("block/polar/nr_pbch", {}, "qpsk", 64, (-10.0, -3.0)),
    "polar_256_128_qpsk": ("block/polar/256_128_crc16", {}, "qpsk", 64, (0.0, 5.0)),
    "tpc_64_57_bpsk": ("block/product/tpc_64_57_sq", {}, "bpsk", 16, (0.0, 6.0)),
    "conv_k7_1_2_qpsk": ("conv/k7_133_171/1_2", {}, "qpsk", 1, (0.0, 6.0)),
}

class Chain:
    def __init__(self, name, seed=1):
        ref, params, const, self.W, self.bracket = CHAINS[name]
        self.name, self.code = name, rp.code.load(ref, **params)
        self.c = rp.symbol.load(const)
        enc = self.code.encoder
        self.conv = ref.startswith("conv/")
        self.k = 2000 if self.conv else enc.num # message bits a word
        self.rate = enc.num/enc.den
        self.bps = self.c.bits_per_symbol
        self.seed = seed
        self._batches = {}

    def batch(self, i):
        """Messages (W, k) and unit noise of batch i, the same at every SNR"""
        if i not in self._batches:
            rng = np.random.default_rng([self.seed, i])
            m = rng.integers(0, 2, (self.W, self.k)).astype(bool)
            s = self.c.modulate(np.asarray(self.code.encoder.encode(m.reshape(-1))))
            n = (rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s)))/np.sqrt(2)
            self._batches[i] = (m, s, n)
        return self._batches[i]

    def run(self, i, esn0):
        """Frame errors of batch i at `esn0` (the level in force)"""
        m, s, n = self.batch(i)
        N0 = 10**(-esn0/10)
        return self.decode(s + np.sqrt(N0)*n, N0, m)

    def decode(self, y, N0, m):
        llr = self.c.llr(y, noise_var=N0, maxlog=_effort.choice("maxlog", self.c.maxlog))
        out = np.asarray(self.code.decoder.decode(llr)).astype(bool)[:m.size]
        if len(out) < m.size:
            return self.W
        return int(np.count_nonzero((out.reshape(m.shape) != m).any(axis=1)))

    def fer_ok(self, esn0, fer, errors):
        """Is the frame error rate at most `fer`? Stops at `errors` errors (no) or after errors/fer frames (yes)"""
        frames_max = int(np.ceil(errors/fer))
        e = f = i = 0
        while f < frames_max:
            e += self.run(i, esn0)
            f += self.W
            i += 1
            if e > fer*frames_max:
                return False, e, f
        return True, e, f

    def threshold(self, fer, errors, step):
        lo, hi = self.bracket
        while not self.fer_ok(hi, fer, errors)[0]:
            lo, hi = hi, hi + 2.0
        while self.fer_ok(lo, fer, errors)[0]:
            lo, hi = lo - 2.0, lo
        while hi - lo > step:
            mid = (lo + hi)/2
            if self.fer_ok(mid, fer, errors)[0]:
                hi = mid
            else:
                lo = mid
        return hi

    def speed(self, esn0, reps=3):
        """Information bits a second, demapping and decoding one batch"""
        m, s, n = self.batch(0)
        N0 = 10**(-esn0/10)
        y = s + np.sqrt(N0)*n
        best = np.inf
        for _ in range(reps):
            t = time.perf_counter()
            self.decode(y, N0, m)
            best = min(best, time.perf_counter() - t)
        return m.size/best

    def ebn0(self, esn0):
        return esn0 - 10*np.log10(self.bps*self.rate)

def qpsk_matched(n, sps, delay, beta=0.35, seed=0):
    """QPSK through RRC and its matched filter, delayed by `delay` samples (as tests/unit/test_effort.py)"""
    P = np.exp(1j*np.pi/4*(2*np.arange(4) + 1))
    s = P[np.random.default_rng(seed).integers(0, 4, n)]
    h = rp.signal.filters.ShapingRRC(None, sps, beta).pulse
    up = np.zeros(n*sps, complex)
    up[::sps] = s
    tx = np.convolve(up, h)
    rx = np.fft.ifft(np.fft.fft(tx)*np.exp(-2j*np.pi*np.fft.fftfreq(len(tx))*delay))
    return (np.convolve(rx, h)/np.sum(h*h)).astype(np.complex64)

def evm_db(y):
    P = np.exp(1j*np.pi/4*(2*np.arange(4) + 1))
    y = y/np.sqrt(np.mean(np.abs(y)**2))
    return 10*np.log10(np.mean(np.min(np.abs(y[:, None] - P[None, :])**2, axis=1)))

def timing(levels, n=60000):
    rows = []
    for sps in (2, 4):
        x = qpsk_matched(n, sps, 0.5 if sps == 2 else 1.3)
        for lvl in levels:
            with rp.effort(lvl):
                best, out = np.inf, None
                for _ in range(3):
                    t = time.perf_counter()
                    d = rp.Data(x, Fs=sps*1e3, sps=sps, flush=True) + rp.sync.time.OerderMeyr(sps)
                    out = np.asarray(d.data)
                    best = min(best, time.perf_counter() - t)
            rows.append((sps, lvl, evm_db(out[n//4:-100]), len(out)/best))
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", nargs="*", default=list(CHAINS) + ["timing"], help=f"of {list(CHAINS) + ['timing']}")
    ap.add_argument("--levels", nargs="*", default=list(_effort.LEVELS))
    ap.add_argument("--fer", type=float, default=0.1)
    ap.add_argument("--errors", type=int, default=20)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true", help="10 errors, 0.1 dB")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.errors, a.step = 10, 0.1
    print(f"native kernels: {'yes' if _accel.available() else 'NO (Python loops)'}; FER {a.fer:g}, {a.errors} errors a point, {a.step} dB")
    md = []
    for name in a.chains:
        if name == "timing":
            print("\ntiming (OerderMeyr, noiseless QPSK, beta 0.35)\n  sps  level      EVM dB   Msym/s")
            for sps, lvl, evm, sp in timing(a.levels):
                print(f"  {sps:3d}  {lvl:9s} {evm:7.1f}  {sp/1e6:7.3f}")
                md.append(f"| timing {sps} sps | {lvl} | EVM {evm:.1f} dB | | {sp/1e6:.3f} Msym/s | |")
            continue
        ch = Chain(name)
        print(f"\n{name}  (rate {ch.rate:.3f}, {ch.bps} bits/symbol)\n  level      Es/N0   Eb/N0   Mbit/s@thr  Mbit/s@thr+1   search s")
        for lvl in a.levels:
            with rp.effort(lvl):
                t = time.perf_counter()
                thr = ch.threshold(a.fer, a.errors, a.step)
                took = time.perf_counter() - t
                s0, s1 = ch.speed(thr), ch.speed(thr + 1.0)
            print(f"  {lvl:9s} {thr:6.2f}  {ch.ebn0(thr):6.2f}   {s0/1e6:9.4f}   {s1/1e6:11.4f}   {took:8.1f}", flush=True)
            md.append(f"| {name} | {lvl} | {thr:.2f} | {ch.ebn0(thr):.2f} | {s0/1e6:.3f} | {s1/1e6:.3f} |")
    if a.markdown:
        print("\n| Chain | Level | Es/N0 dB | Eb/N0 dB | Mbit/s at threshold | Mbit/s 1 dB above |\n|---|---|---|---|---|---|")
        print("\n".join(md))

if __name__ == "__main__":
    main()
