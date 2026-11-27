"""Blind coding analysis measured: which codes and scramblers rpps.analysis.identify breaks, from how many symbols

    python tools/blind_bench.py                                   # every case, the default budgets and Es/N0
    python tools/blind_bench.py --cases conv_k7_34 cascade_idle --esn0 8 4 --symbols 8192 32768
    python tools/blind_bench.py --jobs 4 --update-baseline        # keep it in .benchmarks/ (--check compares)

Each case is a synthetic QPSK stream (random data unless the case says otherwise), its symbols turned a quarter turn,
cut at a random bit, with complex Gaussian noise at the given Es/N0 (unit symbol energy), given to
`identify(symbols, "qpsk", blind=True)`. A case is broken at (Es/N0, symbols) when what identify reports is the code
or scrambler that made the stream:

  conv_k7_12      K=7 (133,171) rate 1/2: a configured code (fec.name)
  conv_k7_34      the same punctured to 3/4: a configured code
  dvbs_rs         dvbs/qpsk_3_4 (3/4 conv, Forney interleaver, RS(204,188)): the RS found inside (fec.inner)
  punct_67        the K=7 mother punctured to 6/7 by a pattern no definition has: the puncture found blind
  conv_k6         K=6 rate 1/2 with generators no definition has: the generators guessed (fec.guesses)
  conv_k8         K=8 rate 1/2, unconfigured: outside scan_conv's K range today
  cascade_idle    3/4 conv over two feedthrough scramblers in cascade (1+x^18+x^23 each) over a repeating word:
                  the scrambler found inside
  ft_3tap         rate 1/2 conv over a 3-tap feedthrough scrambler over framed data: the scrambler found inside
  bch_63_51       BCH(63,51) words, no convolutional code: the block code found blind
  rs_blind        rate 1/2 conv over RS(255,239) with an unusual first root: the RS found blind

Recorded per case and Es/N0: the fewest symbols (of --symbols) at which it is broken (lower is better; unbroken is
recorded as nothing), and per case the seconds identify took at the largest budget. The cases with no tool yet
(conv_k8, ft_3tap) are there so the work that adds one shows up here.
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench

CASES = ["conv_k7_12", "conv_k7_34", "dvbs_rs", "punct_67", "conv_k6", "conv_k8", "cascade_idle", "ft_3tap", "bch_63_51",
         "rs_blind"]
K7 = [[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]] # (133, 171)
K6 = [[1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 1]] # (65, 57): no definition
K8 = [[1, 1, 1, 1, 0, 0, 1, 1], [1, 0, 1, 1, 1, 0, 0, 1]] # (363, 271)
P34 = [[1, 0, 1], [1, 1, 0]]
P67 = [[1, 0, 1, 0, 1, 1], [1, 1, 0, 1, 0, 0]]

def _conv(gens, K, pattern=None):
    from rpps.code.types.conv import conv
    from rpps.logger import Logger
    return conv(Logger().Child("bench", 0), 1, len(gens), gens, K, pattern or [[1]]*len(gens))

def _feedthrough(d, taps):
    """y[n] = d[n] ^ y[n - t] for each tap"""
    y = np.zeros(len(d), np.uint8)
    for n in range(len(d)):
        v = int(d[n])
        for t in taps:
            if n >= t:
                v ^= int(y[n - t])
        y[n] = v
    return y

def _framed(rng, n):
    """Frames of 64 random bytes between 8 flag bytes (0x7E): data with structure for a scrambler to hide"""
    out = []
    while sum(len(x) for x in out) < n:
        out += [np.unpackbits(np.full(8, 0x7E, np.uint8)), np.unpackbits(rng.integers(0, 256, 64).astype(np.uint8))]
    return np.concatenate(out)[:n]

def bits_of(case, n, rng):
    """About n coded bits of a case"""
    import rpps as rp
    if case == "conv_k7_12":
        return _conv(K7, 7).encode(rng.integers(0, 2, n//2).astype(bool))
    if case == "conv_k7_34":
        return _conv(K7, 7, P34).encode(rng.integers(0, 2, n*3//4).astype(bool))
    if case == "punct_67":
        return _conv(K7, 7, P67).encode(rng.integers(0, 2, n*6//7).astype(bool))
    if case == "conv_k6":
        return _conv(K6, 6).encode(rng.integers(0, 2, n//2).astype(bool))
    if case == "conv_k8":
        return _conv(K8, 8).encode(rng.integers(0, 2, n//2).astype(bool))
    if case == "cascade_idle":
        word = np.array([int(c) for c in "101001110110000101111001"], np.uint8)
        d = _feedthrough(_feedthrough(np.tile(word, n*3//4//24 + 1), (20, 3)), (20, 3))
        return _conv(K7, 7, P34).encode(d.astype(bool))
    if case == "ft_3tap":
        return _conv(K7, 7).encode(_feedthrough(_framed(rng, n//2), (23, 18, 5)).astype(bool))
    if case == "bch_63_51":
        from rpps.code.types.bchgf import bchgf
        from rpps.logger import Logger
        return bchgf(Logger().Child("bench", 0), 63, 51, 6, 0x43, 2).encode(rng.integers(0, 2, n//63*51).astype(bool))
    if case == "rs_blind":
        from rpps.code.types.rs import rs
        from rpps.logger import Logger
        code = rs(Logger().Child("bench", 0), 239, 255, 8, 0x11d, 120, 1) # (k, n, m, field polynomial, fcr, prim)
        words = n//2//8//255 + 1
        cw = np.asarray(code.encode(rng.integers(0, 256, words*239), symbols=True)).astype(np.uint8)
        return _conv(K7, 7).encode(np.unpackbits(cw).astype(bool))
    raise ValueError(case)

def symbols_of(case, n_sym, esn0, seed):
    """n_sym QPSK symbols of the case: a random bit cut, a quarter turn, noise at esn0 dB"""
    import rpps as rp
    rng = np.random.default_rng(seed)
    q = rp.symbol.load("qpsk")
    if case == "dvbs_rs":
        pk = max(n_sym*2*3//4//(204*8) + 2, 4)
        ts = rng.integers(0, 256, (pk, 188)).astype(np.uint8)
        ts[:, 0] = 0x47
        d = rp.Data(np.unpackbits(ts.reshape(-1)).astype(bool), flush=True)
        for how, op in rp.profile.load("dvbs/qpsk_3_4").tx():
            if how == "*":
                d = d*op
        s = np.asarray(d.data).astype(np.complex64)[int(rng.integers(0, 50)):][:n_sym]
    else:
        b = np.asarray(bits_of(case, 2*n_sym + 400, rng)).astype(np.uint8)
        b = b[int(rng.integers(0, 101)):][:2*n_sym]
        s = q.modulate(b[:len(b)//2*2])
    s = s*np.exp(1j*np.pi/2)/np.sqrt(np.mean(np.abs(s)**2))
    sigma = np.sqrt(0.5*10**(-esn0/10))
    return (s + sigma*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s)))).astype(np.complex64)

def broken(case, f):
    """Whether identify's CodeDetection f names what made the case"""
    inner = [h.kind for h in f.inner]
    guesses = [g for g in f.guesses if g is not None]
    if case == "conv_k7_12":
        return f.found and f.kind == "conv" and f.name.endswith("/1_2") and "133" in f.name
    if case == "conv_k7_34": # A 3/4 pattern no definition has: configured by name, or the puncture found blind
        return (f.found and f.kind == "conv" and f.name.endswith("/3_4")) or \
            any(h.kind == "conv-punctured" and "rate 3/4" in h.name for h in f.inner)
    if case == "dvbs_rs":
        return f.found and any(h.name == "rs/204_188" for h in f.inner)
    if case == "punct_67":
        return "conv-punctured" in inner
    if case in ("conv_k6", "conv_k8"):
        want = K6 if case == "conv_k6" else K8
        return any(g.kind == "conv" and sorted(map(list, g.definition.get("generator", []))) == sorted(want) for g in guesses)
    if case in ("cascade_idle", "ft_3tap"): # (behind a code found blind, f.found is False)
        return any(k.startswith("scrambler") for k in inner)
    if case == "bch_63_51":
        return any(g.kind == "block" and g.definition.get("type") == "bch" for g in guesses)
    if case == "rs_blind":
        return any(k.startswith("rs") for k in inner)
    raise ValueError(case)

def trial(job):
    case, esn0, n_sym, seed = job
    from rpps.analysis.identify import identify
    s = symbols_of(case, n_sym, esn0, seed)
    t = time.time()
    try:
        f = identify(s, "qpsk", blind=True)
        ok = bool(broken(case, f))
    except Exception as e: # A tool that fails is a case not broken, said once
        print(f"  {case} {esn0} dB {n_sym}: {type(e).__name__}: {e}", flush=True)
        ok = False
    return case, esn0, n_sym, ok, time.time() - t

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", nargs="+", default=CASES, choices=CASES)
    ap.add_argument("--esn0", type=float, nargs="+", default=[10.0, 6.0, 3.0])
    ap.add_argument("--symbols", type=int, nargs="+", default=[8192, 32768, 131072])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    _bench.add_args(ap)
    a = ap.parse_args(argv)
    jobs = [(c, e, n, a.seed) for c in a.cases for e in a.esn0 for n in sorted(a.symbols)]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(trial, jobs))
    rec = _bench.Record("blind_bench")
    print(f"{'case':12} " + " ".join(f"{e:>7.1f}dB" for e in a.esn0) + "   (fewest symbols broken at; - never)    s at max")
    for c in a.cases:
        row = []
        for e in a.esn0:
            ok = [n for cc, ee, n, k, _ in res if cc == c and ee == e and k]
            best = min(ok) if ok else None
            rec.add(f"{c}@{e:g}dB symbols", best, "symbols", better="lower")
            row.append(f"{best:>9d}" if best else f"{'-':>9}")
        secs = max(t for cc, _, n, _, t in res if cc == c and n == max(a.symbols))
        rec.add(f"{c} seconds", secs, "s", better="lower")
        print(f"{c:12} " + " ".join(row) + f"   {secs:20.1f}")
    solved = sum(1 for c in a.cases for e in a.esn0 if any(cc == c and ee == e and k for cc, ee, _, k, _ in res))
    rec.add("broken (case, Es/N0)", solved, "count", better="higher", tol=0)
    print(f"broken: {solved} of {len(a.cases)*len(a.esn0)} (case, Es/N0)")
    return _bench.finish(a, rec)

if __name__ == "__main__":
    sys.exit(main())
