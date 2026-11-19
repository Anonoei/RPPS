"""Blind FEC identification

Takes hard bits (what a demodulator produced, with an unknown bit mapping, rotation and alignment)
and asks which code they are most likely carrying. Every test is a parity relation that holds on
codewords but is a coin flip on anything else, so a hypothesis is scored by how far its relation
sits from 50% (a deviance z-score), then corrected for how many hypotheses were tried.

 conv: rate 1/n feed-forward convolutional codes. For generators g_a, g_b: (c_a * g_b) xor (c_b * g_a) = 0
 block: any linear block code / LDPC given its parity-check matrix H, at every word alignment
 scramblers: an additive scrambler of period T just adds a fixed pattern to the relation, so folding the relation
    by T (with `periods`) finds it. Self-synchronising scramblers are undone with `descramblers` first.

The tests are about structure, not about being right: a noisy channel weakens them, and the verdict
is None when nothing stands out.
"""
import functools
import itertools

import numpy as np

from ..math.gf2 import nullspace, systematic
from ..math.prob import deviance_z, z_critical
from ..code.coding import puncture_period, mismatch, cached

class Hypothesis:
    __slots__ = ("kind", "name", "z", "params", "coding", "guess")
    def __init__(self, kind, name, z, params=None, coding=None, guess=None):
        self.kind = kind
        self.name = name
        self.z = z
        self.params = params or {}
        self.coding = coding # (folder, config name) when it is in config/fec
        self.guess = guess # The Guess (definition to save) a blind find comes with

    def __repr__(self):
        c = f" [{self.coding[0]}/{self.coding[1]}]" if self.coding else ""
        return f"<{self.kind} {self.name} z={self.z:.1f}{c} {self.params}>"

class Report:
    """Ranked hypotheses and a verdict"""
    __slots__ = ("hyps", "trials", "z_crit", "facts")
    def __init__(self, hyps, trials, facts):
        self.hyps = sorted(hyps, key=lambda h: -h.z)
        self.trials = trials
        self.z_crit = z_critical(trials)
        self.facts = facts

    @property
    def best(self):
        return self.hyps[0] if self.hyps else None

    @property
    def verdict(self):
        """The best hypothesis if it stands out from chance, else None"""
        return self.best if self.best and self.best.z > self.z_crit else None

    def __str__(self):
        out = [f"{self.trials} hypotheses tried, significance threshold z>{self.z_crit:.1f}"]
        out += [f"  {h}" for h in self.hyps[:5]]
        out.append(f"verdict: {self.verdict if self.verdict else 'no FEC structure found'}")
        return "\n".join(out)

# --- Bit streams from symbols -------------------------------------------------------------------------

def _constellation(c):
    from .. import symbol
    return symbol.load_or_generate(c) if isinstance(c, str) else c

def _hard(y, pts, labels):
    """Hard bits of symbols y: the labels (values) of the nearest points"""
    idx = np.argmin(np.abs(np.asarray(y)[:, None] - np.asarray(pts)[None, :]), axis=1)
    bps = int(np.log2(len(pts)))
    return ((np.asarray(labels, dtype=np.int64)[idx][:, None] >> np.arange(bps-1, -1, -1)) & 1).astype(np.uint8).reshape(-1)

def diff_alphabets(constellation):
    """Phase-change alphabets a differential receiver of `constellation` would demap against: {name: Constellation}

    The dpsk definitions (config/constellation/dpsk) with as many points, and the constellation itself when 1 is one of
    its points (DBPSK is BPSK)."""
    from .. import symbol
    c = _constellation(constellation)
    out = {}
    for stem in symbol.ls().get("dpsk", []):
        d = symbol.load(stem)
        if d.order == c.order:
            out[stem] = d
    if np.min(np.abs(c.points - 1)) < 1e-6:
        out[c.name] = c
    return out

def bit_streams(symbols, constellation, differential=True, keys=None):
    """Hard bits of symbols under every map a receiver that doesn't know the carrier phase or the IQ sense has to try

    ("abs", i): the constellation's map i (its `maps`: every rotation, with and without spectral inversion)
    ("diff", name, i): phase changes y[n]/y[n-1] demapped with map i of a differential alphabet (`diff_alphabets`), only its
        unrotated maps (a phase change doesn't see the carrier phase, only the IQ sense)
    keys: only these. Returns {key: bits}
    """
    c = _constellation(constellation)
    y = np.asarray(symbols, dtype=np.complex128)
    out = {}
    for i, m in enumerate(c.maps):
        if keys is None or ("abs", i) in keys:
            out[("abs", i)] = _hard(y, c.points, m.arr)
    if differential and len(y) > 1:
        r = y[1:]/np.where(np.abs(y[:-1]) < 1e-12, 1.0, y[:-1])
        r = r/np.maximum(np.abs(r), 1e-12)
        for name, d in diff_alphabets(c).items():
            for i, m in enumerate(d.maps):
                if abs(np.angle(np.exp(1j*m.rotation))) > 1e-6:
                    continue
                if keys is None or ("diff", name, i) in keys:
                    out[("diff", name, i)] = _hard(r, d.points, m.arr)
    return out

# --- Convolutional ---------------------------------------------------------------------------------

def _polys(K):
    """Generators of constraint length K: first and last tap set (otherwise it is a shorter code)"""
    return [np.array((1,) + m + (1,), dtype=np.uint8) for m in itertools.product((0, 1), repeat=K-2)]

def _convolve_all(c, polys, K):
    """c * g for every generator, kept at length len(c)-K+1 (the part with full history)"""
    n = len(c)
    out = np.zeros((len(polys), n-K+1), dtype=np.uint8)
    for k in range(K):
        shifted = c[K-1-k:n-k]
        for i, g in enumerate(polys):
            if g[k]:
                out[i] ^= shifted
    return out

def _fold_z(s, T):
    """Per row of s (R, L): how non-uniform the relation is over the phase of a period-T pattern (chi^2 as z)"""
    m = s.shape[1]//T*T
    if m < 4*T:
        return np.zeros(len(s))
    mp = s[:, :m].reshape(len(s), -1, T).mean(axis=1)
    npp = m//T
    chi = np.sum(4*(mp-0.5)**2*npp, axis=1)
    return (chi-T)/np.sqrt(2*T)

def _period(s, z_min=7.0):
    """Strongest periodic structure in a 0/1 relation sequence: (period, z) or None.

    A convolutional relation holds uniformly over time. A block code's holds with a pattern that repeats with the block.
    """
    x = s.astype(np.float64) - s.mean()
    spec = np.abs(np.fft.rfft(x))
    spec[:3] = 0
    k = int(np.argmax(spec))
    z = float(spec[k]/np.sqrt(np.mean(spec[3:]**2) + 1e-30))/1.5
    return (len(s)/k, z) if z > z_min else None

def scan_conv(bits, ks=range(3, 8), rates=(2, 3), periods=(), tag=None):
    """Rate 1/n convolutional code scan on one bit stream. Returns (hypotheses, number of trials)"""
    hyps, trials = [], 0
    bits = np.asarray(bits, dtype=np.uint8)
    for n in rates:
        offsets = range(n) if n == 2 else (0,)
        for off in offsets:
            b = bits[off:]
            m = len(b)//n
            streams = [b[i:m*n:n] for i in range(n)]
            for K in ks:
                polys = _polys(K)
                conv = [_convolve_all(c, polys, K) for c in streams]
                for a, bb in itertools.combinations(range(n), 2):
                    for ia, ga in enumerate(polys):
                        # relation (c_a * g_b) ^ (c_b * g_a) for every g_b at once
                        s = conv[a] ^ conv[bb][ia]
                        k = s.sum(axis=1, dtype=np.int64)
                        # polys[ib] is g_b: relation uses c_a*g_b with c_b*g_a: here conv[a][ib], conv[bb][ia]
                        z = deviance_z(k, s.shape[1])
                        trials += len(polys)
                        j = int(np.argmax(np.abs(z)))
                        if abs(z[j]) > 3.0:
                            params = {"n": n, "K": K, "offset": off, "streams": (a, bb),
                                      "g": {a: tuple(int(x) for x in ga), bb: tuple(int(x) for x in polys[j])},
                                      "stream": tag}
                            per = _period(s[j]) if abs(z[j]) > 6 else None
                            if per is None:
                                hyps.append(Hypothesis("conv", f"rate 1/{n} K={K}", float(abs(z[j])), params))
                            else: # Holds only in places: some other linear code (block, LDPC) with short checks
                                params["period_pairs"] = per[0]
                                hyps.append(Hypothesis("linear-block", f"short checks, period {per[0]:.1f} pairs", float(abs(z[j])), params))
                        for T in periods:
                            zf = _fold_z(s, T)
                            trials += len(polys)
                            j = int(np.argmax(zf))
                            if zf[j] > 3.0:
                                hyps.append(Hypothesis("conv+scrambler", f"rate 1/{n} K={K} T={T}", float(zf[j]),
                                    {"n": n, "K": K, "offset": off, "T": T, "streams": (a, bb),
                                     "g": {a: tuple(int(x) for x in ga), bb: tuple(int(x) for x in polys[j])},
                                     "stream": tag}))
    return hyps, trials

# --- Block / LDPC ---------------------------------------------------------------------------------

def repeat_check(count):
    """Parity checks of a repetition code: all bits of the word equal"""
    H = np.zeros((count-1, count), dtype=bool)
    for i in range(count-1):
        H[i, i] = H[i, i+1] = True
    return H

def scan_block(bits, syndrome, n, name, tag=None, max_offsets=None):
    """Fraction of words satisfying every parity check, at each word alignment. Returns (hypotheses, trials)

    syndrome: words (W, n) -> parity check failures (W, r)
    """
    bits = np.asarray(bits, dtype=np.uint8)
    hyps, trials = [], 0
    r = syndrome(np.zeros((1, n), dtype=np.uint8)).shape[1]
    p0 = 2.0**-r
    for off in range(n if max_offsets is None else min(n, max_offsets)):
        w = len(bits[off:])//n
        if w < 50:
            break
        words = bits[off:off+w*n].reshape(w, n)
        zero = int((~syndrome(words).any(axis=1)).sum())
        trials += 1
        z = float(deviance_z(zero, w, p0))
        # A real code at a sane error rate passes most words; a high baseline (few checks) passes some by chance
        if z > 3.0 and zero/w >= max(0.4, 8*p0):
            hyps.append(Hypothesis("block", name, z, {"n": n, "k": n-r, "offset": off, "words_ok": zero/w, "stream": tag}))
    return hyps, trials

_BLOCKS = None

def config_blocks():
    """(name, folder, config name, n, syndrome function) for the block codes in config/fec (loaded once)"""
    global _BLOCKS
    if _BLOCKS is None:
        _BLOCKS = _load_blocks()
    return _BLOCKS

def _load_blocks():
    from .. import code as coding
    out = []
    for folder in ("block",):
        for name in coding.names(folder):
            try:
                enc = coding.load(f"{folder}/{name}").encoder
            except Exception: # A config that doesn't load can't be a hypothesis
                continue
            if hasattr(enc, "syndrome"):
                syn = enc.syndrome
            elif hasattr(enc, "check"): # linear: check is H transposed
                chk = np.asarray(enc.check, dtype=np.uint8)
                syn = lambda w, chk=chk: (w.astype(np.uint8) @ chk) % 2
            elif enc.num == 1: # repetition: all bits of the word equal
                H = repeat_check(enc.den).astype(np.uint8)
                syn = lambda w, H=H: (w.astype(np.uint8) @ H.T) % 2
            else:
                continue
            out.append((name.partition("/")[0], folder, name, enc.den, syn))
    return out

# --- Whiteness -----------------------------------------------------------------------------------

def whiteness(bits, max_lag=4096):
    """Basic facts: bias and the strongest autocorrelation (both in z). Scrambled/coded/encrypted data is flat"""
    x = 1.0 - 2.0*np.asarray(bits, dtype=np.float64)
    N = len(x)
    bias = float(abs(x.mean())*np.sqrt(N))
    F = np.fft.rfft(x - x.mean(), 2*N)
    ac = np.fft.irfft(np.abs(F)**2)[1:max_lag+1]/N/x.var()
    return {"bits": N, "bias_z": bias, "autocorr_z": float(np.abs(ac).max()*np.sqrt(N)),
            "autocorr_lag": int(np.argmax(np.abs(ac))+1), "lags": max_lag}

# --- Entry points -----------------------------------------------------------------------------------

def descramblers():
    """Self-synchronising descramblers {name: taps}: out[t] = in[t] ^ in[t-k] for each tap"""
    return {"g3ruh": (12, 17), "v34": (18, 23), "none": ()}

def detect_fec(streams, ks=range(3, 8), rates=(2, 3), periods=(), blocks=True, descramble=False, max_offsets=256):
    """Identify the FEC of unknown bit streams

    streams: bits, or {tag: bits} when the bit mapping is ambiguous (see `bit_streams`)
    Returns a Report
    """
    if not isinstance(streams, dict):
        streams = {"bits": streams}
    hyps, trials = [], 0
    facts = None
    block_codes = config_blocks() if blocks else []
    for tag, bits in streams.items():
        variants = {"": np.asarray(bits, dtype=np.uint8)}
        if descramble:
            for name, taps in descramblers().items():
                b = np.asarray(bits, dtype=np.uint8).copy()
                for k in taps:
                    b[k:] ^= np.asarray(bits, dtype=np.uint8)[:-k]
                variants[name] = b
        for dname, b in variants.items():
            t = (tag, dname) if dname else tag
            if facts is None:
                facts = whiteness(b)
            h, n = scan_conv(b, ks, rates, periods, t)
            hyps += h
            trials += n
            for name, folder, cfg, bn, syn in block_codes:
                h, n = scan_block(b, syn, bn, name, t, max_offsets)
                for hh in h:
                    hh.coding = (folder, cfg)
                hyps += h
                trials += n
    _label_configs(hyps)
    return Report(hyps, trials, facts)

def _label_configs(hyps):
    """Attach the matching config/fec entry to convolutional hypotheses"""
    from .. import code as coding
    known = {}
    for name in coding.names("conv"):
        c = coding.load(f"conv/{name}")
        g = np.asarray(c.encoder.generator, dtype=np.uint8)
        known[name] = {tuple(map(tuple, g)), tuple(map(tuple, g[::-1]))}
    for h in hyps:
        if h.kind.startswith("conv") and "g" in h.params and len(h.params["g"]) == 2:
            gs = tuple(h.params["g"][i] for i in sorted(h.params["g"]))
            for name, forms in known.items():
                if gs in forms or tuple(reversed(gs)) in forms:
                    h.coding = ("conv", name)

# --- Identifying a chain from config/fec definitions ---------------------------------------------------

def find_sync(bits, markers=(0x47, 0xB8), period=204):
    """Bit offset and byte phase at which `markers` repeat every `period` bytes. Returns (fraction, bit offset, phase)"""
    bits = np.asarray(bits, dtype=np.uint8)
    best = (0.0, 0, 0)
    for off in range(8):
        b = bits[off:]
        B = np.packbits(b[:len(b)//8*8])
        n = len(B)//period*period
        if n < 4*period:
            continue
        hit = np.isin(B[:n], markers).reshape(-1, period).mean(axis=0)
        p = int(np.argmax(hit))
        if hit[p] > best[0]:
            best = (float(hit[p]), off, p)
    return best

def scan_defined_conv(streams, names=None, steps=2500, skip=0.1, segments=(0.1, 0.45, 0.8)):
    """Try every convolutional definition in config/fec/conv on every bit mapping and puncture alignment

    A correct hypothesis lets the decoder reproduce the received bits (the channel error rate, well below
    what a wrong one gets). Each of `segments` (fractions into the stream) is tried on its own: a symbol the timing slipped
    moves the puncture phase, so a segment across it fits two phases half well, and the others still fit one. A
    hypothesis says which segment (its offset holds there). Returns (hypotheses, trials).
    """
    from .. import code as coding
    names = names or coding.names("conv")
    hyps, trials = [], 0
    rng = np.random.default_rng(0)
    for name in names:
        code = coding.load(f"conv/{name}")
        enc = code.encoder
        pat = enc.puncture if enc.puncture is not None else np.ones((enc.den, 1), dtype=bool)
        L = pat.shape[1]
        nb = int(pat.sum())
        per = max(steps//L, 1)
        rand = rng.integers(0, 2, per*nb).astype(np.uint8)
        k0, n0 = mismatch(code, rand)
        base = max(k0/n0, 1e-4) # What a wrong hypothesis gets
        for tag, bits in streams.items():
            starts = sorted({int(len(bits)*f) for f in segments if int(len(bits)*f) + nb + per*nb <= len(bits)})
            if not starts and len(bits) >= int(len(bits)*skip) + nb + per*nb:
                starts = [int(len(bits)*skip)]
            for si, start in enumerate(starts):
                for off in range(nb):
                    seg = np.asarray(bits[start+off:start+off+per*nb], dtype=np.uint8)
                    if len(seg) < per*nb:
                        continue
                    k, n = mismatch(code, seg)
                    trials += 1
                    z = float(-deviance_z(k, n, base)) # Fewer mismatches than a wrong code gives
                    if z > 3.0 and k/n < 0.5*base: # And clearly fewer, not a marginal difference on a short code
                        hyps.append(Hypothesis("conv-config", name, z, # offset: from the start of the stream, not of the segment
                            {"offset": (start + off) % nb, "mismatch": k/n, "baseline": base, "stream": tag, "segment": si},
                            ("conv", name)))
    return hyps, trials

def decode_conv(bits, hyp):
    """Run the config definition behind a conv-config hypothesis on a whole bit stream"""
    from .. import code as coding
    code = coding.load("/".join(hyp.coding))
    seg = np.asarray(bits[hyp.params["offset"]:], dtype=np.uint8)
    nb = puncture_period(code)
    seg = seg[:len(seg)//nb*nb]
    return code.decoder.decode(seg)

RS_WORDS = 12 # Words a Reed-Solomon hypothesis is scored on (each passes by chance 256^-nroots of the time)
_RS_PROBE = 8 # Words tried first at every phase: a random word has zero syndromes 256^-nroots of the time

def _rs_zero(code, seg, n, phases, words):
    """Zero-syndrome words among the last `words` whole codewords at each phase of seg: (len(phases),) counts

    The last ones: the start of a window is where the sync before it was still settling"""
    first = (len(seg) - max(phases) - 1)//n - words # Codewords from each phase that fit, less those used
    rows = np.stack([seg[p + first*n:p + (first + words)*n].reshape(words, n) for p in phases])
    S = code.syndromes(rows.reshape(-1, n)).reshape(len(phases), words, -1)
    flat = (rows == rows[:, :, :1]).all(axis=2) # One value repeated (idle, a dead carrier) is a codeword of any RS code
    return (~S.any(axis=2) & ~flat).sum(axis=1)

def decode_tail(bits, code, fraction=0.5, probe=4000):
    """Viterbi output of the later part of a stream, at the puncture phase that holds there

    Inner layers are looked for in the later part of a window, where the sync before it has settled; a symbol the timing
    slipped on the way moves the puncture phase, so it is found again at the start of that part."""
    bits = np.asarray(bits, dtype=np.uint8)
    nb = puncture_period(code)
    start = int(len(bits)*(1 - fraction))
    seg = bits[start:start + probe//nb*nb + nb]
    best = min(range(nb), key=lambda o: (lambda k, n: k/max(n, 1))(*mismatch(code, seg[o:o + probe//nb*nb])))
    tail = bits[start + best:]
    return code.decoder.decode(tail[:len(tail)//nb*nb]).astype(np.uint8)

def scan_rs(bits, names=None, interleavers=True):
    """Find a Reed-Solomon code (config/fec/block type rs), optionally behind a Forney interleaver (config/interleave/forney)

    Every polarity, bit offset and codeword phase is probed on a few words; the phases where any word has zero syndromes
    are scored on RS_WORDS words. Both are the last words of the stream (its start is where the sync was settling). Behind an interleaver the deinterleaver runs once per commutator phase, and codewords start
    on its branch 0 (as the interleaver's definition has them).
    Returns (hypotheses, trials, short): short is True when some combination had too few bits to be tried.
    params: polarity, bit_offset, byte_phase (where a codeword starts in the bytes as given, before deinterleaving),
        interleaver (its config name or None), words_ok
    """
    from .. import code as coding
    names = names or coding.names("block", "rs")
    inters = [(None, None)]
    if interleavers:
        for nm in coding.names("interleave"):
            enc = coding.load(f"interleave/{nm}").encoder
            if hasattr(enc, "I") and hasattr(enc, "M") and getattr(enc, "symbol", 8) == 8:
                inters.append((nm, enc))
    hyps, trials, short = [], 0, False
    bits = np.asarray(bits, dtype=np.uint8)
    for name in names:
        code = coding.load(f"block/{name}").encoder
        if getattr(code, "m", 8) != 8: # The search runs on bytes: codes over other fields (P25's GF(64)) don't apply
            continue
        n = code.den
        p0 = 256.0**-code.nroots
        for pol in (0, 1):
            b = bits ^ pol
            for off in range(8):
                B = np.packbits(b[off:][:(len(b)-off)//8*8]).astype(np.int64)
                for iname, inter in inters:
                    if inter is None:
                        groups = [(0, B, range(n))]
                    else: # Output byte t of a deinterleaver started at B[c] was sent (branch 0, no delay) at B[c + t - delay]
                        start = inter.delay
                        step = inter.I if n % inter.I == 0 else 1
                        groups = [(c, inter.decode(B[c:], symbols=True)[start:], range(0, n, step)) for c in range(inter.I)]
                    for c, seg, phases in groups:
                        phases = list(phases)
                        if len(seg) < (RS_WORDS + 1)*n:
                            short = True
                            continue
                        trials += len(phases)
                        probe = _rs_zero(code, seg, n, phases, _RS_PROBE)
                        for ph in [p for p, k in zip(phases, probe) if k >= 2]:
                            zero = int(_rs_zero(code, seg, n, [ph], RS_WORDS)[0])
                            z = float(deviance_z(zero, RS_WORDS, p0))
                            # seg[ph] was sent at B[c + ph]
                            hyps.append(Hypothesis("rs", name, z, {"polarity": pol, "bit_offset": off,
                                "byte_phase": (c + ph) % n, "interleaver": iname, "words_ok": zero/RS_WORDS}, ("block", name)))
    return hyps, trials, short

# --- Guessing an unknown code --------------------------------------------------------------------------

def frame_length(bits, min_n=16, top=5):
    """Candidate frame/block lengths from autocorrelation peaks (sync markers, repeated headers)

    Only sees fixed patterns that repeat every n bits; a code with no marker has none. Needs >= ~8 frames.
    Returns [(n, z)] strongest first, after removing multiples of a stronger shorter peak.
    """
    x = 1.0 - 2.0*np.asarray(bits, dtype=np.float64)
    N = len(x)
    x -= x.mean()
    F = np.fft.rfft(x, 2*N)
    ac = np.fft.irfft(np.abs(F)**2)[:N//4]/N/x.var()
    lags = np.arange(len(ac))
    z = np.abs(ac)*np.sqrt(N - lags)
    z[:min_n] = 0
    crit = z_critical(len(z))
    cand = []
    for i in np.argsort(-z)[:200]:
        if z[i] < crit:
            break
        cand.append((int(i), float(z[i])))
    cand.sort(key=lambda c: c[0])
    out = []
    for n, zz in cand:
        if not any(n % m == 0 for m, _ in out):
            out.append((n, zz))
    out.sort(key=lambda c: -c[1])
    return out[:top]

def recover_dual(bits, n, offset=0, trials=40, seed=0, z_min=8.0):
    """Recover parity checks of an unknown linear block code of length n from (noisy) hard bits

    Words that are codewords satisfy every parity check h. For a random set of words, the null space of
    their matrix contains the checks that no error among them broke, and anything that is a check satisfies
    the other words too: so null vectors are verified against all words, and those that stand out
    from 50% are kept. Needs about n words and a word error rate that leaves some subsets of words clean.

    Returns dict(H, k, words, needed, ok). H is a basis of the checks (dense, not sparse).
    """
    rng = np.random.default_rng(seed)
    bits = np.asarray(bits, dtype=np.uint8)
    m = (len(bits) - offset)//n
    words = bits[offset:offset+m*n].reshape(m, n)
    out = {"H": np.zeros((0, n), dtype=np.uint8), "k": n, "words": m, "needed": 2*n, "ok": False}
    if m < n:
        return out
    keep = np.zeros((0, n), dtype=np.uint8)
    rank, stale = 0, 0
    for _ in range(trials):
        best_rank = rank
        for s in (n//4, n//2, (3*n)//4, n-1):
            idx = rng.choice(m, size=min(s, m), replace=False)
            N = nullspace(words[idx])
            if len(N) == 0:
                continue
            sat = 1.0 - ((words.astype(np.uint8) @ N.T.astype(np.uint8)) % 2).mean(axis=0)
            z = (sat - 0.5)*2*np.sqrt(m)
            good = N[z > z_min]
            if len(good):
                cat = np.concatenate((keep, good))
                B = nullspace(nullspace(cat)) # Row space of cat: its dual's dual
                keep = B
                rank = len(keep)
            best_rank = max(best_rank, rank)
        stale = stale + 1 if rank == best_rank and best_rank == out["H"].shape[0] else 0
        out["H"] = keep
        if stale >= 3:
            break
    out["H"] = keep
    out["k"] = n - len(keep)
    out["ok"] = len(keep) > 0
    out["needed"] = max(2*n, 2*out["k"] + 64)
    return out

class Guess:
    """A guessed definition (a code, a scrambler). Decoding only happens from a saved config: `save` it, then load it
    (rp.code.load("block/guess/<name>"), rp.code.scram.load("feedthrough/guess/<name>"))"""
    __slots__ = ("kind", "definition", "evidence")
    FOLDER = {"block": ("fec", "block"), "rs": ("fec", "block"), "conv": ("fec", "conv"),
              "scramble": ("scramble", "feedthrough")} # kind: (config domain, folder in it)

    def __init__(self, kind, definition, evidence=None):
        self.kind = kind
        self.definition = definition
        self.evidence = evidence or {}

    def __repr__(self):
        d = self.definition
        if self.kind == "block":
            if d.get("type") == "product":
                return f"<Guess product ({d['rows']['n']},{d['rows']['k']}) x ({d['cols']['n']},{d['cols']['k']}) {self.evidence}>"
            if "checks" in d:
                return f"<Guess {d.get('type', 'block')} n={d['n']} k={d['k']} {len(d['checks'])} checks {self.evidence}>"
            extra = {k: v for k, v in d.items() if k not in ("base", "type", "n", "k", "comment")}
            return f"<Guess {d['type']} ({d['n']},{d['k']}) {extra} {self.evidence}>"
        if self.kind == "rs":
            return f"<Guess RS({d['n']},{d['k']}) poly {d['poly']:#x} fcr {d['fcr']} prim {d['prim']} {self.evidence}>"
        if self.kind == "scramble":
            return f"<Guess feedthrough scrambler taps {d['poly']} {self.evidence}>"
        e = d["encode"]
        if e.get("puncture") is not None:
            pat = np.asarray(e["puncture"])
            rate = f"{e['num']*pat.shape[1]}/{int(pat.sum())} (punctured {pat.tolist()})"
        else:
            rate = f"{e['num']}/{e['den']}"
        return f"<Guess conv K={e['constraint']} rate {rate} {self.evidence}>"

    def save(self, name, path=None):
        """Write config/<domain>/<folder>/guess/<name>.json (or <path>/<folder>/guess/<name>.json). Returns the file"""
        import json
        from pathlib import Path
        from .. import _config
        domain, sub = self.FOLDER[self.kind]
        base = Path(path) if path else _config.find_config(domain)
        folder = base / sub / "guess"
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / f"{name.lower()}.json"
        f.write_text(json.dumps({**self.definition, "guess": self.evidence}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else list(o)))
        return f

def guess_conv(hyp):
    """Definition from a conv hypothesis (generators found by the parity relation, plain or with a scrambler)"""
    p = hyp.params
    gens = [list(p["g"][i]) for i in sorted(p["g"])]
    n = p["n"]
    if len(gens) < n: # Pair relations only name two of the generators
        return None
    enc = {"type": "conv", "num": 1, "den": n, "constraint": p["K"], "generator": gens}
    dec = {**enc, "type": "viterbi"}
    return Guess("conv", {"base": "conv", "type": "split", "encode": enc, "decode": dec},
                 {"z": hyp.z, "offset": p["offset"]})

def guess_block(bits, n=None, offsets=None, hints=()):
    """Find a block code's length, alignment and parity checks. Returns a Guess or None

    n: block length if known, else taken from `hints` and the autocorrelation (frame_length)
    The H it finds is a dense basis of the dual code; message positions come from elimination. Large n
    (thousands of bits) needs ~n words and error-free-ish data: see recover_dual's `needed`.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    cands = [n] if n else list(dict.fromkeys(list(hints) + [c for c, _ in frame_length(bits)]))
    for c in cands:
        offs = list(offsets) if offsets is not None else list(range(c if c <= 512 else 1))
        # Wrong alignments still verify the checks that fall inside one codeword: the right one finds the most
        scores = []
        for off in offs:
            r = recover_dual(bits, c, off, trials=3)
            scores.append((len(r["H"]), off))
        if not scores or max(scores)[0] == 0:
            continue
        off = max(scores)[1]
        r = recover_dual(bits, c, off, trials=40)
        H = r["H"].astype(bool)
        try:
            _, info = systematic(H)
            sysp = sorted(int(i) for i in info)
        except Exception:
            sysp = None
        checks = [list(map(int, np.nonzero(row)[0])) for row in H]
        d = {"base": "block", "type": "ldpc", "n": int(c), "k": int(r["k"]), "checks": checks}
        if sysp is not None:
            d["systematic"] = sysp
        return Guess("block", d, {"offset": int(off), "words": r["words"], "needed_words": r["needed"]})
    return None

def scan_hdlc(streams, names=None, min_frames=3):
    """HDLC framing (config/frame): many flags, and frames whose FCS checks out

    Looks at flag counts first (a 0x7E pattern shows up at random once per 256 bits), then deframes the
    streams that have far more than that. Returns (hypotheses, trials).
    """
    from .. import link as encap
    names = names or encap.ls().get("frame", [])
    hyps, trials = [], 0
    for name in names:
        h = encap.load(f"frame/{name}").encoder
        f = h.flag_bits
        for tag, bits in streams.items():
            for pol in (0, 1):
                b = np.asarray(bits, dtype=np.uint8) ^ pol
                work = b
                if h.nrzi:
                    work = np.concatenate(([0], (b[1:] == b[:-1]).astype(np.uint8)))
                win = np.lib.stride_tricks.sliding_window_view(work, len(f))
                nflag = int((win == f).all(axis=1).sum())
                trials += 1
                if nflag < 6:
                    continue
                frames, aborts = h.deframe(b[:200000])
                good = [p for p, ok in frames if ok]
                if len(good) >= min_frames:
                    z = float(deviance_z(len(good), max(len(frames), 1), 2.0**-(8*h.fcs.width//8)))
                    hyps.append(Hypothesis("hdlc", name, z, {"frames": len(good), "bad": len(frames)-len(good), "polarity": pol,
                        "stream": tag}, ("frame", name)))
    return hyps, trials

# --- One answer: which code a symbol stream carries ------------------------------------------------------

CONV_SYMBOLS = 2**15 # Symbols the convolutional and block scans look at (the inner layers use all there are)
MARKER_PERIODS = 64  # Codewords enough to find a pattern of up to 16 markers (frame_markers: three of each, past the first half)

def frame_markers(bits, period, offset, column, unit=8, threshold=0.9, lengths=(1, 2, 4, 8, 16)):
    """The values that start every `period` units at `column` (units after bit `offset`): a pattern for `sync.Align`

    The shortest repeating pattern each of whose positions holds one value at least `threshold` of the time (DVB-S:
    0xB8 then 0x47 seven times), over the later half of the periods. Returns (pattern, worst fraction), or (None, 0.0)."""
    b = np.asarray(bits, dtype=np.uint8)[offset:]
    span = period*unit
    P = len(b)//span
    if P < 8:
        return None, 0.0
    vals = b[:P*span].reshape(P, period, unit)[:, column].astype(np.int64) @ (1 << np.arange(unit-1, -1, -1))
    if P >= 32: # The later half (the start of a window is where the sync before it was settling), but enough periods
        vals = vals[max(0, min(P//2, P - 3*max(lengths))):] # left for the longest pattern (a low code rate brings fewer)
    P = len(vals)
    for L in lengths:
        if P < 3*L: # At least three periods at every position of the pattern
            break
        pat, frac = [], 1.0
        for r in range(L):
            cnt = np.bincount(vals[r::L], minlength=1 << unit)
            pat.append(int(np.argmax(cnt)))
            frac = min(frac, cnt.max()/len(vals[r::L]))
        if frac >= threshold:
            return pat, float(frac)
    return None, 0.0

def _transform(points, src, dst, rotate=True):
    """(angle, conj) that moves symbols decoded right by labels `src` to where labels `dst` decode them right, or None

    A received symbol near point p carries src[p]; after y -> conj?(y)*e^(j angle) it is near q[p], which dst labels
    dst[q[p]]: the transform where those agree for every point. rotate False: inversion only (phase changes)."""
    from ..symbol.mapping import symmetries
    src, dst = np.asarray(src), np.asarray(dst)
    for th, flip, q in symmetries(points):
        if not rotate and abs(np.angle(np.exp(1j*th))) > 1e-6:
            continue
        if np.array_equal(dst[q], src):
            return float(th), bool(flip)
    return None

def _fec_steps(steps):
    """The code steps that are codes: fec (type, name), and a line code or a sync marker, which a find never has (ax25/afsk1200
    is HDLC behind NRZI, not the HDLC a find names). Scramblers and alignment are not compared: a find leaves those to the profile"""
    return [(s["type"], s["name"]) if s["code"] == "fec" else (s["code"], s.get("name"))
            for s in steps if s.get("code") in ("fec", "line", "marker")]

def _symbol_steps(steps):
    """(differential?, constellation name) of a profile's symbol steps"""
    from .. import symbol
    diff = any(s.get("symbol") == "differential" for s in steps)
    cons = [s for s in steps if s.get("symbol") == "constellation"]
    from ..profile.steps import constellation
    return diff, [constellation(s).name for s in cons], cons

@functools.lru_cache(maxsize=None)
def _profile_codes():
    """The names of the convolutional codes shipped profiles decode with"""
    from .. import _config
    out = set()
    for fam, modes in _config.ls("profile").items():
        for mode in modes:
            try:
                obj = _config.read(_config.find("profile", f"{fam}/{mode}"))
            except Exception:
                continue
            out |= {st["name"] for st in obj.get("steps", []) if st.get("code") == "fec" and st.get("type") == "conv"
                    and isinstance(st.get("name"), str)}
    return frozenset(out)

def match_profile(steps):
    """A shipped profile (config/profile) with the same code and symbol steps as `steps`: (name, its steps) or (None, None)"""
    from .. import _config
    want = (_fec_steps(steps), _symbol_steps(steps)[:2])
    for fam, modes in _config.ls("profile").items():
        for mode in modes:
            name = f"{fam}/{mode}"
            try:
                obj = _config.resolve(_config.read(_config.find("profile", name)), {}, f"profile/{name}")
                got = (_fec_steps(obj["steps"]), _symbol_steps(obj["steps"])[:2])
            except Exception: # A profile whose steps a header names (no fixed code) can't match
                continue
            if got == want:
                return name, obj["steps"]
    return None, None

class CodeDetection:
    """Result of `identify`: the code a symbol stream carries, and how to decode it. Stored in stream.est["fec"]

    found: a code was identified (complete: and nothing more would come of more symbols)
    kind, name: the outer code ("conv", "block", "hdlc") and its config name; inner: hypotheses found inside it (RS: in
        `steps` when the bytes that start its codewords were found too, which Align needs)
    stream: the bit stream that decodes (`bit_streams` key); equivalent: others that decode as well (a code that can't
        tell inverted bits apart leaves that to what's inside it: Align's polarity)
    offset, period: where in the window the code's structure starts (bits) and how often it repeats
    steps: profile steps from the timed symbols on; profile_name: the shipped profile that has the same ones
    target: the map the decoder uses; correction: (angle, conj) that turns the symbols to it (rp.sync.AcquireFEC applies it)
    mismatch, baseline: re-encoding mismatch of a convolutional code, and what a wrong hypothesis gets
    slips: carrier turns and symbol slips rp.sync.AcquireFEC caught and put right; latency: seconds it held the symbols for
    report: the ranked hypotheses of the tier that decided (none did: of every tier)
    guesses: with blind, the `Guess`es for what no definition covers (a code, an RS code or a scrambler inside a convolutional
        code, see `solve`), outermost first; they are in `inner` but never in `steps`: save one to decode with it
    """
    __slots__ = ("found", "complete", "mod", "kind", "name", "stream", "equivalent", "offset", "period", "inner",
                 "steps", "profile_name", "target", "correction", "z", "mismatch", "baseline", "report", "guesses", "n",
                 "slips", "latency")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.found = bool(self.found)
        self.complete = bool(self.complete)
        self.inner = self.inner or []
        self.equivalent = self.equivalent or []
        self.slips = self.slips or 0
        self.guesses = self.guesses or []

    @property
    def differential(self):
        return self.stream is not None and self.stream[0] == "diff"

    def correct(self, y):
        """Symbols turned to the decoder's map"""
        if self.correction is None:
            return y
        th, flip = self.correction
        y = np.conj(y) if flip else y
        return y if th == 0 else (y*np.exp(1j*th)).astype(np.asarray(y).dtype, copy=False)

    def profile(self):
        """The profile that decodes it: the shipped one with the same steps, else one built from `steps`"""
        from .. import profile
        if not self.found:
            raise ValueError(f"no code found: {self}")
        if self.profile_name:
            return profile.load(self.profile_name)
        return profile.Profile("detected", self.definition())

    def rx(self):
        """Receive runner from the timed, carrier-corrected symbols rp.sync.AcquireFEC gives"""
        from ..base.types import Rep
        return self.profile().rx(start=Rep.SYMBOLS, applied=("timing", "carrier"))

    def definition(self):
        return {"base": "profile", "comment": f"detected by rp.analysis.fec.identify: {self}", "steps": self.steps}

    def save(self, name, path=None):
        """Write config/profile/guess/<name>.json (or <path>/guess/<name>.json). Returns the file"""
        import json
        from pathlib import Path
        from .. import _config
        base = Path(path) if path else _config.find_config("profile")
        folder = base / "guess"
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / f"{name.lower()}.json"
        f.write_text(json.dumps(self.definition(), indent=1))
        return f

    def __str__(self):
        if not self.found:
            g = f", blind guesses {self.guesses}" if self.guesses else ""
            return f"no known code ({self.n} symbols{g})"
        inner = "".join(f" + {h.params['interleaver'] + ' + ' if h.params.get('interleaver') else ''}{h.name}" for h in self.inner)
        prof = f" = profile {self.profile_name}" if self.profile_name else ""
        mm = f", mismatch {self.mismatch:.4f}" if self.mismatch is not None else ""
        return f"{self.kind} {self.name}{inner} on {self.mod} {self.stream}{prof} (z={self.z:.1f}{mm})"

    __repr__ = __str__

def _pick(hyps, z_crit, key, fits):
    """The best hypothesis that stands out, if the code fits these bits at its alignment only

    A code fits its own bits at one alignment (puncture phase, codeword start). Bits that fit at several did not come
    from it: constant bits (an unmodulated carrier) are a codeword of every linear code at every alignment, and bits
    repeated by a symbol rate locked too high fit a repetition code at all but a few. fits(h, other): other is as good"""
    good = [h for h in hyps if h.z > z_crit]
    if not good:
        return None
    best = min(good, key=lambda h: (key(h), -h.z))
    same = [h for h in good if h.coding == best.coding and h.params["stream"] == best.params["stream"]
            and h.params.get("segment") == best.params.get("segment") # (a slip between segments moves the offset)
            and h.params["offset"] != best.params["offset"] and fits(best, h)]
    return None if same else best

def identify(symbols, constellation, names=None, blind=False, differential=True, profiles=True):
    """Which code from config/fec timed, carrier-corrected symbols carry, under which map, and how to decode it

    In turn, stopping at the first that stands out: configured convolutional codes (any puncture phase), with a Reed-Solomon
    code behind them (scan_rs; then the markers that start its codewords, for Align); HDLC framing; configured block codes.
    blind: look for what no definition covers, as `Guess`es only (nothing decodes from a guess). With no configured code: a
        known mother code with an unknown puncture pattern (rpps.analysis.puncture) and what is inside it, a binary block code
        on the channel bits (rpps.analysis.block), then unknown rate 1/n convolutional codes (detect_fec). Inside a
        convolutional code with no configured RS behind it: `solve` (RS, block codes, scramblers). Tens of seconds.
    symbols: the symbols (any gain); constellation: a Constellation or its name (the detected modulation)
    names: only these convolutional definitions. Returns a CodeDetection
    """
    c = _constellation(constellation)
    y = np.asarray(symbols, dtype=np.complex128)
    if len(y):
        y = y/np.sqrt(np.mean(np.abs(y)**2))
    out = CodeDetection(mod=c.name, n=len(y))
    if len(y) < 1024:
        return out
    head = bit_streams(y[:CONV_SYMBOLS], c, differential)
    pre = [{"sync": "timing"}]

    def symbol_steps(key):
        if key[0] == "diff":
            d = diff_alphabets(c)[key[1]]
            return [{"symbol": "differential"}, {"symbol": "constellation", "name": key[1], "mapping": d.maps[key[2]].comment}]
        from .. import _config
        return [{"symbol": "constellation", **({"name": c.name} if _config.find("constellation", c.name) else {"shape": c.name})}]

    tried = [] # Every tier's hypotheses: the report when nothing stands out
    hyps, trials = scan_defined_conv(head, names)
    rep = Report(hyps, trials, None)
    tried.append(rep)
    conv = _pick(rep.hyps, rep.z_crit, lambda h: h.params["mismatch"]/h.params["baseline"],
                 lambda b, h: h.params["mismatch"] < 0.5*(b.params["mismatch"] + b.params["baseline"]))
    if conv is not None and profiles:
        # Codes that are one another under a different map (a rate 1/2 code with its generators swapped is the same code
        # with I and Q swapped) fit as well, and which one wins is chance: prefer the one a shipped profile decodes with
        shipped = _profile_codes()
        if conv.coding[1] not in shipped:
            alike = [h for h in rep.hyps if h.coding[1] in shipped and h.z > rep.z_crit
                     and h.params["mismatch"] < 0.5*(conv.params["mismatch"] + conv.params["baseline"])]
            if alike:
                conv = min(alike, key=lambda h: h.params["mismatch"]/h.params["baseline"])
    if conv is not None:
        key = conv.params["stream"]
        same = [h for h in rep.hyps if h.coding == conv.coding and h.params["stream"] != key and h.z > rep.z_crit
                and h.params["mismatch"] < 0.5*(conv.params["mismatch"] + conv.params["baseline"])]
        same = list({str(h.params["stream"]): h for h in same}.values())
        out = CodeDetection(found=True, mod=c.name, kind="conv", name=conv.coding[1], stream=key, offset=conv.params["offset"],
                            period=puncture_period(cached("/".join(conv.coding))),
                            equivalent=[h.params["stream"] for h in same], z=conv.z, mismatch=conv.params["mismatch"],
                            baseline=conv.params["baseline"], report=rep, n=len(y))
        steps = pre + symbol_steps(key) + [{"sync": "puncture", "fec": out.name},
                                           {"code": "fec", "type": "conv", "name": out.name}]
        info = decode_tail(bit_streams(y, c, differential, keys=[key])[key], cached("/".join(conv.coding)))
        h, t, short = scan_rs(info)
        rs_rep = Report(h, t, None)
        rs = rs_rep.verdict
        out.complete = rs is not None or not short # More symbols could still show a Reed-Solomon code inside
        if rs is None and blind:
            _attach(out, solve(info))
        if rs is not None:
            out.inner.append(rs)
            p = rs.params
            n = cached(f"block/{rs.name}").encoder.den
            pat, _ = frame_markers(info ^ p["polarity"], n, p["bit_offset"], p["byte_phase"])
            if pat is None and len(info)//(n*8) < MARKER_PERIODS:
                out.complete = False # Too few codewords to see the markers that start them: more symbols may show them
            if pat is not None:
                p["pattern"] = pat
                steps += [{"sync": "align", "period": n, "pattern": pat, "name": "packets"}]
                if p["interleaver"]:
                    steps += [{"code": "fec", "type": "interleave", "name": p["interleaver"]}]
                steps += [{"code": "fec", "type": "block", "name": rs.name}]
        out.steps = steps
        return _finish(out, c, profiles)

    hyps, trials = scan_hdlc(head)
    hyps = [h for h in hyps if h.params["polarity"] == 0] # Another map gives the other polarity: the profile has none
    rep = Report(hyps, trials, None)
    tried.append(rep)
    if rep.verdict is not None:
        v = rep.verdict
        out = CodeDetection(found=True, complete=True, mod=c.name, kind="hdlc", name=v.coding[1], stream=v.params["stream"],
                            offset=0, period=None, z=v.z, report=rep, n=len(y))
        out.steps = pre + symbol_steps(out.stream) + [{"code": "uncoded"}, {"link": "frame", "name": out.name}]
        return _finish(out, c, profiles)

    hyps, trials = [], 0
    blocks = config_blocks()
    for key, bits in head.items():
        for name, folder, cfg, bn, syn in blocks:
            h, t = scan_block(bits, syn, bn, name, key, max_offsets=bn)
            for hh in h:
                hh.coding = (folder, cfg)
            hyps += h
            trials += t
    rep = Report(hyps, trials, None)
    v = _pick(rep.hyps, rep.z_crit, lambda h: (-h.params["words_ok"], -h.params["n"]),
              lambda b, h: h.params["words_ok"] >= 0.5*b.params["words_ok"])
    if v is not None:
        out = CodeDetection(found=True, complete=True, mod=c.name, kind="block", name=v.coding[1], stream=v.params["stream"],
                            offset=v.params["offset"], period=v.params["n"], z=v.z, report=rep, n=len(y))
        out.steps = pre + symbol_steps(out.stream) + [{"code": "fec", "type": "block", "name": out.name}]
        if blind: # Its words may be the rows of a product code (a turbo product code): columns of them that are codewords too
            from . import block as blind_block
            bits, n1 = head[out.stream], out.period
            p = blind_block.product(bits, n1, out.offset)
            found = None
            if p is not None:
                rows = recover_dual(bits, n1, out.offset, trials=20)
                found = blind_block._product(bits, n1, rows["k"], rows["H"], p[0], out.offset + p[1]*n1)
            if found is not None:
                _attach(out, [found])
        return _finish(out, c, profiles)

    tried.append(rep)
    out.report = Report([h for r in tried for h in r.hyps], sum(r.trials for r in tried), None)
    out.complete = len(y) >= CONV_SYMBOLS # The scans above look no further: more symbols would not change this
    if blind:
        found = _blind(head, c, y[:CONV_SYMBOLS])
        if found:
            _attach(out, found)
            return out
        sub = {k: v for k, v in head.items() if k[0] == "abs"}
        brep = detect_fec(sub, max_offsets=32)
        out.report = brep
        v = brep.verdict
        if v is not None:
            g = None
            if v.kind.startswith("conv"):
                g = guess_conv(v)
            elif v.kind == "linear-block":
                g = guess_block(sub[v.params["stream"]], hints=[int(round(2*v.params["period_pairs"]))])
            out.guesses = [g] if g is not None else []
    return out

def _attach(out, found):
    """Add [(hypothesis, Guess or None)] to a detection: each hypothesis as an inner layer, carrying its guess"""
    for h, g in found:
        h.guess = g
        out.inner.append(h)
        if g is not None:
            out.guesses.append(g)

def solve(bits, rs_lengths=None, max_taps=2, exhaustive=False, block=True):
    """Peel unknown layers off bits (a decoder's output): [(hypothesis, Guess or None)], outermost first

    1. Reed-Solomon on the bits as they are: a configured one (scan_rs, a Forney interleaver too), else a blind one
       (rpps.analysis.rs.search over `rs_lengths`, or every length with `exhaustive`)
    2. a binary block code (rpps.analysis.block.solve_block: BCH / cyclic, product, LDPC or any linear code, by rank)
    3. a feedthrough scrambler (rpps.analysis.scrambler.search, up to `max_taps` taps); if one is found, the same codes on
       what it gives, then HDLC framing. A scrambler hypothesis with no taps means the bits already had structure.
    4. none found (descrambled data that is random itself, compressed or encrypted, stays white): Reed-Solomon behind each
       configured feedthrough scrambler (its structure is algebraic: a scrambler search can't see it)
    Nothing here decodes: a Guess is a definition once it is saved. Empty: nothing found (the bits stay white: another
    layer, or random payload such as encrypted data)."""
    from . import rs as blind_rs, scrambler, block as blind_block

    def code_layer(b, with_block=block):
        h, t, _ = scan_rs(b)
        v = Report(h, t, None).verdict
        if v is not None:
            return [(v, None)]
        h, t, g = blind_rs.search(b, lengths=rs_lengths, exhaustive=exhaustive)
        if h:
            return [(h[0], g)]
        return blind_block.solve_block(b) if with_block else []

    bits = np.asarray(bits, dtype=np.uint8)
    found = code_layer(bits)
    if found:
        return found
    # The whole stream an LFSR sequence (a PRBS, or scramblers over repeating data): that explains every bit, and a
    # feedthrough search would only find a part of it
    ha, ta, ga = scrambler.additive_search(bits)
    if ha and ha[0].params["holds"] > 0.95:
        return [(ha[0], ga)]
    h, t, g = scrambler.search(bits, max_taps=max_taps)
    if not h and ha: # Not a feedthrough scrambler: an additive one showing through constant stretches
        return [(ha[0], ga)]
    if not h: # Bits that stay white: maybe RS-coded random payload behind a known scrambler (RS structure is algebraic)
        from .. import code
        for name in code.scram.ls().get("feedthrough", []):
            obj = cached_scrambler(f"feedthrough/{name}")
            found = code_layer(scrambler.descramble(bits, obj["poly"]), with_block=False)
            if found:
                return [(Hypothesis("scrambler", f"feedthrough/{name}", found[0][0].z, {"taps": list(obj["poly"]), "config": name}),
                         None)] + found
        return []
    out = [(h[0], g)]
    taps = h[0].params["taps"]
    d = scrambler.descramble(bits, taps) if taps else bits
    if taps:
        found = code_layer(d)
        if found:
            return out + found
    hd = Report(*scan_hdlc({"bits": d}), None).verdict
    if hd is not None:
        out.append((hd, None))
    return out

def _distinct(head, c):
    """The coherent bit streams that differ by more than inverted bits: maps whose turn is below 180 degrees"""
    keys = [k for k in head if k[0] == "abs" and np.angle(np.exp(1j*c.maps[k[1]].rotation)) >= -1e-9
            and np.angle(np.exp(1j*c.maps[k[1]].rotation)) < np.pi - 1e-9]
    return {k: head[k] for k in keys} or {k: v for k, v in head.items() if k[0] == "abs"}

def _blind(head, c, y=None):
    """With no configured code: a known mother code with a puncture pattern none of them has (then what is inside it), or
    a binary block code, on the channel bits (y: the symbols, for soft values). [(hypothesis, Guess or None)], outermost
    first, or []"""
    from . import puncture, block as blind_block
    hyps, trials = puncture.search(head)
    v, g = puncture.best(hyps, trials)
    if v is not None:
        code = puncture._code(v.params["K"], v.params["generator"], np.asarray(v.params["puncture"], dtype=bool))
        bits = np.asarray(head[v.params["stream"]][v.params["offset"]:], dtype=np.uint8)
        m = puncture_period(code)
        info = code.decoder.decode(bits[:len(bits)//m*m]).astype(np.uint8)
        return [(v, g)] + solve(info)
    for key, bits in _distinct(head, c).items():
        llr = None
        if y is not None: # Soft values on this map: the cleanest words are the ones a sparse check search uses
            from .. import symbol
            cc = symbol.load_or_generate(c.name)
            cc.set_mapping(key[1])
            llr = cc.llr(y)[:len(bits)]
        found = blind_block.solve_block(bits, llr=llr)
        if found:
            for h, _ in found:
                h.params["stream"] = key
            return found
    return []

@functools.lru_cache(maxsize=None)
def cached_scrambler(ref):
    """A scrambler definition's JSON (its "poly": the taps), read once"""
    from .. import _config
    return _config.read(_config.find("scramble", ref))

def _finish(out, c, profiles):
    """The shipped profile with the same steps, if any, and the correction that turns the symbols to the map it decodes with"""
    own = out.steps
    name, steps = match_profile(own) if profiles else (None, None)
    for out.profile_name, out.steps in ((name, steps), (None, own)):
        if out.steps is None:
            continue
        out.target, out.correction = _target(out, c)
        if out.correction is not None:
            return out
    out.found = False # (no turn of these symbols gives the map their own steps name: can't happen)
    return out

def _target(out, c):
    """(the map the steps decode with, the correction that turns the symbols to it, or None)"""
    from .. import symbol
    spec = _symbol_steps(out.steps)[2][0]
    from ..profile.steps import constellation
    target = constellation(spec).mapping
    if out.differential: # A phase change sees no turn of the carrier, only the IQ sense
        alph = diff_alphabets(c)[out.stream[1]]
        return target, _transform(alph.points, alph.maps[out.stream[2]].arr, target.arr, rotate=False)
    return target, _transform(c.points, c.maps[out.stream[1]].arr, target.arr)
