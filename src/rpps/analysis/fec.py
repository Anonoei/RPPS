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
import itertools

import numpy as np

from ..math.gf2 import nullspace, systematic
from ..math.prob import deviance_z, z_critical

class Hypothesis:
    __slots__ = ("kind", "name", "z", "params", "coding")
    def __init__(self, kind, name, z, params=None, coding=None):
        self.kind = kind
        self.name = name
        self.z = z
        self.params = params or {}
        self.coding = coding # (folder, config name) when it is in config/coding

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

def psk_bit_streams(symbols, points, differential=True):
    """Hard bits of PSK symbols for every rotation / reflection / labelling / differential coding

    Which one is right is unknown: the carrier phase has an M-fold ambiguity, and the mapping isn't known.
    Returns {tag: bits}.
    """
    pts = np.asarray(points)
    M = len(pts)
    bps = int(np.log2(M))
    step = 2*np.pi/M
    phi0 = np.angle(pts[0]) % step
    out = {}
    gray = lambda i: i ^ (i >> 1)
    for conj in (0, 1):
        y = np.conj(symbols) if conj else symbols
        for rot in range(M):
            idx = np.round((np.angle(y) - phi0)/step).astype(int) + rot
            for diff in ((0, 1) if differential else (0,)):
                k = idx % M
                if diff:
                    k = (k[1:] - k[:-1]) % M
                for lab, code in (("bin", k), ("gray", gray(k))):
                    bits = ((code[:, None] >> np.arange(bps-1, -1, -1)) & 1).astype(np.uint8).reshape(-1)
                    out[(f"conj{conj}", f"rot{rot}", "diff" if diff else "abs", lab)] = bits
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

def config_blocks():
    """(name, folder, config name, n, syndrome function) for the block codes in config/coding"""
    from .. import code as coding
    out = []
    for folder in ("block",):
        for name in coding.names(folder):
            try:
                enc = coding.load(folder, name)._enc
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

    streams: bits, or {tag: bits} when the bit mapping is ambiguous (see `psk_bit_streams`)
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
    """Attach the matching config/coding entry to convolutional hypotheses"""
    from .. import code as coding
    known = {}
    for name in coding.names("conv"):
        c = coding.load("conv", name)
        g = np.asarray(c._enc.generator, dtype=np.uint8)
        known[name] = {tuple(map(tuple, g)), tuple(map(tuple, g[::-1]))}
    for h in hyps:
        if h.kind.startswith("conv") and "g" in h.params and len(h.params["g"]) == 2:
            gs = tuple(h.params["g"][i] for i in sorted(h.params["g"]))
            for name, forms in known.items():
                if gs in forms or tuple(reversed(gs)) in forms:
                    h.coding = ("conv", name)

# --- Identifying a chain from config/coding definitions ---------------------------------------------------

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

def _mismatch(code, seg):
    """Re-encode the Viterbi decision and compare with what was received"""
    enc, dec = code._enc, code._dec
    u = dec.decode(seg)
    e = enc.encode(u.astype(bool))
    n = min(len(e), len(seg))
    return int(np.sum(e[:n] != seg[:n].astype(bool))), n

def scan_defined_conv(streams, names=None, steps=2500, skip=0.1):
    """Try every convolutional definition in config/coding/conv on every bit mapping and puncture alignment

    A correct hypothesis lets the decoder reproduce the received bits (the channel error rate, well below
    what a wrong one gets). Returns (hypotheses, trials).
    """
    from .. import code as coding
    names = names or coding.names("conv")
    hyps, trials = [], 0
    rng = np.random.default_rng(0)
    for name in names:
        code = coding.load("conv", name)
        enc = code._enc
        pat = enc.puncture if enc.puncture is not None else np.ones((enc.den, 1), dtype=bool)
        L = pat.shape[1]
        nb = int(pat.sum())
        per = max(steps//L, 1)
        rand = rng.integers(0, 2, per*nb).astype(np.uint8)
        k0, n0 = _mismatch(code, rand)
        base = max(k0/n0, 1e-4) # What a wrong hypothesis gets
        for tag, bits in streams.items():
            start = int(len(bits)*skip)
            for off in range(nb):
                seg = np.asarray(bits[start+off:start+off+per*nb], dtype=np.uint8)
                if len(seg) < per*nb:
                    continue
                k, n = _mismatch(code, seg)
                trials += 1
                z = float(-deviance_z(k, n, base)) # Fewer mismatches than a wrong code gives
                if z > 3.0 and k/n < 0.5*base: # And clearly fewer, not a marginal difference on a short code
                    hyps.append(Hypothesis("conv-config", name, z,
                        {"offset": off, "mismatch": k/n, "baseline": base, "stream": tag}, ("conv", name)))
    return hyps, trials

def decode_conv(bits, hyp):
    """Run the config definition behind a conv-config hypothesis on a whole bit stream"""
    from .. import code as coding
    code = coding.load(*hyp.coding)
    seg = np.asarray(bits[hyp.params["offset"]:], dtype=np.uint8)
    nb = int(code._enc.puncture.sum()) if code._enc.puncture is not None else code._enc.den
    seg = seg[:len(seg)//nb*nb]
    return code._dec.decode(seg)

def scan_rs(bits, names=None, interleavers=True, packets=200):
    """Find a Reed-Solomon code (config/coding/block type rs), optionally behind an interleaver (config/coding/interleave)

    Returns (hypotheses, trials). The score is how many words have all-zero syndromes.
    """
    from .. import code as coding
    from ..code.types import rs as RS
    names = names or coding.names("block", "rs")
    inters = [None] + [c for c in (coding.load("interleave", n)._enc for n in coding.names("interleave"))
                       if hasattr(c, "I") and hasattr(c, "M")] if interleavers else [None] # Forney: the ones RS sits behind
    hyps, trials = [], 0
    bits = np.asarray(bits, dtype=np.uint8)
    for name in names:
        code = coding.load("block", name)._enc
        if getattr(code, "m", 8) != 8: # The search runs on bytes: codes over other fields (P25's GF(64)) don't apply
            continue
        n = code.den
        # Interleavers eat (I-1)*I*M symbols of start-up; use what is left, at least 60 words
        packets = int(min(packets, max(len(bits)//8//n - 60, 0))) or packets
        for pol in (0, 1):
            b = bits ^ pol
            for off in range(8):
                B = np.packbits(b[off:][:(len(b)-off)//8*8]).astype(np.int64)
                for inter in inters:
                    if inter is None:
                        C = B
                        phases = range(n)
                    else:
                        phases = range(n)
                    for ph in phases:
                        seg = B[ph:ph+(packets + 20)*n]
                        if len(seg) < 80*n:
                            continue
                        if inter is not None:
                            seg = inter.decode(seg, symbols=True)[(inter.I-1)*inter.M*inter.I:]
                        words = seg[:len(seg)//n*n].reshape(-1, n)
                        zero = int((~code.syndromes(words[:60]).any(axis=1)).sum())
                        trials += 1
                        if zero >= 3:
                            z = float(deviance_z(zero, 60, 256.0**-code.nroots))
                            hyps.append(Hypothesis("rs", name, z, {"polarity": pol, "bit_offset": off, "byte_phase": ph,
                                "interleaver": None if inter is None else (inter.I, inter.M), "words_ok": zero/60},
                                ("block", name)))
    return hyps, trials

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
    """A guessed FEC definition. Decoding only happens from a saved config: `save` it, then rp.code.load it"""
    __slots__ = ("kind", "definition", "evidence")
    FOLDER = {"block": "block", "conv": "conv"}

    def __init__(self, kind, definition, evidence=None):
        self.kind = kind
        self.definition = definition
        self.evidence = evidence or {}

    def __repr__(self):
        d = self.definition
        if self.kind == "block":
            return f"<Guess block n={d['n']} k={d['k']} {len(d['checks'])} checks {self.evidence}>"
        return f"<Guess conv K={d['encode']['constraint']} rate {d['encode']['num']}/{d['encode']['den']} {self.evidence}>"

    def save(self, name, path=None):
        """Write config/coding/<kind>/guess/<name>.json (or <path>/<kind>/guess/<name>.json). Returns the file"""
        import json
        from pathlib import Path
        from ..code import serial
        base = Path(path) if path else serial.root()
        folder = base / self.FOLDER[self.kind] / "guess"
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
    """HDLC framing (config/encap/frame): many flags, and frames whose FCS checks out

    Looks at flag counts first (a 0x7E pattern shows up at random once per 256 bits), then deframes the
    streams that have far more than that. Returns (hypotheses, trials).
    """
    from .. import link as encap
    names = names or encap.ls().get("frame", [])
    hyps, trials = [], 0
    for name in names:
        h = encap.load("frame", name)._enc
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
