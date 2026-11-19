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
from .result import Hypothesis, Report, Guess, pick
from .code.conv.mother import scan_conv, scan_defined_conv, guess_conv
from .code.block.linear import recover_dual
from .framing.bits import frame_length, frame_markers, MARKER_PERIODS, scan_hdlc
from .sequence.scrambler import descramblers, cached_scrambler
from .sequence import scrambler
from .code.conv import puncture
from .code.block import rs as blind_rs, linear as blind_block
from ..symbol.mapping import hard_bits
from ..code.scram.scrambler import Feedthrough

# --- Bit streams from symbols -------------------------------------------------------------------------

def _constellation(c):
    from .. import symbol
    return symbol.load_or_generate(c) if isinstance(c, str) else c


def diff_alphabets(constellation):
    """Phase-change alphabets a differential receiver of `constellation` would demap against: {name: Constellation}

    The PSK definitions with as many points (config/constellation/psk: QPSK is the pi/4-DQPSK alphabet, qpsk_axes 802.11b's;
    labelling variants such as the DVB-S2 maps are skipped), and the constellation itself when 1 is one of its points
    (DBPSK is BPSK)."""
    from .. import symbol, _config
    c = _constellation(constellation)
    out = {}
    for stem in symbol.ls().get("psk", []):
        if stem.startswith("dvbs2_"):
            continue
        d = symbol.load(f"psk/{stem}")
        if d.order == c.order and np.allclose(np.abs(d.points), 1, atol=1e-6):
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
            out[("abs", i)] = hard_bits(y, c.points, m.arr)
    if differential and len(y) > 1:
        r = y[1:]/np.where(np.abs(y[:-1]) < 1e-12, 1.0, y[:-1])
        r = r/np.maximum(np.abs(r), 1e-12)
        for name, d in diff_alphabets(c).items():
            for i, m in enumerate(d.maps):
                if abs(np.angle(np.exp(1j*m.rotation))) > 1e-6:
                    continue
                if keys is None or ("diff", name, i) in keys:
                    out[("diff", name, i)] = hard_bits(r, d.points, m.arr)
    return out

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

# --- One answer: which code a symbol stream carries ------------------------------------------------------

CONV_SYMBOLS = 2**15 # Symbols the convolutional and block scans look at (the inner layers use all there are)


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
    return [(s["code"], s.get("name")) for s in steps if s.get("code") in ("block", "conv", "interleave", "line", "marker")]


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
                obj = _config.resolve(_config.read(_config.find("profile", f"{fam}/{mode}")), {}, f"profile/{fam}/{mode}")
            except Exception:
                continue
            out |= {st["name"] for st in obj.get("steps", []) if st.get("code") == "conv"
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
        return {"base": "profile", "comment": f"detected by rp.analysis.identify.identify: {self}", "steps": self.steps}

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


def identify(symbols, constellation, names=None, blind=False, differential=True, profiles=True):
    """Which code from config/fec timed, carrier-corrected symbols carry, under which map, and how to decode it

    In turn, stopping at the first that stands out: configured convolutional codes (any puncture phase), with a Reed-Solomon
    code behind them (scan_rs; then the markers that start its codewords, for Align); HDLC framing; configured block codes.
    blind: look for what no definition covers, as `Guess`es only (nothing decodes from a guess). With no configured code: a
        known mother code with an unknown puncture pattern (rpps.analysis.code.conv.puncture) and what is inside it, a binary block code
        on the channel bits (rpps.analysis.code.block.linear), then unknown rate 1/n convolutional codes (detect_fec). Inside a
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
    conv = pick(rep.hyps, rep.z_crit, lambda h: h.params["mismatch"]/h.params["baseline"],
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
        steps = pre + symbol_steps(key) + [{"sync": "puncture", "conv": out.name},
                                           {"code": "conv", "name": out.name}]
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
                    steps += [{"code": "interleave", "name": p["interleaver"]}]
                steps += [{"code": "block", "name": rs.name}]
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
    v = pick(rep.hyps, rep.z_crit, lambda h: (-h.params["words_ok"], -h.params["n"]),
              lambda b, h: h.params["words_ok"] >= 0.5*b.params["words_ok"])
    if v is not None:
        out = CodeDetection(found=True, complete=True, mod=c.name, kind="block", name=v.coding[1], stream=v.params["stream"],
                            offset=v.params["offset"], period=v.params["n"], z=v.z, report=rep, n=len(y))
        out.steps = pre + symbol_steps(out.stream) + [{"code": "block", "name": out.name}]
        if blind: # Its words may be the rows of a product code (a turbo product code): columns of them that are codewords too
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
       (rpps.analysis.code.block.rs.search over `rs_lengths`, or every length with `exhaustive`)
    2. a binary block code (rpps.analysis.code.block.linear.solve_block: BCH / cyclic, product, LDPC or any linear code, by rank)
    3. a feedthrough scrambler (rpps.analysis.sequence.scrambler.search, up to `max_taps` taps); if one is found, the same codes on
       what it gives, then HDLC framing. A scrambler hypothesis with no taps means the bits already had structure.
    4. none found (descrambled data that is random itself, compressed or encrypted, stays white): Reed-Solomon behind each
       configured feedthrough scrambler (its structure is algebraic: a scrambler search can't see it)
    Nothing here decodes: a Guess is a definition once it is saved. Empty: nothing found (the bits stay white: another
    layer, or random payload such as encrypted data)."""

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
            found = code_layer(Feedthrough.descramble(bits, obj["poly"]), with_block=False)
            if found:
                return [(Hypothesis("scrambler", f"feedthrough/{name}", found[0][0].z, {"taps": list(obj["poly"]), "config": name}),
                         None)] + found
        return []
    out = [(h[0], g)]
    taps = h[0].params["taps"]
    d = Feedthrough.descramble(bits, taps) if taps else bits
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