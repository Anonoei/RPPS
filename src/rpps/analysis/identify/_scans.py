"""Scans of hard bits for block / LDPC codes, whiteness and Reed-Solomon (rpps.analysis.identify)"""
import numpy as np

from ...math.gf2 import systematic
from ...math.prob import deviance_z
from ...code.coding import puncture_period, mismatch, cached
from ..result import Hypothesis, Report, Guess
from ..code.conv.mother import scan_conv
from ..code.block.linear import recover_dual
from ..framing.bits import frame_length
from ..sequence.scrambler import descramblers

# --- Block / LDPC ---------------------------------------------------------------------------------

def repeat_check(count):
    """Parity checks of a repetition code: all bits of the word equal"""
    H = np.zeros((count-1, count), dtype=bool)
    for i in range(count-1):
        H[i, i] = H[i, i+1] = True
    return H


SCREEN_WORDS = 24 # Words an alignment is screened on before all of them are checked

def scan_block(bits, fails, r, n, name, tag=None, max_offsets=None):
    """Fraction of words satisfying every parity check, at each word alignment. Returns (hypotheses, trials)

    fails: words (W, n) -> (W,) bool, the words that fail some check (`config_blocks` builds these); r: number of checks
    An alignment is screened on its first SCREEN_WORDS words: under 15% of them passing, the rest are not checked (a
    wrong alignment passes 2^-r of the time; a code just at the cut below, two fifths, falls under 15% of 24 words
    0.2% of the time)
    """
    bits = np.asarray(bits, dtype=np.uint8)
    hyps, trials = [], 0
    p0 = 2.0**-r
    for off in range(n if max_offsets is None else min(n, max_offsets)):
        w = len(bits[off:])//n
        if w < 50:
            break
        words = bits[off:off+w*n].reshape(w, n)
        if w > SCREEN_WORDS + 8 and (~fails(words[:SCREEN_WORDS])).mean() < max(0.15, 4*p0):
            trials += 1
            continue
        zero = int((~fails(words)).sum())
        trials += 1
        z = float(deviance_z(zero, w, p0))
        # A real code at a sane error rate passes most words; a high baseline (few checks) passes some by chance
        if z > 3.0 and zero/w >= max(0.4, 8*p0):
            hyps.append(Hypothesis("block", name, z, {"n": n, "k": n-r, "offset": off, "words_ok": zero/w, "stream": tag}))
    return hyps, trials


_BLOCKS = None


FIRST_CHECKS = 8 # Checks tried on every word before the rest: a word not of the code fails one of them but for 2^-8


def _fails(partial, full, r):
    """words -> which fail some check, from partial(words, m) (the first m checks) and full(words): the first
    FIRST_CHECKS on every word, all of them only on the words that pass those (on bits not of the code, about 2^-8 of them)"""
    def fails(words):
        if r <= 2*FIRST_CHECKS:
            return full(words).any(axis=1)
        out = partial(words, FIRST_CHECKS).any(axis=1)
        rest = np.flatnonzero(~out)
        if len(rest):
            out[rest] = full(words[rest]).any(axis=1)
        return out
    return fails


def config_blocks():
    """(name, folder, config name, n, r checks, fails function) for the block codes in config/fec (loaded once)"""
    global _BLOCKS
    if _BLOCKS is None:
        _BLOCKS = _load_blocks()
    return _BLOCKS


def _load_blocks():
    from ... import code as coding
    out = []
    for folder in ("block",):
        for name in coding.names(folder):
            try:
                enc = coding.load(f"{folder}/{name}").encoder
            except Exception: # A config that doesn't load can't be a hypothesis
                continue
            if hasattr(enc, "syndrome"): # LDPC: sparse checks
                r = enc.M
                fails = _fails(enc.syndrome, enc.syndrome, r)
            elif hasattr(enc, "check"): # linear: check is H transposed
                chk = np.asarray(enc.check, dtype=np.uint8)
                r = chk.shape[1]
                fails = _fails(lambda w, m, chk=chk: (w.astype(np.uint8) @ chk[:, :m]) % 2, lambda w, chk=chk: (w.astype(np.uint8) @ chk) % 2, r)
            elif enc.num == 1: # repetition: all bits of the word equal
                H = repeat_check(enc.den).astype(np.uint8)
                r = H.shape[0]
                fails = _fails(lambda w, m, H=H: (w.astype(np.uint8) @ H[:m].T) % 2, lambda w, H=H: (w.astype(np.uint8) @ H.T) % 2, r)
            else:
                continue
            out.append((name.partition("/")[0], folder, name, enc.den, r, fails))
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
            for name, folder, cfg, bn, r, fails in block_codes:
                h, n = scan_block(b, fails, r, bn, name, t, max_offsets)
                for hh in h:
                    hh.coding = (folder, cfg)
                hyps += h
                trials += n
    _label_configs(hyps)
    return Report(hyps, trials, facts)


def _label_configs(hyps):
    """Attach the matching config/fec entry to convolutional hypotheses"""
    from ... import code as coding
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
    flat_rows = rows.reshape(-1, n)
    # S_1 of every word first: a word not of the code has it nonzero but for 1/256; all of them only for the rest
    zero = code.syndromes(flat_rows, 1)[:, 0] == 0
    rest = np.flatnonzero(zero)
    if len(rest):
        zero[rest] = ~code.syndromes(flat_rows[rest]).any(axis=1)
    zero = zero.reshape(len(phases), words)
    flat = (rows == rows[:, :, :1]).all(axis=2) # One value repeated (idle, a dead carrier) is a codeword of any RS code
    return (zero & ~flat).sum(axis=1)


def decode_tail(bits, code, fraction=0.5, probe=4000, settle=16384):
    """Viterbi output of the later part of a stream, at the puncture phase that holds there

    Inner layers are looked for in the later part of a window, where the sync before it has settled: the last `fraction`
    of it, or all but the first `settle` bits when that is more (a long window keeps nearly all its bits for the layers
    inside, which need codewords: an RS behind a deinterleaver loses its delay too). A symbol the timing slipped on the
    way moves the puncture phase, so it is found again at the start of that part."""
    bits = np.asarray(bits, dtype=np.uint8)
    nb = puncture_period(code)
    start = min(int(len(bits)*(1 - fraction)), settle)
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
    from ... import code as coding
    names = names or coding.names("block", "rs")
    inters = [(None, None)]
    if interleavers:
        for nm in coding.names("interleave"):
            enc = cached(f"interleave/{nm}").encoder # (loaded once a process: scan_rs runs on every decoded stream)
            if hasattr(enc, "I") and hasattr(enc, "M") and getattr(enc, "symbol", 8) == 8:
                inters.append((nm, enc))
    hyps, trials, short = [], 0, False
    bits = np.asarray(bits, dtype=np.uint8)
    for name in names:
        code = cached(f"block/{name}").encoder
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
