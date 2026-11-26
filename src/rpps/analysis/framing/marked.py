"""Frames behind a sync marker, the scrambler restarted at each, the block code inside (rpps.analysis.framing.marked)

`layer(bits)` on any bit stream: a shipped profile's marker (`markers`) repeating at one period, within MAX_ERRORS
bits, either polarity; then none or each additive scrambler restarted per frame; then each byte RS code dividing the
frame, behind each block interleaver with a `depth`, scored by the words that decode within t against a random word's
chance (`chance`). Returns `Frames`, whose `steps` are the profile steps that read it (CCSDS TM: ASM, randomizer, RS
interleaved to depth I)."""
import functools
import math
from collections import Counter

import numpy as np

from ...math.prob import deviance_z

MAX_ERRORS = 2 # Bit errors a marker may have and still be one
MIN_MARKER = 16 # Markers shorter than this match random bits too often to frame anything
WORDS = 16 # Codewords a code hypothesis is scored on (at most)

@functools.lru_cache(maxsize=None)
def _markers(state):
    from ... import sequence
    from ..identify._profiles import _profiles
    out = {}
    for _, obj, _ in _profiles():
        for s in obj.get("steps", []):
            ref = s.get("sequence")
            if not isinstance(ref, str) or not (s.get("sync") == "delimit" or s.get("code") == "marker") or ref in out:
                continue
            try:
                bits = np.asarray(sequence.load(ref).bits, dtype=np.uint8).reshape(-1)
            except Exception:
                continue
            if len(bits) >= MIN_MARKER:
                out[ref] = bits
    return out

def markers():
    """{name: bits} of the profiles' sync markers. MIN_MARKER bits or longer"""
    from ... import _config
    return dict(_markers(_config.state()))

def chance(code):
    """A random word's chance to decode. Within t of an RS codeword: about C(n, t) (q - 1)^t / q^(2t), q = 2^m"""
    n, k, m = code.den, code.num, getattr(code, "m", 8)
    t = (n - k)//2
    q = 2.0**m
    lg = math.lgamma(n + 1) - math.lgamma(t + 1) - math.lgamma(n - t + 1) + t*math.log(q - 1) - 2*t*math.log(q)
    return min(1.0, math.exp(lg))

class Frames:
    """A frame layer `layer` found. steps: the profile steps that read it (sync delimit, code marker, scramble,
    interleave, block), as identify lays steps out"""
    __slots__ = ("marker", "inverted", "period", "first", "frames", "scrambler", "interleaver", "depth", "code", "ok",
                 "words", "z")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def steps(self):
        L = self.period - len(markers().get(self.marker, ()))
        out = [{"sync": "delimit", "id": "asm", "sequence": self.marker, "length": L, "keep": True, "polarity": True,
                "errors": MAX_ERRORS},
               {"code": "marker", "sequence": self.marker, "length": L}]
        if self.scrambler:
            out.append({"code": "scramble", "name": self.scrambler, "length": L})
        if self.interleaver:
            out.append({"code": "interleave", "name": self.interleaver, "args": {"depth": self.depth}})
        if self.code:
            out.append({"code": "block", "name": self.code})
        return out

    def __repr__(self):
        inside = f", {self.code} x{self.depth}" + (f" behind {self.interleaver}" if self.interleaver else "") + \
                 f" ({self.ok}/{self.words} words decode, z={self.z:.1f})" if self.code else ""
        return (f"<Frames {self.marker}{' inverted' if self.inverted else ''} every {self.period} bits from {self.first}"
                f" ({self.frames} frames), scrambler {self.scrambler or 'none'}{inside}>")

def _find(bits, marker, max_errors=MAX_ERRORS, min_frames=4):
    """Where `marker` repeats in bits: (period, first, frames, inverted) of the residue class at the commonest spacing
    holding the most hits (at least `min_frames`), or None"""
    b = 2.0*np.asarray(bits, dtype=np.float64) - 1.0
    m = 2.0*np.asarray(marker, dtype=np.float64) - 1.0
    L = len(m)
    if len(b) < min_frames*L:
        return None
    n = 1 << int(np.ceil(np.log2(len(b) + L)))
    corr = np.fft.irfft(np.fft.rfft(b, n)*np.conj(np.fft.rfft(m, n)), n)[:len(b) - L + 1]
    need = L - 2*max_errors - 0.5
    best = None
    for inverted in (False, True):
        hits = np.flatnonzero((-corr if inverted else corr) >= need)
        if len(hits) < min_frames:
            continue
        gaps = Counter(np.diff(hits).tolist())
        for period, _ in gaps.most_common(3):
            if period <= L:
                continue
            res = Counter((hits % period).tolist())
            r, count = res.most_common(1)[0]
            if count >= min_frames and (best is None or count > best[2]):
                first = int(hits[hits % period == r][0])
                best = (int(period), first, int(count), inverted)
    return best

def _bodies(bits, period, first, L, inverted, count=None):
    """(F, period - L) the frames' bodies (after the marker), in the polarity the marker says"""
    x = np.asarray(bits, dtype=np.uint8)[first:]
    F = len(x)//period if count is None else min(count, len(x)//period)
    if F == 0:
        return np.zeros((0, period - L), np.uint8)
    f = x[:F*period].reshape(F, period)[:, L:]
    return f ^ 1 if inverted else f

def _scramblers():
    from ...code.scram import serial
    out = [None]
    for name in serial.ls(blind=True).get("additive", []):
        out.append(f"additive/{name}")
    return out

@functools.lru_cache(maxsize=None)
def _keystream(name, n):
    from ...code.scram import serial
    return np.asarray(serial.load(name).seq.bits(n), dtype=np.uint8).reshape(-1)[:n]

def _rs_codes():
    from ... import code as coding
    from ...code.coding import cached
    out = []
    for name in coding.names("block", "rs", blind=True):
        c = cached(f"block/{name}").encoder
        if getattr(c, "m", 8) == 8:
            out.append((name, c))
    return out

def _interleavers(nbytes, n):
    """[(name, depth, encoder)] of the block interleavers with a depth that fit `nbytes` as `nbytes/n` codewords"""
    from ... import code as coding, _config
    if nbytes % n:
        return []
    I = nbytes//n
    out = []
    for name in coding.names("interleave", "block", blind=True):
        try:
            obj = _config.read(_config.find("interleave", name))
        except Exception:
            continue
        if "depth" not in obj.get("params", {}) or obj.get("symbol", 1) != 8:
            continue
        try:
            enc = coding.load(f"interleave/{name}", depth=I).encoder
        except Exception:
            continue
        if getattr(enc, "n", None) == nbytes:
            out.append((name, I, enc))
    return out

def _decodes(code, words):
    """Codewords (W, 8n) bits that decode: (decoded, of)"""
    W = len(words)
    if W == 0:
        return 0, 0
    _, err = code.decode(words.reshape(-1).astype(bool), status=True)
    err = np.asarray(err)[:W]
    flat = (words.reshape(W, -1, 8) == words.reshape(W, -1, 8)[:, :1]).all(axis=(1, 2)) # (one value: any RS code's)
    return int(np.count_nonzero((err >= 0) & ~flat)), W

def _codes_in(body, z_need):
    """The code in frame bodies (F, B): (name, interleaver, depth, ok, words, z) of the best above z_need, or None"""
    F, B = body.shape
    if B % 8:
        return None, 0
    nbytes = B//8
    best, trials = None, 0
    for name, code in _rs_codes():
        n = code.den
        if nbytes % n:
            continue
        p0 = chance(code)
        opts = [(None, nbytes//n, None)] + _interleavers(nbytes, n)
        for iname, I, enc in opts:
            rows = []
            for f in body[:max(1, -(-WORDS//I))]:
                x = enc.decode(f.astype(bool)) if enc is not None else f
                rows.append(np.asarray(x, dtype=np.uint8)[:B].reshape(-1, 8*n))
            words = np.concatenate(rows)[:WORDS]
            ok, W = _decodes(code, words)
            trials += 1
            z = float(deviance_z(ok, W, p0)) if W else 0.0
            # (at depth 1 an interleaver is the identity: a tie goes to it, as profiles name one at every depth)
            if z > z_need and (best is None or (ok/W, z, iname is not None) > (best[3]/best[4], best[5], best[1] is not None)):
                best = (name, iname, I, ok, W, z)
    return best, trials

def _structure(body, head=64):
    """Bits of the frames' first `head` that are the same in every frame (F, B): a frame header's constant fields
    (version, ids); random data leaves about head/2^F"""
    h = np.asarray(body)[:, :head]
    return int(np.count_nonzero((h == h[:1]).all(axis=0))) if len(h) > 1 else 0

def layer(bits, min_frames=4, z=None):
    """The frame layer in `bits`: a Frames, or None. None when no marker repeats. Without a code that
    decodes inside, the Frames has the marker, period and scrambler as far as they go (code None). z: the z a code
    hypothesis must pass (default: rpps.math.prob.z_critical over the hypotheses tried)"""
    from ...math.prob import z_critical
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    found = []
    for ref, m in markers().items():
        f = _find(bits, m, min_frames=min_frames)
        if f is not None:
            found.append((f[2], ref, f))
    if not found:
        return None
    _, ref, (period, first, count, inverted) = max(found, key=lambda t: t[0])
    L = len(markers()[ref])
    body = _bodies(bits, period, first, L, inverted)
    out = Frames(marker=ref, inverted=inverted, period=period, first=first, frames=count, scrambler=None,
                 interleaver=None, depth=None, code=None, ok=None, words=None, z=None)
    hyps = []
    for scr in _scramblers():
        try:
            d = body ^ _keystream(scr, body.shape[1])[None, :] if scr else body
        except Exception:
            continue
        need = z if z is not None else z_critical(max(1, 4*len(_rs_codes())*(1 + len(_scramblers()))))
        best, _ = _codes_in(d, need)
        if best is not None:
            hyps.append((best, scr, _structure(d)))
    if hyps:
        # A keystream that is itself a codeword (the CCSDS randomizer's 255-bit period over one 255-byte word) decodes
        # either way: then what the frames carry tells, the bits that stay the same frame to frame (a header's)
        (code, iname, I, ok, W, zz), scr, _ = max(hyps, key=lambda h: (h[0][3]/h[0][4], h[0][5], h[2]))
        out.scrambler, out.code, out.interleaver, out.depth, out.ok, out.words, out.z = scr, code, iname, I, ok, W, zz
    return out
