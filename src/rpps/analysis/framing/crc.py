"""Which CRC checks a set of frames, and where it sits: the generator, init and xorout, the bits it covers

    fit = rp.analysis.framing.crc.search(frames)     # frames: bit arrays as sent (bool), any lengths
    print(fit)        # "CRC-16 poly 0x1021 init 0xffff xorout 0x0 (crc16_ccitt_false) over bits 0..95, field at 96..111, ..."
    fit.check(frame)  # True when a frame passes
    fit.guess.save("mine")   # config/crc/guess/mine.json when no shipped definition matched

The method (RevEng's, on bit streams). On bits fed in the order sent, a CRC's register is linear in the message:
r(m) = I x^n + m(x) x^w mod P, and the field sent is r XOR a constant, most significant bit first (a reflected CRC sent
least significant bit first reads the same way, with xorout reflected; `refin` only says how bytes become that stream).
So for two frames of one length, the XOR of their bits up to the field's end is a polynomial P divides, whatever init
and xorout are:

  1. Generator and field end: the GCD of a few such differences, the frame cut at each candidate end from the last bit
     back (powers of x stripped). A degree in `widths` is a candidate; the end moves back over constant columns
     (fill after the field). When bits before the coverage differ between frames (an uncovered header: an address, a
     counter), the differences are first combined to cancel them (row reduction on the leading columns), which needs
     more frames than that header has varying bits.
  2. Coverage start: K = field XOR r0 (r0 the register from zero over the covered bits) is the same for every frame of a
     length (K = I x^n mod P XOR xorout) at the right start only. Every start is scored at once (suffix XORs of
     x^j mod P); the start passing the most frames wins, a start where init and xorout come out plain (0 or all ones,
     or a shipped definition) breaking ties across constant leading bits, which any init absorbs.
  3. Init and xorout: two message lengths separate them (K_n1 XOR K_n2 = I (x^n1 + x^n2) mod P), all but the inits
     P's common factors with x^n1 + x^n2 hide (any generator with x + 1); with one length only their combination is
     known. Then (`ambiguous`) a shipped definition (config/crc) or plain values (0, all ones) pick one.
  4. Evidence: frames not used to find the generator that pass, against 2^-w each by chance.

Limits: frames must be error free (decoded, or a CRC would not be there to find); several frames are tried in
turn (`trials`), so some bad frames are tolerated. A per-frame value XORed onto the field (LTE's RNTI mask) looks like
a varying xorout and is not found. Needs four frames of one length and a few more to confirm.
"""
import numpy as np

from ...math import gf2
from ...math.bits import reflect
from ..result import Guess

class CRCFit:
    """`search`: width, poly (without the x^w term), init, xorout (in the definition's convention: reflected when
    `refout`), refin, refout, start (first covered bit), end (bit after the field, frame length minus `tail`), tail,
    names (the config/crc definitions that match: bit streams cannot tell those differing in refin / refout only apart),
    name (the first of them, else None), ambiguous (init and xorout not separated: one frame
    length, or a generator sharing a factor with x^n1 + x^n2, as any with x + 1 does), hits / frames (frames that pass), used (frames that found the generator), log10_chance, guess"""
    __slots__ = ("width", "poly", "init", "xorout", "refin", "refout", "start", "tail", "names", "ambiguous", "hits",
                 "frames", "used", "log10_chance")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def name(self):
        return self.names[0] if self.names else None

    @property
    def definition(self):
        return {"base": "crc", "type": "crc", "width": self.width, "poly": hex(self.poly), "init": hex(self.init),
                "refin": self.refin, "refout": self.refout, "xorout": hex(self.xorout)}

    @property
    def guess(self):
        """A Guess to save as config/crc/guess/<name>.json"""
        return Guess("crc", self.definition, {"start": self.start, "tail": self.tail, "hits": self.hits,
                                              "frames": self.frames, "log10_chance": round(self.log10_chance, 1),
                                              "ambiguous": self.ambiguous, "matches": self.names})

    def end(self, length):
        """The bit after the CRC field in a frame of `length` bits"""
        return length - self.tail

    def _xstar(self):
        return reflect(self.xorout, self.width) if self.refout else self.xorout

    def check(self, frame):
        """True when the frame (bits as sent) carries this CRC at this place"""
        b = np.asarray(frame).astype(bool).reshape(-1)
        E = len(b) - self.tail
        if E - self.width < self.start:
            return False
        P = (1 << self.width) | self.poly
        xp = _xpow(P, self.width, E)
        r0 = int(np.bitwise_xor.reduce(np.where(b[self.start:E - self.width], xp[E - 1 - np.arange(self.start, E - self.width)], 0)))
        n = E - self.width - self.start
        return _value(b[E - self.width:E]) == r0 ^ gf2.pmod(self.init << n, P) ^ self._xstar()

    def __str__(self):
        nm = f" ({' = '.join(self.names)})" if self.names else ""
        amb = " (init and xorout not separated by these frame lengths: one of the pairs that fit)" if self.ambiguous else ""
        return (f"CRC-{self.width} poly {self.poly:#x} init {self.init:#x} xorout {self.xorout:#x}"
                f"{' refout' if self.refout else ''}{nm}{amb}: covers from bit {self.start}, field ends {self.tail} bits "
                f"before the frame's end; {self.hits}/{self.frames} frames pass ({self.used} found it), "
                f"chance 1e{self.log10_chance:.0f}")

    __repr__ = __str__

def _value(bits):
    """Bits, most significant first, as an int"""
    v = 0
    for b in np.asarray(bits).astype(np.uint8):
        v = (v << 1) | int(b)
    return v

def _poly(bits):
    """Bits (first = highest power) as a GF(2) polynomial, a Python int"""
    b = np.asarray(bits).astype(np.uint8)
    return int.from_bytes(np.packbits(b).tobytes(), "big") >> ((-len(b)) % 8) if len(b) else 0

def _xpow(P, w, n):
    """x^j mod P for j < n, uint64"""
    out = np.zeros(max(n, 1), dtype=np.uint64)
    c, top, m, p = 1, 1 << w, (1 << w) - 1, P & ((1 << w) - 1)
    for j in range(n):
        out[j] = c
        c <<= 1
        if c & top:
            c = (c ^ top) ^ p
    return out

def _strip_x(g):
    while g and not g & 1:
        g >>= 1
    return g

def _frames(frames):
    out = []
    for f in frames:
        if isinstance(f, (bytes, bytearray)):
            f = np.unpackbits(np.frombuffer(bytes(f), np.uint8))
        f = np.asarray(f)
        if f.dtype == np.uint8 and f.size and f.max() > 1:
            f = np.unpackbits(f)
        out.append(f.astype(bool).reshape(-1))
    return out

def _log10_binom_tail(n, h, w):
    """log10 of the chance that h of n frames pass a w-bit check by chance (union bound C(n, h) 2^-wh)"""
    from math import lgamma, log
    if h <= 0:
        return 0.0
    return (lgamma(n + 1) - lgamma(h + 1) - lgamma(n - h + 1))/log(10) - h*w*np.log10(2)

def _catalogue(w, p):
    """Shipped config/crc definitions of this width and generator: [(name, init, xorout, refin, refout)]"""
    from ... import link
    out = []
    for name in link.ls()["crc"]:
        try:
            c = link.load(f"crc/{name}").encoder
        except Exception: # a definition that does not load is not a candidate
            continue
        if getattr(c, "width", None) == w and getattr(c, "poly", None) == p and hasattr(c, "init"):
            out.append((name, c.init, c.xorout, c.refin, c.refout))
    return out

def _keys(B, P, w, E, starts):
    """K = field XOR r0 for every frame (rows of B, all E + tail long) and every start: (len(B), len(starts)) uint64"""
    xp = _xpow(P, w, E)
    contrib = np.where(B[:, :E - w], xp[E - 1 - np.arange(E - w)][None, :], np.uint64(0))
    suffix = np.bitwise_xor.accumulate(contrib[:, ::-1], axis=1)[:, ::-1] # suffix[:, a] = r0 over [a, E - w)
    field = np.array([_value(r) for r in B[:, E - w:E]], dtype=np.uint64)
    return suffix[:, starts] ^ field[:, None]

def _mode(col):
    v, c = np.unique(col, return_counts=True)
    i = int(np.argmax(c))
    return v[i], int(c[i])

def _plain(w):
    return (0, (1 << w) - 1)

def _solve_init(K1, n1, K2, n2, P, w):
    """(I, null) with I (x^n1 + x^n2) = K1 ^ K2 mod P: one solution and a basis of the inits that change nothing (when P
    shares a factor with x^n1 + x^n2: x + 1 always does), or None when the lengths disagree"""
    basis, null = {}, [] # pivot bit: (image, I)
    for j in range(w):
        img, I = gf2.pmod(1 << (j + n1), P) ^ gf2.pmod(1 << (j + n2), P), 1 << j
        for b in sorted(basis, reverse=True):
            if img >> b & 1:
                img, I = img ^ basis[b][0], I ^ basis[b][1]
        if img:
            basis[img.bit_length() - 1] = (img, I)
        else:
            null.append(I)
    t, I = K1 ^ K2, 0
    for b in sorted(basis, reverse=True):
        if t >> b & 1:
            t, I = t ^ basis[b][0], I ^ basis[b][1]
    return (I, null) if t == 0 else None

def _small_irreducibles(max_deg=8):
    out = []
    for f in range(2, 1 << (max_deg + 1)):
        if f & 1 or f == 2:
            d = f.bit_length() - 1
            if all(gf2.pmod(f, g) for g in out if 2*(g.bit_length() - 1) <= d):
                out.append(f)
    return out

_SMALL = _small_irreducibles()

def _reductions(g, widths, depth=3):
    """g and g with up to `depth` small irreducible factors divided out, of a degree in widths: a GCD of a few
    differences keeps the factors they happen to share besides the generator"""
    seen, todo = {g}, [g]
    for _ in range(depth):
        nxt = []
        for h in todo:
            for f in _SMALL:
                if f.bit_length() < h.bit_length() and gf2.pmod(h, f) == 0:
                    q = gf2.pdivmod(h, f)[0]
                    if q not in seen:
                        seen.add(q)
                        nxt.append(q)
        todo = nxt
    return sorted((h for h in seen if h.bit_length() - 1 in widths), key=lambda h: -h.bit_length())

def _gcd_ends(D, widths, min_end):
    """(GCD with powers of x stripped, e) of difference rows D (full frames of bits), cut at each end from the last back,
    where some reduction of it has a degree in widths"""
    for e in range(D.shape[1], min_end - 1, -1):
        g = _strip_x(gf2.pgcd(*[_poly(r[:e]) for r in D]))
        if g.bit_length() - 1 >= min(widths):
            yield g, e

def _cancel_leading(D, e, keep=5):
    """Rows of D combined to vanish on as many leading columns as possible (cut at e): the last rows of its RREF"""
    R = gf2.rref(D[:, :e])[0].astype(bool)
    return R[-keep:] if len(R) >= 2 else None

def search(frames, widths=range(3, 65), trials=8, false_alarm=1e-3, seed=0):
    """The CRC the frames (bit arrays as sent; bytes are unpacked MSB first) carry, as a CRCFit, or None. Frames of the
    most common length find the generator; every frame is checked against it. widths: CRC sizes to accept"""
    F = _frames(frames)
    widths = set(int(w) for w in widths if 1 <= int(w) <= 64)
    lengths, counts = np.unique([len(f) for f in F], return_counts=True)
    rng = np.random.default_rng(seed)
    best, tried = None, set()

    def consider(g, e, B, L, used):
        nonlocal best
        for P in _reductions(g, widths):
            if (P, e) in tried:
                continue
            tried.add((P, e))
            f = _verify(F, B, P, L, e, used)
            if f is not None and (best is None or (f.hits, f.width) > (best.hits, best.width)): # a divisor of P passes wherever P does
                best = f
        return best is not None and best.hits == best.frames

    for L in lengths[np.argsort(-counts)]:
        L = int(L)
        B = np.array([f for f in F if len(f) == L])
        B = B[np.sort(np.unique(B, axis=0, return_index=True)[1])] # repeated frames say nothing new
        if len(B) < 3:
            continue
        done = False
        for t in range(trials):
            pick = rng.choice(len(B), size=min(6, len(B)), replace=False) if t else np.arange(min(6, len(B)))
            for g, e in _gcd_ends(B[pick[1:]] ^ B[pick[0]], widths, min(widths) + 1):
                if consider(g, e, B, L, len(pick)):
                    done = True
                    break
            if done or (best is not None and t >= 2):
                break
        if (best is None or best.hits < best.frames) and len(B) >= 8: # an uncovered header that varies: cancel it first
            for e in range(L, min(widths), -1):
                D = _cancel_leading(B[1:] ^ B[0], e)
                if D is not None and consider(_strip_x(gf2.pgcd(*[_poly(r) for r in D])), e, B, L, len(D) + 1):
                    break
        if best is not None:
            break
    if best is None or best.log10_chance > np.log10(false_alarm) - np.log10(max(len(tried), 1)):
        return None
    return best

def _verify(F, B, P, L, e, used):
    """CRCFit for generator P with the field ending at e in frames of L bits (B), or before constant columns there
    (fill, which any generator passes over: those ends are tried too, against every frame), or None"""
    w = P.bit_length() - 1
    ends = [e]
    while ends[-1] - 1 > w and B[:, ends[-1] - 1].all() == B[:, ends[-1] - 1].any():
        ends.append(ends[-1] - 1)
    best = None
    for end in ends: # the earliest of equals: the field before the fill
        f = _verify_end(F, B, P, L, end, used)
        if f is not None and (best is None or f.hits >= best.hits):
            best = f
    return best

def _verify_end(F, B, P, L, e, used):
    w = P.bit_length() - 1
    if e - w < 1:
        return None
    tail = L - e
    starts = np.arange(0, e - w)
    K = _keys(B, P, w, e, starts)
    score = np.array([_mode(K[:, a])[1] for a in range(len(starts))])
    top = score.max()
    if top < min(len(B), 3):
        return None
    cands = np.flatnonzero(score == top)
    p = P ^ (1 << w)
    cat = _catalogue(w, p)
    # Every frame, at each length, with the field `tail` bits before its end
    groups = {}
    for f in F:
        if len(f) - tail - w >= 1:
            groups.setdefault(len(f), []).append(f)
    choice = None
    for a in cands: # prefer a shipped or plain init/xorout, then the latest start (leading zeros into a zero register say nothing)
        a = int(a)
        Ks = {}
        for n_len, fs in groups.items():
            if n_len - tail - w <= a:
                continue
            Kg = _keys(np.array(fs), P, w, n_len - tail, np.array([a]))[:, 0]
            v, c = _mode(Kg)
            Ks[n_len] = (int(v), c, len(fs))
        fit = _init_xorout(Ks, P, w, tail, a, cat)
        if fit is None:
            continue
        init, xstar, names, refin, refout, ambiguous, hits, total = fit
        plain = bool(names) or (init in _plain(w) and xstar in _plain(w))
        rank = (hits, plain, a)
        if choice is None or rank > choice[0]:
            choice = (rank, a, init, xstar, names, refin, refout, ambiguous, hits, total)
    if choice is None:
        return None
    _, a, init, xstar, names, refin, refout, ambiguous, hits, total = choice
    free = hits - used
    if free < 2:
        return None
    lc = _log10_binom_tail(total - used, free, w)
    xor = reflect(xstar, w) if refout else xstar
    return CRCFit(width=w, poly=p, init=init, xorout=xor, refin=refin, refout=refout, start=a, tail=tail,
                  names=names, ambiguous=ambiguous, hits=hits, frames=total, used=used, log10_chance=float(lc))

def _init_xorout(Ks, P, w, tail, a, cat):
    """init, xorout (unreflected), matching config/crc names, refout, ambiguous, frames passing, frames, from K per length"""
    if not Ks:
        return None
    n = {L: L - tail - w - a for L in Ks}
    def passing(I, X):
        return sum(c for L, (v, c, tot) in Ks.items() if gf2.pmod(I << n[L], P) ^ X == v)
    total = sum(tot for _, _, tot in Ks.values())
    full = sum(c for _, c, _ in Ks.values())
    match = [(name, I, reflect(xo, w) if refout else xo, refin, refout) for name, I, xo, refin, refout in cat]
    match = [m for m in match if passing(m[1], m[2]) == full] # shipped definitions that explain every length
    if match: # bit streams cannot tell refin / refout apart: every one that fits is named
        _, I, X, refin, refout = match[0]
        return I, X, [m[0] for m in match], refin, refout, len(Ks) < 2, full, total
    Ls = sorted(Ks, key=lambda L: -Ks[L][1])
    if len(Ls) >= 2:
        for L2 in Ls[1:]:
            sol = _solve_init(Ks[Ls[0]][0], n[Ls[0]], Ks[L2][0], n[L2], P, w)
            if sol is None:
                continue
            I0, null = sol
            choices = []
            for m in range(1 << min(len(null), 8)): # the inits two lengths cannot tell apart: a plain one if there is one
                I = I0
                for i, v in enumerate(null):
                    if m >> i & 1:
                        I ^= v
                X = Ks[Ls[0]][0] ^ gf2.pmod(I << n[Ls[0]], P)
                choices.append((passing(I, X), I in _plain(w) and X in _plain(w), -m, I, X))
            h, _, _, I, X = max(choices)
            return I, X, [], False, False, bool(null), h, total
    K, nL = Ks[Ls[0]][0], n[Ls[0]]
    for I in _plain(w):
        X = K ^ gf2.pmod(I << nL, P)
        if X in _plain(w):
            return I, X, [], False, False, len(Ks) < 2, passing(I, X), total
    return 0, K, [], False, False, len(Ks) < 2, passing(0, K), total
