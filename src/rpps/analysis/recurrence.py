"""Every linear recurrence up to a degree, scored at once: the taps of a scrambler (feedthrough or additive), a
differential code, or a short code's checks, from hard bits

    found = rp.analysis.recurrence.search(bits, max_degree=20)
    print(found[0])     # e.g. "1 + x^3 + x^20 (taps 20, 3): holds 87.5% (z 412.0), weight 3: data bias 0.75 per bit if
                        #  additive, 0.75 if feedthrough"

A polynomial f(D) = 1 + f_1 D + ... + f_L D^L is a check on bits s when f(D) s = 0 holds more often than chance. Both
scramblers make one:
  feedthrough (self-synchronising, multiplicative): s = d / f(D), so f(D) s = d: the check holds as often as a data bit
      is its common value (idle fill, flags, headers: data bias b, the same for every multiple of f)
  additive (synchronous, an LFSR with f its feedback polynomial): s = d + k with f(D) k = 0, so f(D) s = f(D) d: the data
      bits at the taps, holding with bias b^w (w the taps, weight of f), weaker the more taps
A differential code (1 + D) or a short code's parity check is one too. Counting the stream's (L+1)-bit windows by value and
taking the Walsh-Hadamard transform of the counts gives every polynomial of degree up to L at once: how many windows it
holds on less how many it fails (the same transform as rpps.analysis.tcm; Ding et al. 2022/2023 test the primitive
polynomials one by one with the same statistic). Checks that are multiples of a smaller one found are dropped, as are
those without a constant term (shifts).

Below the stream's own structure (no bias in the data: compressed, encrypted, or coded) nothing holds: then the checks
of a code under a feedthrough scrambler are f(D) h(D), and `gcd` of several gives f (Han, Zhang et al., IEEE 2018).
"""
import numpy as np

from .tcm import fwht

class Recurrence:
    """A check found: taps (delays, descending, without 0), poly (int: bit j = the coefficient of D^j), degree, weight
    (taps + 1), holds (fraction of windows it holds on), z (its bias over chance), bias (2 holds - 1, signed)"""
    __slots__ = ("poly", "taps", "degree", "weight", "holds", "z", "bias")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def name(self):
        return " + ".join(["1"] + [f"x^{t}" if t > 1 else "x" for t in sorted(self.taps)])

    def data_bias(self, additive):
        """The data's bias per bit this check implies: |bias| for a feedthrough scrambler, |bias|^(1/weight) additive"""
        b = abs(self.bias)
        return b**(1/self.weight) if additive else b

    def __str__(self):
        return (f"{self.name} (taps {', '.join(map(str, self.taps))}): holds {100*self.holds:.1f}% (z {self.z:.1f}), "
                f"weight {self.weight}: data bias {self.data_bias(True):.3f} per bit if additive, "
                f"{self.data_bias(False):.3f} if feedthrough")

    __repr__ = __str__

def window_counts(bits, L):
    """Counts of the (L+1)-bit windows of bits, as an index whose bit j is the bit at delay j (bit 0 the newest)"""
    b = np.asarray(bits, dtype=np.int64) & 1
    n = len(b) - L
    idx = np.zeros(n, dtype=np.int64)
    for j in range(L + 1):
        idx |= b[L - j:L - j + n] << j
    return np.bincount(idx, minlength=1 << (L + 1)), n

def pmod(a, b):
    """a mod b over GF(2) (ints as polynomials)"""
    db = b.bit_length()
    while a and a.bit_length() >= db:
        a ^= b << (a.bit_length() - db)
    return a

def pmul(a, b):
    """a b over GF(2)"""
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out

def gcd(*polys):
    """Greatest common divisor over GF(2) of polynomials as ints"""
    g = 0
    for p in polys:
        a, b = g, int(p)
        while b:
            a, b = b, pmod(a, b)
        g = a
    return g

def soft_z(t, taps):
    """The soft statistic of a check over signed reliabilities t = tanh(LLR/2) (s's +-1 expected values): sum over
    positions of the product over its taps, over its own standard deviation (a check that holds makes it large)"""
    t = np.asarray(t, dtype=np.float64)
    L = max(taps)
    N = len(t)
    p = np.ones(N - L)
    for j in taps:
        p = p*t[L - j:N - j]
    return float(abs(p.sum())/np.sqrt(max(np.sum(p**2), 1e-300))), float(np.sign(p.sum()))

def search(bits, max_degree=20, pfa=1e-6, top=8, llr=None, shortlist=256):
    """Checks f(D) s = 0 with a constant term, of degree up to max_degree (memory: 2^(max_degree+1) counts), significant at
    pfa over all of them, multiples of a stronger smaller one dropped: Recurrences, smallest degree first among the
    strongest.

    llr: soft values of the bits (log P(0)/P(1); bits may then be None): the hard transform shortlists the `shortlist`
    strongest candidates, significant or not, and each is scored by the product of its own taps' tanh(LLR/2) (2-4 dB more
    than hard decisions: a check over 3 taps no longer pays for errors in the 18 other bits of a window)"""
    if llr is not None:
        llr = np.asarray(llr, dtype=np.float64)
        bits = (llr < 0).astype(np.uint8)
    if max_degree > 24:
        raise ValueError("max_degree up to 24 (2^25 counts); a sparse search (rpps.analysis.scrambler) goes further")
    L = int(max_degree)
    counts, n = window_counts(bits, L)
    w = fwht(counts)
    w[0] = 0
    w[::2] = 0 # No constant term: a shift of a smaller check (or nothing)
    z = np.abs(w)/np.sqrt(n)
    zmin = np.sqrt(2*np.log((1 << L)/pfa))
    if llr is not None:
        t = np.tanh(np.clip(llr, -30, 30)/2)
        cand = np.argsort(-z)[:shortlist]
        z = np.zeros_like(z)
        for v in cand:
            v = int(v)
            taps = [j for j in range(L + 1) if v >> j & 1]
            zz, sg = soft_z(t, taps)
            z[v] = zz
            w[v] = sg*abs(w[v]) # the sign (holds or fails) from the soft sum; the size for `bias` stays the hard one's
    idx = np.flatnonzero(z >= zmin)
    idx = idx[np.argsort(-z[idx])][:512]
    out = []
    # Smallest degree first among checks of about the strength of the best: then a multiple is never taken first
    for v in sorted(idx, key=lambda v: (-round(np.log(z[v])/0.1), int(v).bit_length())):
        v = int(v)
        if any(pmod(v, r.poly) == 0 for r in out):
            continue
        taps = tuple(j for j in range(L, 0, -1) if v >> j & 1)
        b = float(w[v]/n)
        out.append(Recurrence(poly=v, taps=taps, degree=v.bit_length() - 1, weight=len(taps) + 1, holds=(1 + b)/2,
                              z=float(z[v]), bias=b))
        if len(out) >= top:
            break
    out.sort(key=lambda r: (-round(np.log(r.z)/0.1), r.degree))
    return out

def descramble_feedthrough(bits, poly):
    """d = f(D) s: the data under a feedthrough scrambler (the first `degree` bits are not fully descrambled)"""
    s = np.asarray(bits, dtype=np.uint8)
    d = s.copy()
    for j in range(1, int(poly).bit_length()):
        if poly >> j & 1:
            d[j:] ^= s[:-j]
    return d

def _square(poly):
    """f(D)^2 = f(D^2) over GF(2)"""
    out = 0
    for j in range(poly.bit_length()):
        if poly >> j & 1:
            out |= 1 << (2*j)
    return out

class Keystream:
    """Result of `keystream`: bits (the LFSR sequence, regenerated over the whole stream), agree (the share of the
    decoded bits it matches), data (the stream with it taken off), checks (share of the checks holding after decoding)"""
    __slots__ = ("bits", "agree", "data", "checks")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        return (f"keystream regenerated: matches {100*self.agree:.1f}% of the decoded bits, checks hold "
                f"{100*self.checks:.1f}%")

    __repr__ = __str__

def keystream(bits, poly, data_bias=None, squarings=4, iters=30, alpha=0.8):
    """An additive scrambler's keystream from the scrambled stream, its polynomial known (`search` finds it) and the data
    under it biased (idle fill, mostly-constant fields): the fast correlation attack (Meier & Staffelbach 1989). Every bit
    is the keystream seen through a binary channel whose crossover is the data's minority share; the keystream obeys
    f(D) and its squares f(D^2^i), sparse checks at every position; min-sum over them decodes it, and the whole sequence is
    regenerated from the most reliable window of `degree` decoded bits. data_bias: 2 P(data = common value) - 1 (default
    from how often f holds: (2 holds - 1)^(1 / weight)). The data's common value is taken as 0 (flip the result if not).
    On 20000 bits it is exact down to data 60/40 for DVB's 1 + x^14 + x^15 and 70/30 for a five-term polynomial (more
    taps, weaker checks)"""
    s = np.asarray(bits, dtype=np.uint8)
    n = len(s)
    L = int(poly).bit_length() - 1
    w = bin(int(poly)).count("1")
    if data_bias is None:
        d = descramble_feedthrough(s, poly)[L:] # f(D) s = f(D) data for an additive scrambler
        b_w = abs(2*np.mean(d == 0) - 1)
        data_bias = max(min(b_w**(1/w), 0.999), 1e-3)
    prior = np.log((1 + data_bias)/(1 - data_bias))*(1 - 2.0*s) # LLR of k = s (data 0 most likely)
    polys = [int(poly)]
    for _ in range(squarings):
        if polys[-1].bit_length() - 1 > n//4:
            break
        polys.append(_square(polys[-1]))
    checks = []
    for p in polys:
        taps = np.array([j for j in range(p.bit_length()) if p >> j & 1])
        t = np.arange(taps.max(), n)
        checks.append(t[:, None] - taps[None, :]) # (instances, w): positions of each check instance
    msgs = [np.zeros(c.shape) for c in checks]
    total = prior.copy()
    for _ in range(iters):
        new = prior.copy()
        for i, idx in enumerate(checks):
            q = total[idx] - msgs[i]
            a = np.abs(q)
            sgn = np.where(q < 0, -1.0, 1.0)
            prod = np.prod(sgn, axis=1, keepdims=True)
            o = np.argsort(a, axis=1)
            m1 = np.take_along_axis(a, o[:, :1], axis=1)
            m2 = np.take_along_axis(a, o[:, 1:2], axis=1)
            mag = np.where(np.arange(idx.shape[1])[None, :] == o[:, :1], m2, m1)
            msgs[i] = alpha*prod*sgn*mag
            np.add.at(new, idx.ravel(), msgs[i].ravel())
        total = new
    k = (total < 0).astype(np.uint8)
    held = float(np.mean([np.mean(np.bitwise_xor.reduce(k[c], axis=1) == 0) for c in checks]))
    # Regenerate from the most reliable L consecutive decoded bits, forward and back through the recurrence
    rel = np.convolve(np.abs(total), np.ones(L), mode="valid")
    a0 = int(np.argmax(rel))
    taps = [j for j in range(1, L + 1) if poly >> j & 1]
    out = np.zeros(n, np.uint8)
    out[a0:a0 + L] = k[a0:a0 + L]
    for t in range(a0 + L, n): # k[t] = sum over taps k[t - j]
        v = 0
        for j in taps:
            v ^= out[t - j]
        out[t] = v
    for t in range(a0 - 1, -1, -1): # k[t + L] = k[t] + sum over other taps: k[t] = k[t + L] + sum_{j<L} k[t + L - j]
        v = out[t + L]
        for j in taps:
            if j != L:
                v ^= out[t + L - j]
        out[t] = v
    return Keystream(bits=out, agree=float(np.mean(out == k)), data=s ^ out, checks=held)
