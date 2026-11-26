"""Trellis-coded modulation found blind: the parity check a trellis code puts on the constellation's labels

    found = rp.analysis.code.conv.tcm.search(symbols, 8)            # locked symbols (any phase), an 8-point PSK
    print(found[0])                                       # e.g. "h = (11, 02, 04) octal, natural labels, rotation 3:
                                                          #  Ungerboeck 8 states, z 41.0"

A trellis code over an M-point constellation with m = log2 M label bits (z^0 the least significant) satisfies a parity
check (Ungerboeck 1982):  sum_i h^i(D) z^i(D) = 0  over GF(2), every symbol. Ungerboeck's systematic feedback codes are
written by those h^i (octal); a pragmatic code (a rate-1/2 convolutional code on the low bits, the rest uncoded) has one
too (its two generators, crossed); so has any other linear trellis code. Over a window of W = nu + 1 symbols a check is a
W m-bit vector, and with hard decisions it holds more often than not: (1 - 2p)^weight of the time, p the label bit error
rate. Counting the windows of the stream by value and taking the Walsh-Hadamard transform of the counts gives, for every
one of the 2^(W m) vectors at once, how many windows it holds on minus how many it fails (the WHT-BM method of rate k/n
convolutional code recovery). A check stands out against the sqrt(N) of chance.

The labels depend on what is not known: the carrier phase (M rotations, and conjugation for an inverted spectrum) and
the labelling (natural, i.e. set partitioning, for Ungerboeck codes; Gray for pragmatic ones; or the constellation's
own). Every combination is tried; the label permutation does the rotation, so decisions are made once.
"""
import numpy as np

from ....math.gf2 import fwht, rank_ints

# Ungerboeck's 8PSK rate 2/3 codes (1987, Part II, table III): states -> (h0, h1, h2) octal
UNGERBOECK_8PSK = {4: (0o5, 0o2, 0o0), 8: (0o11, 0o2, 0o4), 16: (0o23, 0o4, 0o16), 32: (0o45, 0o16, 0o34),
                   64: (0o103, 0o30, 0o66), 128: (0o277, 0o54, 0o122), 256: (0o435, 0o72, 0o130)}
# Rate 1/2 codes on QPSK / 4-level labels, (h0, h1): the same table for 4 points (Ungerboeck 1987, table I)
UNGERBOECK_4 = {4: (0o5, 0o2), 8: (0o13, 0o4), 16: (0o23, 0o6), 32: (0o45, 0o10), 64: (0o103, 0o024)}

class Check:
    """A parity check found, and how well it holds

    h (one polynomial per label bit, z^0 first, as ints whose bit d is the delay-d tap), nu
    (memory), labels ('natural', 'gray' or 'own'), rotation (points the decisions were turned by), conj, z (its bias
    over chance), holds (fraction of windows it holds on), match (the known code it is, or None), expected (how often a
    code's check of these taps would hold, given the label bit errors the symbols' noise makes), code (holds at least
    CODE of the way from chance to expected: a code; else the labels are only correlated, by ISI, oversampling or the
    wrong modulation)"""
    __slots__ = ("h", "nu", "labels", "rotation", "conj", "z", "holds", "match", "expected", "code")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def octal(self):
        return tuple(f"{p:o}" for p in self.h)

    def __str__(self):
        m = f": {self.match}" if self.match else ""
        if not self.code:
            m += f"; correlated labels, not a code (a code's check would hold {100*self.expected:.1f}%)"
        c = ", conjugated" if self.conj else ""
        return (f"h = ({', '.join(self.octal)}) octal (z^0 first), nu {self.nu}, {self.labels} labels, rotation "
                f"{self.rotation}{c}, holds {100*self.holds:.1f}%, z {self.z:.1f}{m}")

    __repr__ = __str__

def decisions(y, M):
    """Nearest of M PSK points at angles 2 pi k / M + offset (the offset taken from the M-th power): k, 0..M-1"""
    y = np.asarray(y, dtype=np.complex128)
    off = np.angle(np.sum(y**M))/M if M > 1 else 0.0
    # Points at off + 2 pi k/M; for the usual offsets (0 or pi/M) this is the nearest point
    return np.round((np.angle(y) - off)/(2*np.pi/M)).astype(np.int64) % M

def labellings(M, own=None):
    """{name: label of the point at angle index k}"""
    k = np.arange(M)
    out = {"natural": k, "gray": k ^ (k >> 1)}
    if own is not None:
        out["own"] = np.asarray(own, dtype=np.int64)
    return out

def label_windows(z, m, W):
    """Counts of every W-symbol window of m-bit labels z, as an index whose bit (W-1-d) m + i is z^i at delay d"""
    z = np.asarray(z, dtype=np.int64)
    idx = np.zeros(len(z) - W + 1, dtype=np.int64)
    for j in range(W): # j = W-1-d: oldest symbol of the window at j = 0
        idx |= z[j:len(z) - W + 1 + j] << (j*m)
    return np.bincount(idx, minlength=1 << (W*m)), len(idx)

def _polys(v, m, W):
    """Window vector -> per-label-bit polynomials (bit d = delay d), shifted so the smallest delay used is 0"""
    h = [0]*m
    for j in range(W):
        d = W - 1 - j
        for i in range(m):
            if v >> (j*m + i) & 1:
                h[i] |= 1 << d
    low = min(((p & -p).bit_length() - 1 for p in h if p), default=0)
    return [p >> low for p in h]

def _vector(h, m, W, shift=0):
    v = 0
    for i, p in enumerate(h):
        for d in range(W):
            if p >> d & 1 and d + shift < W:
                v |= 1 << ((W - 1 - d - shift)*m + i)
            elif p >> d & 1:
                return None
    return v

def _match(h, m):
    table = UNGERBOECK_8PSK if m == 3 else UNGERBOECK_4 if m == 2 else {}
    for states, ref in table.items():
        if list(ref) == list(h):
            return f"Ungerboeck {states} states"
    return None

CODE = 0.6 # A check's bias over the bias a code would have at this noise: more is a code

def label_errors(y, M, labs, k, seed=0):
    """Per label bit, the probability a hard decision gets it wrong at the noise of these symbols: {labelling: array(m)}"""
    from ....sync import quality
    m = int(round(np.log2(M)))
    off = np.angle(np.sum(np.asarray(y, dtype=np.complex128)**M))/M if M > 1 else 0.0
    pts = np.exp(1j*(2*np.pi*np.arange(M)/M + off))
    sigma = quality.measure(y, pts).sigma
    rng = np.random.default_rng(seed)
    n = 20000
    t = rng.integers(0, M, n)
    z = pts[t] + sigma*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
    d = np.round((np.angle(z) - off)/(2*np.pi/M)).astype(np.int64) % M
    out = {}
    for name, lab in labs.items():
        diff = lab[t] ^ lab[d]
        out[name] = np.array([np.mean(diff >> i & 1) for i in range(m)])
    return out

def _expected(h, p):
    """How often a check with taps h holds on labels with bit error rates p (independent errors): (1 + prod(1-2p))/2"""
    b = 1.0
    for i, poly in enumerate(h):
        b *= (1 - 2*p[i])**bin(poly).count("1")
    return (1 + b)/2

def _label_tanh(y, M, lab, r, c):
    """tanh(LLR/2) of each label bit (N, m) for one variant (labelling, rotation r, conjugated c), max-log over the points
    at the measured noise: +1 sure 0, -1 sure 1"""
    from ....sync import quality
    m = int(round(np.log2(M)))
    y = np.asarray(y, dtype=np.complex128)
    off = np.angle(np.sum(y**M))/M if M > 1 else 0.0
    pts = np.exp(1j*(2*np.pi*np.arange(M)/M + off))
    q = quality.measure(y, pts)
    yn = y/np.sqrt(np.mean(np.abs(y)**2))
    d2 = np.abs(yn[:, None] - pts[None, :])**2 # (N, M)
    p = np.arange(M)
    label = lab[((-p if c else p) + r) % M] # the label point p stands for in this variant
    T = np.zeros((len(y), m))
    for i in range(m):
        one = (label >> i) & 1 == 1
        llr = (d2[:, one].min(axis=1) - d2[:, ~one].min(axis=1))/(2*q.sigma**2)
        T[:, i] = np.tanh(np.clip(llr, -60, 60)/2)
    return T

def _soft_z(T, v, m, W):
    """The soft statistic of check vector v (window bit (W-1-d) m + i: label bit i at delay d) over label tanh values T"""
    N = len(T)
    p = np.ones(N - W + 1)
    for j in range(W):
        d = W - 1 - j
        for i in range(m):
            if v >> (j*m + i) & 1:
                p = p*T[W - 1 - d:N - d, i]
    return float(abs(p.sum())/np.sqrt(max(np.sum(p**2), 1e-300)))

def search(y, M, max_nu=5, own=None, conj=True, labelling=None, pfa=1e-6, top=4, correlations=False, soft=False,
           shortlist=32):
    """Parity checks of a trellis code on M-PSK symbols y (locked to within a rotation). max_nu: largest memory tried
    (windows of max_nu + 1 symbols: (max_nu + 1) log2 M bits, 2^that candidates). own: the constellation's labels by
    angle index, tried as well. Returns Checks, most significant first (empty: no code of that memory on these labels).
    A check that holds well short of what a code's would at this noise is a correlation of the labels (ISI, a wrong rate or
    modulation), not a code: left out unless `correlations`, and marked (Check.code False).
    soft: each variant's `shortlist` strongest hard candidates, significant or not, are scored by the product of their
    taps' label-bit tanh(LLR/2) (max-log at the measured noise): a few dB more than hard decisions"""
    m = int(round(np.log2(M)))
    W = max_nu + 1
    if W*m > 24:
        raise ValueError(f"{W*m}-bit windows: keep (max_nu + 1) log2 M within 24")
    k = decisions(y, M)
    labs = labellings(M, own)
    if labelling is not None:
        labs = {labelling: labs[labelling]}
    perr = label_errors(y, M, labs, k)
    variants = [(name, lab, r, c) for name, lab in labs.items() for r in range(M) for c in ((False, True) if conj else (False,))]
    tests = len(variants)*(1 << (W*m))
    zmin = np.sqrt(2*np.log(tests/pfa))
    found = []
    for name, lab, r, c in variants:
        kk = (-k if c else k) + r
        z = lab[kk % M]
        counts, N = label_windows(z, m, W)
        bias = fwht(counts)
        bias[0] = 0
        zs = bias/np.sqrt(N)
        if soft:
            T = _label_tanh(y, M, lab, r, c)
            for v in np.argsort(-np.abs(zs))[:shortlist]:
                zv = _soft_z(T, int(v), m, W)
                if zv >= zmin:
                    found.append((zv, int(v), name, r, c, float((N + bias[v])/(2*N)), N))
            continue
        best = np.argsort(-np.abs(zs))[:64]
        for v in best:
            if abs(zs[v]) < zmin:
                break
            found.append((abs(float(zs[v])), int(v), name, r, c, float((N + bias[v])/(2*N)), N))
    found.sort(key=lambda t: -t[0])
    out = []
    span = {} # per variant: vectors of the checks taken and their shifts
    for zv, v, name, r, c, holds, N in found:
        key = (name, r, c)
        have = span.setdefault(key, [])
        if have and rank_ints(have + [v]) == rank_ints(have):
            continue # A shift or sum of checks already taken
        h = _polys(v, m, W)
        nu = max(p.bit_length() for p in h) - 1
        hv = holds if holds >= 0.5 else 1 - holds
        exp = _expected(h, perr[name])
        out.append(Check(h=h, nu=nu, labels=name, rotation=r, conj=c, z=zv, holds=hv, match=_match(h, m), expected=exp,
                         code=(hv - 0.5) >= CODE*(exp - 0.5)))
        for s in range(W):
            u = _vector(h, m, W, s)
            if u is not None:
                have.append(u)
        if len(out) >= 4*top:
            break
    # Equivalent forms of one code turn up under several rotations / labellings with about the same z: put the one a
    # table names first, then the unconjugated, then the smallest rotation
    # (soft: a labelling's own LLRs score its equivalent forms a little differently: wider buckets)
    zb = lambda c: round(np.log(c.z)/(0.15 if soft else 0.05)) # 5% buckets
    out.sort(key=lambda c: (not c.code, -zb(c), c.match is None, c.conj, c.rotation))
    if not correlations:
        out = [c for c in out if c.code]
    return out[:top]

def definition(h, m, points=None):
    """The trellis code of check h as an `fsm` definition (rpps.code.types.fsm): the systematic feedback encoder whose
    upper m-1 label bits are the input (z^1 the first input bit) and z^0 is solved from sum_i h^i z^i = 0 (h[0] with taps
    at delay 0 and nu). The state is the delays the check reaches (z^0's last nu, each z^i's as far as its h^i does);
    the output word is the label (z^0 its last bit). points: the constellation in label order, to decode symbols"""
    h = [int(p) for p in h]
    h0 = h[0]
    nu = max(p.bit_length() for p in h) - 1
    if nu < 1 or not (h0 & 1 and h0 >> nu & 1):
        raise ValueError("h0 needs taps at delay 0 and nu")
    k = m - 1
    depth = [nu] + [max(h[i].bit_length() - 1, 0) for i in range(1, m)] # delays kept of each label bit
    shift = np.concatenate(([0], np.cumsum(depth)[:-1]))
    def step(state, u):
        hist = [[(state >> (shift[i] + d)) & 1 for d in range(depth[i])] for i in range(m)] # hist[i][d-1]: z^i at t-d
        z = [0] + [(u >> (k - i)) & 1 for i in range(1, m)]
        acc = 0
        for d in range(1, nu + 1):
            if h0 >> d & 1:
                acc ^= hist[0][d - 1]
        for i in range(1, m):
            for d in range(0, depth[i] + 1):
                if h[i] >> d & 1:
                    acc ^= z[i] if d == 0 else hist[i][d - 1]
        z[0] = acc
        new = 0
        for i in range(m):
            for d in range(depth[i]):
                b = z[i] if d == 0 else hist[i][d - 1]
                new |= b << (shift[i] + d)
        return new, sum(z[i] << i for i in range(m))
    # The states reachable from 0, numbered as found
    index, order, nxt, out = {0: 0}, [0], [], []
    while len(nxt) < len(order):
        s = order[len(nxt)]
        row_n, row_o = [], []
        for u in range(1 << k):
            n2, w = step(s, u)
            if n2 not in index:
                index[n2] = len(order)
                order.append(n2)
            row_n.append(index[n2])
            row_o.append(w)
        nxt.append(row_n)
        out.append(row_o)
    obj = {"base": "block", "type": "fsm", "comment": f"trellis code of the check h = {[oct(p) for p in h]}, {m}-bit labels",
           "k": k, "n": m, "next": nxt, "output": out, "start": 0, "end": None}
    if points is not None:
        pts = np.asarray(points, dtype=np.complex128)
        obj["symbols"] = {"bits": m, "points": [[float(p.real), float(p.imag)] for p in pts]}
    return obj

def encode(bits, h, m):
    """Labels of a systematic feedback trellis encoder for the check h (see `definition`): the upper m-1 label bits are
    bits (rows of m-1, z^1 first), z^0 is solved from sum_i h^i z^i = 0"""
    from ....code import init
    code = init("tcm", definition(h, m)).encoder
    sent = np.asarray(code.encode(np.asarray(bits, dtype=np.uint8).reshape(-1)), dtype=np.int64).reshape(-1, m)
    return (sent * (1 << np.arange(m - 1, -1, -1))).sum(axis=1)

