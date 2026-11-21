"""Polar codes (Arikan): encoding, successive cancellation (SC) and list decoding (SCL), CRC-aided

    {"base": "block", "type": "polar", "n": 256, "k": 128,
     "frozen": [...]                       the frozen positions of u, or
     "order": "pw" | "bhattacharyya",      the N - K least reliable positions of a construction (polarization weight,
     "design_db": 0.0,                     the beta-expansion NR follows; Bhattacharyya parameters at Es/N0 design_db)
     "list": 8,                            paths kept (1: successive cancellation)
     "crc": "crc/crc16_xmodem"}            CRC-aided: the last `width` information bits carry a CRC of the others, and the
                                           list decoder picks the best path whose CRC checks (k counts the CRC bits)

A codeword is x = u G, G = F^(x n) with F = [[1, 0], [1, 1]] (no bit reversal); u holds the message at the information
positions (in increasing order) and 0 at the frozen ones. G is its own inverse.

Decoding works on LLRs (log P(0)/P(1)), hard bits becoming +-1. SC decides each u_i in order from its LLR given the
decisions before it; SCL keeps the `list` most likely paths, each scored by the penalty sum of log(1 + e^-(1-2u)L)
(Balatsoukas-Stimming, Parizi, Burg, "LLR-based successive cancellation list decoding of polar codes", IEEE TSP 2015).
Every frame of a call is decoded at once (arrays over frames and paths). `errors` per codeword: bits the chosen codeword
differs from the hard decisions, or -1 when a CRC-aided decoder found no path that checks.

`sc(llr, decide)`, `transform`, `pw_order` and `bhattacharyya_order` are what the blind tools build on
(rpps.analysis.code.block.polar).
"""
import numpy as np

from ._code import _code, register, as_llr, word_errors

def transform(N):
    """G = F^(x n) for N = 2^n, uint8"""
    n = int(np.log2(N)) if N > 0 else -1
    if n < 0 or 2**n != N:
        raise ValueError(f"polar: N {N} is not a power of 2")
    G = np.array([[1]], np.uint8)
    F = np.array([[1, 0], [1, 1]], np.uint8)
    for _ in range(n):
        G = np.kron(G, F)
    return G

def encode_u(u):
    """x = u G for rows u (..., N), by the butterflies (N log N XORs)"""
    x = np.array(u, dtype=np.uint8, copy=True)
    N = x.shape[-1]
    h = 1
    while h < N:
        y = x.reshape(*x.shape[:-1], N//(2*h), 2, h)
        y[..., 0, :] ^= y[..., 1, :]
        h *= 2
    return x

def pw_order(N):
    """Positions from least to most reliable by polarization weight (sum of 2^(j/4) over the index's set bits j: the
    beta-expansion 3GPP's NR sequence follows; SNR-independent)"""
    n = int(np.log2(N))
    i = np.arange(N)
    pw = sum(((i >> j) & 1)*2**(j/4) for j in range(n))
    return np.argsort(pw, kind="stable")

def bhattacharyya_order(N, design_db=0.0):
    """Positions from least to most reliable by the Bhattacharyya parameter on a BI-AWGN channel at Es/N0 design_db
    (Z = exp(-Es/N0) a channel, then 2Z - Z^2 / Z^2 down the tree, the index's top bit first)"""
    n = int(np.log2(N))
    z = np.full(N, np.exp(-10**(design_db/10)))
    for lev in range(n - 1, -1, -1):
        bit = (np.arange(N) >> lev) & 1
        z = np.where(bit == 1, z**2, 2*z - z**2)
    return np.argsort(-z, kind="stable")

ORDERS = {"pw": lambda N, db: pw_order(N), "bhattacharyya": bhattacharyya_order}

def box(a, b):
    """LLR of the parity of two bits (exact box-plus)"""
    return 2*np.arctanh(np.clip(np.tanh(np.clip(a, -30, 30)/2)*np.tanh(np.clip(b, -30, 30)/2), -1 + 1e-15, 1 - 1e-15))

def sc(llr, decide):
    """Successive-cancellation decoding of frames (W, N) of LLRs, every frame at once. decide(i, L) -> (W,) bits: u_i of
    every frame from its SC LLR L (W,) (a frozen position returns zeros; this is where a blind test looks at it).
    Returns (u (W, N) uint8, Lu (W, N): each u_i's LLR given the u before it as decided)"""
    L = np.asarray(llr, dtype=np.float64)
    W, N = L.shape
    transform(N) # checks N
    u = np.zeros((W, N), np.uint8)
    Lu = np.zeros((W, N))

    def node(Ln, i0):
        m = Ln.shape[1]
        if m == 1:
            Lu[:, i0] = Ln[:, 0]
            b = np.asarray(decide(i0, Ln[:, 0]), dtype=np.uint8)
            u[:, i0] = b
            return b[:, None]
        h = m//2
        a, c = Ln[:, :h], Ln[:, h:]
        v = node(box(a, c), i0) # u_a G' = x_first + x_second
        w = node(c + (1 - 2.0*v)*a, i0 + h) # u_b G' = x_second
        return np.concatenate((v ^ w, w), axis=1)

    node(L, 0)
    return u, Lu

def _take(a, perm):
    """Rows of each frame's paths in a (W, P, ...) picked by perm (W, P')"""
    return np.take_along_axis(a, perm.reshape(perm.shape + (1,)*(a.ndim - 2)), axis=1)

def scl(llr, frozen, size):
    """List decoding of frames (W, N) of LLRs: the `size` best paths of each frame, best first. frozen: (N,) bool.
    Returns (u (W, P, N) uint8, metric (W, P): each path's penalty, smaller is likelier)"""
    L = np.asarray(llr, dtype=np.float64)
    W, N = L.shape
    transform(N)
    frozen = np.asarray(frozen, dtype=bool)
    state = {"u": np.zeros((W, 1, N), np.uint8), "pm": np.zeros((W, 1))}

    def node(Ln, i0):
        """Ln (W, P, m): returns (x bits (W, P', m), perm (W, P'): the paths it ended with, as rows of its input's)"""
        W_, P, m = Ln.shape
        if m == 1:
            Li, pm = Ln[:, :, 0], state["pm"]
            if frozen[i0]:
                state["pm"] = pm + np.logaddexp(0.0, -Li)
                state["u"][:, :, i0] = 0
                return np.zeros((W_, P, 1), np.uint8), np.broadcast_to(np.arange(P), (W_, P))
            cand = np.concatenate((pm + np.logaddexp(0.0, -Li), pm + np.logaddexp(0.0, Li)), axis=1) # (W, 2P): u = 0, 1
            keep = min(2*P, size)
            idx = np.argsort(cand, axis=1, kind="stable")[:, :keep]
            perm, bit = idx % P, (idx // P).astype(np.uint8)
            state["pm"] = np.take_along_axis(cand, idx, axis=1)
            state["u"] = _take(state["u"], perm)
            state["u"][:, :, i0] = bit
            return bit[:, :, None], perm
        h = m//2
        a, c = Ln[:, :, :h], Ln[:, :, h:]
        v, p1 = node(box(a, c), i0)
        a, c = _take(a, p1), _take(c, p1)
        w, p2 = node(c + (1 - 2.0*v)*a, i0 + h)
        v = _take(v, p2)
        return np.concatenate((v ^ w, w), axis=2), np.take_along_axis(p1, p2, axis=1)

    node(L[:, None, :], 0)
    order = np.argsort(state["pm"], axis=1, kind="stable")
    return _take(state["u"], order), np.take_along_axis(state["pm"], order, axis=1)

@register("polar")
class polar(_code):
    """A polar code (see the module): `info` the information positions, `crc` the CRC the last bits of the message carry"""
    soft = True

    def __init__(self, log, n, k, frozen, size=1, crc=None):
        self.N, self.K = int(n), int(k)
        transform(self.N)
        frozen = np.sort(np.asarray(frozen, dtype=np.int64))
        if len(frozen) != self.N - self.K or len(np.unique(frozen)) != len(frozen) or (len(frozen) and not 0 <= frozen[0] <= frozen[-1] < self.N):
            raise ValueError(f"polar ({self.N}, {self.K}): needs {self.N - self.K} distinct frozen positions below {self.N}")
        self.frozen = np.zeros(self.N, bool)
        self.frozen[frozen] = True
        self.info = np.flatnonzero(~self.frozen)
        self.size, self.crc = int(size), crc
        self.width = 0 if crc is None else crc.width
        if self.width >= self.K:
            raise ValueError(f"polar ({self.N}, {self.K}): a {self.width}-bit CRC leaves no message")
        super().__init__(log, f"polar_{self.N}_{self.K}", self.K - self.width, self.N)

    @classmethod
    def from_config(cls, log, obj):
        n, k = int(obj["n"]), int(obj["k"])
        if "frozen" in obj:
            frozen = obj["frozen"]
        else:
            order = obj.get("order", "pw")
            if order not in ORDERS:
                raise ValueError(f"polar: order {order!r}, one of {sorted(ORDERS)} (or give \"frozen\")")
            frozen = ORDERS[order](n, float(obj.get("design_db", 0.0)))[:n - k]
        crc = None
        if obj.get("crc"):
            from ... import _config
            crc = _config.load_ref(obj["crc"]).encoder
        return cls(log, n, k, frozen, obj.get("list", 1), crc)

    def encode(self, bits):
        msg = np.asarray(bits).astype(np.uint8).reshape(-1, self.num)
        if self.crc is not None:
            msg = np.array([self.crc.encode(m) for m in msg], dtype=np.uint8).reshape(len(msg), self.K)
        u = np.zeros((len(msg), self.N), np.uint8)
        u[:, self.info] = msg
        return encode_u(u).reshape(-1).astype(bool)

    def _checks(self, words):
        """Whether each message's CRC field matches (words (..., K) information bits)"""
        flat = words.reshape(-1, self.K)
        return np.array([self.crc.syndrome(w) == 0 for w in flat]).reshape(words.shape[:-1])

    def decode(self, bits):
        x = np.asarray(bits)
        llr = as_llr(x[:len(x)//self.N*self.N]).reshape(-1, self.N)
        W = len(llr)
        if W == 0:
            self.errors = np.zeros(0, int)
            return np.zeros(0, bool)
        ok = np.ones(W, bool)
        if self.size <= 1 and self.crc is None:
            u, _ = sc(llr, lambda i, L: np.zeros(len(L), np.uint8) if self.frozen[i] else (L < 0).astype(np.uint8))
        else:
            paths, _ = scl(llr, self.frozen, max(self.size, 1))
            if self.crc is None:
                u = paths[:, 0]
            else:
                good = self._checks(paths[:, :, self.info])
                pick = np.where(good.any(axis=1), np.argmax(good, axis=1), 0)
                ok = good.any(axis=1)
                u = paths[np.arange(W), pick]
        self.errors = word_errors(llr < 0, encode_u(u), ok)
        return u[:, self.info[:self.num]].reshape(-1).astype(bool)
