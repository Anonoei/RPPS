import numpy as np

from ... import _accel
from ._code import _code, register, word_errors
from ...math.gf2 import systematic, to_checks

def gallager(n, wc=3, wr=6, seed=0):
    """Regular (wc, wr) parity-check matrix by Gallager's construction. n must be a multiple of wr"""
    if n % wr:
        raise ValueError(f"n={n} must be a multiple of wr={wr}")
    rng = np.random.default_rng(seed)
    rows = n // wr
    base = np.zeros((rows, n), dtype=bool)
    for r in range(rows):
        base[r, r*wr:(r+1)*wr] = True
    return np.vstack([base] + [base[:, rng.permutation(n)] for _ in range(wc-1)])


@register("ldpc")
class ldpc(_code):
    """Linear code defined by sparse parity checks, with a normalised min-sum belief propagation decoder

    n: codeword length
    checks: for each parity check, the codeword positions it covers
    k/systematic: the message length, and where the message sits in the codeword: "head", "tail" or a list of
        positions. When omitted both come from GF(2) elimination, which needs a dense H (small codes only)
    iters/alpha: maximum iterations / min-sum normalisation
    decode() takes hard bits or LLRs (log(P(0)/P(1)), positive means 0) for whole codewords, returns the message.
    After a decode, `converged` says which codewords satisfied every parity check.
    With rpps._native the codewords are decoded on several threads, each stopping as soon as it converges (the
    numpy decoder iterates the whole batch until every codeword has).
    """
    soft = True

    @classmethod
    def from_config(cls, log, obj):
        """Parity checks as dense rows ("h"), sparse lists ("checks" + "n"), a DVB-S2 style address "table", or a "gallager" construction"""
        from .ira import ira_checks
        if "table" in obj:
            n, checks = obj["n"], ira_checks(obj["n"], obj["k"], obj["table"], obj.get("group", 360))
        elif "checks" in obj:
            n, checks = obj["n"], obj["checks"]
        else:
            h = np.array(obj["h"], dtype=bool) if "h" in obj else gallager(**obj["gallager"])
            n, checks = h.shape[1], to_checks(h)
        return cls(log, n, checks, obj.get("k"), obj.get("systematic"), obj.get("iters", 50), obj.get("alpha", 0.8))

    def __init__(self, log, n, checks, k=None, systematic_pos=None, iters=50, alpha=0.8):
        self.n = n
        self.checks = [np.asarray(c, dtype=np.int64) for c in checks]
        self._G = None
        if systematic_pos is None and k is None:
            self._G, info = systematic(self.dense())
            k = len(info)
        elif isinstance(systematic_pos, str):
            info = np.arange(k) if systematic_pos == "head" else np.arange(n-k, n)
        elif systematic_pos is None:
            raise ValueError("k needs `systematic` (head, tail or positions), or neither")
        else:
            info = np.asarray(systematic_pos, dtype=np.int64)
            k = len(info)
        self.info = info
        super().__init__(log, None, k, n)
        self.iters = iters
        self.alpha = alpha
        self.converged = None

        M = len(self.checks)
        self.M, self.N = M, n
        deg = np.array([len(c) for c in self.checks])
        self.deg = deg.astype(np.int64)
        self.var = np.concatenate(self.checks) # Edges, grouped by check
        self.dmax = int(deg.max())
        self.cstart = np.concatenate(([0], np.cumsum(deg)[:-1]))
        self.slot = np.full((M, self.dmax), -1, dtype=np.int64) # Edge index per (check, slot)
        for d in range(self.dmax):
            ok = deg > d
            self.slot[ok, d] = self.cstart[ok] + d
        self.valid = self.slot >= 0
        self.order = np.argsort(self.var, kind="stable") # Edges grouped by variable
        vdeg = np.bincount(self.var, minlength=n)
        self.vend = np.cumsum(vdeg)
        self.vstart = self.vend - vdeg

    def dense(self):
        H = np.zeros((len(self.checks), self.n), dtype=bool)
        for i, c in enumerate(self.checks):
            H[i, c] = True
        return H

    @property
    def H(self):
        return self.dense()

    @property
    def G(self):
        if self._G is None:
            self._G, _ = systematic(self.dense())
            if len(self._G) != self.num or not np.array_equal(np.sort(_), np.sort(self.info)):
                raise ValueError("Message positions in the definition do not match the checks")
        return self._G

    def encode(self, bits):
        bits = np.asarray(bits).astype(np.uint8)
        pad = (-len(bits)) % self.num
        if pad:
            bits = np.append(bits, np.zeros(pad, dtype=np.uint8))
        u = bits.reshape(-1, self.num)
        if self._G is None and not isinstance(self.info, str):
            G, info = systematic(self.dense())
            # Re-express so the message lands where the definition says it does
            if not np.array_equal(info, self.info):
                G = self._reposition(G, info)
            self._G = G
        return ((u @ self._G.astype(np.uint8)) % 2).reshape(-1).astype(bool)

    def _reposition(self, G, info):
        """Row-reduce G so its identity columns are self.info"""
        G = G.astype(np.uint8)
        sub = G[:, self.info]
        # Solve sub @ X = I over GF(2)
        K = len(self.info)
        A = np.concatenate((sub, np.eye(K, dtype=np.uint8)), axis=1)
        for c in range(K):
            piv = np.nonzero(A[c:, c])[0]
            if len(piv) == 0:
                raise ValueError("Message positions are not an information set of this code")
            piv = piv[0] + c
            A[[c, piv]] = A[[piv, c]]
            rows = np.nonzero(A[:, c])[0]
            rows = rows[rows != c]
            A[rows] ^= A[c]
        return ((A[:, K:] @ G) % 2).astype(bool)

    def syndrome(self, words):
        """Parity check failures for words (B, N), without a dense H"""
        w = np.asarray(words, dtype=np.uint8)
        return (np.add.reduceat(w[:, self.var], self.cstart, axis=1) % 2).astype(np.uint8)

    def decode_codewords(self, llr):
        """Min-sum on (B, N) LLRs. Returns hard decisions (B, N)"""
        if _accel.native is not None:
            hard, self.converged = _accel.native.ldpc_minsum(
                np.ascontiguousarray(llr, dtype=np.float64), self.var, self.cstart.astype(np.int64), self.deg,
                self.order.astype(np.int64), self.vstart.astype(np.int64), self.vend.astype(np.int64), int(self.iters), float(self.alpha))
            return hard
        B = llr.shape[0]
        ch = llr
        v2c = ch[:, self.var].copy() # (B, E)
        total = ch.copy()
        done = np.zeros(B, dtype=bool)
        slot0 = np.maximum(self.slot, 0)
        flat = self.slot[self.valid]
        for _ in range(self.iters):
            # Check nodes: sign product and the two smallest magnitudes, excluding each edge itself
            m = np.where(self.valid[None], v2c[:, slot0], np.inf) # (B, M, dmax)
            neg = (m < 0) & self.valid[None]
            sgn = np.where(neg.sum(axis=2, keepdims=True) % 2 == 1, -1.0, 1.0) * np.where(neg, -1.0, 1.0)
            mag = np.abs(m)
            idx1 = np.argmin(mag, axis=2)
            min1 = np.take_along_axis(mag, idx1[..., None], axis=2)
            mag2 = mag.copy()
            np.put_along_axis(mag2, idx1[..., None], np.inf, axis=2)
            min2 = mag2.min(axis=2, keepdims=True)
            is_min = np.arange(self.dmax)[None, None, :] == idx1[..., None]
            c2v_slot = self.alpha * sgn * np.where(is_min, min2, min1) # (B, M, dmax)
            c2v = np.zeros_like(v2c)
            c2v[:, flat] = c2v_slot[:, self.valid]
            # Variable nodes
            cs = np.concatenate((np.zeros((B, 1)), np.cumsum(c2v[:, self.order], axis=1)), axis=1)
            total = ch + cs[:, self.vend] - cs[:, self.vstart]
            v2c = total[:, self.var] - c2v
            done = ~self.syndrome(total < 0).any(axis=1)
            if done.all():
                break
        self.converged = done
        return total < 0

    def decode(self, bits, soft=None):
        x = np.asarray(bits)
        if soft is None:
            soft = x.dtype.kind == "f"
        llr = x.astype(np.float64) if soft else 2.0*(1.0 - 2.0*x.astype(np.float64)) # Hard bit: LLR +-2
        llr = llr[:len(llr)//self.N*self.N].reshape(-1, self.N)
        cw = self.decode_codewords(llr)
        self.errors = word_errors(llr < 0, cw, self.converged)
        return cw[:, self.info].reshape(-1)
