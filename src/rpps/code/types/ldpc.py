import numpy as np

from ... import _accel, _effort
from ._code import _code, register, word_errors, as_llr
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


_LAYERS = {} # (N, M, edges) -> the code's check layers: one colouring per code per process

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
            code = cls(log, n, checks, obj.get("k"), obj.get("systematic"), obj.get("iters", 50), obj.get("alpha", 0.8))
            if obj.get("systematic") == "head": # The accumulator encodes it directly; elimination would need a dense G
                code._ira = (obj["table"], obj.get("group", 360))
            return code
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

    _ira = None # (address table, group) of a code built from a DVB-S2 style table: encoded by its accumulator

    def encode(self, bits):
        bits = np.asarray(bits).astype(np.uint8)
        pad = (-len(bits)) % self.num
        if pad:
            bits = np.append(bits, np.zeros(pad, dtype=np.uint8))
        if self._ira is not None:
            from .ira import ira_encode
            return ira_encode(bits, self.n, self.num, *self._ira)
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
        """Min-sum on (B, N) LLRs. Returns hard decisions (B, N); `converged` says which codewords satisfied every check"""
        hard, self.converged = self.min_sum(llr)
        return hard

    def layers(self):
        """The checks in layers that share no variable (greedy colouring, rows in order), computed once per code: the layered
        schedule updates a layer at a time, each layer seeing what the ones before it found"""
        key = (self.N, self.M, hash(self.var.tobytes()), hash(self.deg.tobytes()))
        if key not in _LAYERS:
            used, rows = [], []
            for r, c in enumerate(self.checks):
                vs = set(c.tolist())
                for u, lst in zip(used, rows):
                    if not (u & vs):
                        u |= vs
                        lst.append(r)
                        break
                else:
                    used.append(vs)
                    rows.append([r])
            _LAYERS[key] = [np.array(l, dtype=np.int64) for l in rows]
        return _LAYERS[key]

    def min_sum(self, llr):
        """Layered normalized min-sum on (B, N) LLRs: (hard decisions (B, N), converged (B,)). Each codeword stops when every
        check holds. Layered (a layer of checks at a time, the variables updated after each) converges in about half the
        iterations of flooding: at Eb/N0 0.9 dB (DVB-S2 1/2 normal frame) 50 layered iterations decode what 100 flooding did.
        Keeps nothing on the code, so threads can share it"""
        layers = self.layers()
        iters = _effort.iters(self.iters)
        if _accel.native is not None:
            return _accel.native.ldpc_minsum(np.ascontiguousarray(llr, dtype=np.float64), self.var, self.cstart.astype(np.int64),
                                             self.deg, np.concatenate(layers), iters, float(self.alpha))
        total = np.array(llr, dtype=np.float64) # (B, N): the a-posteriori values
        r = np.zeros((total.shape[0], len(self.var))) # check to variable messages, per edge
        done = np.zeros(total.shape[0], dtype=bool)
        cols = np.arange(self.dmax)
        for _ in range(iters):
            act = np.nonzero(~done)[0] # (a codeword that has converged stops, as in the C++ decoder)
            if not len(act):
                break
            t, rr = total[act], r[act]
            for rows in layers:
                ok = self.valid[rows]
                e = np.where(ok, self.slot[rows], 0) # (rows, dmax) edges
                q = np.where(ok[None], t[:, self.var[e]] - rr[:, e], np.inf) # variable to check, without this check's own
                a = np.abs(q)
                neg = (q < 0) & ok[None]
                i1 = np.argmin(a, axis=2)
                m1 = np.take_along_axis(a, i1[..., None], axis=2)
                a2 = a.copy()
                np.put_along_axis(a2, i1[..., None], np.inf, axis=2)
                m2 = a2.min(axis=2, keepdims=True)
                sgn = np.where(neg.sum(axis=2, keepdims=True) % 2 == 1, -1.0, 1.0)*np.where(neg, -1.0, 1.0)
                new = self.alpha*sgn*np.where(cols[None, None, :] == i1[..., None], m2, m1)
                ev = e[ok]
                rr[:, ev] = new[:, ok]
                t[:, self.var[ev]] = q[:, ok] + new[:, ok]
            total[act], r[act] = t, rr
            done[act] = ~self.syndrome(t < 0).any(axis=1)
        return total < 0, done

    def decode(self, bits, soft=None):
        llr = as_llr(bits, soft, 2.0)
        llr = llr[:len(llr)//self.N*self.N].reshape(-1, self.N)
        cw, converged = self.min_sum(llr)
        self.converged, self.errors = converged, word_errors(llr < 0, cw, converged)
        return cw[:, self.info].reshape(-1)
