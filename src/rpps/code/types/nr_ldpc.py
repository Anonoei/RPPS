"""5G NR LDPC code with its rate matching (3GPP TS 38.212 5.2.2, 5.3.2, 5.4.2): one code block, K' bits to E bits

    {"base": "block", "type": "nr_ldpc", "bg": 1,
     "graph": [[[column, V(set 0), ..., V(set 7)], ...], ...],   the base graph's rows (Tables 5.3.2-2, 5.3.2-3)
     "kp": 8448,               K': the code block's bits (CRC included); the rest of its K = 22 Z (BG2: 10 Z) are filler
     "z": 0,                   the lifting size (0: the smallest Z of any set with Kb Z >= K', 5.2.2)
     "kb": null,               BG2's Kb (default from K' as one block's B: 10 above 640, 9 above 560, 8 above 192, else 6)
     "e": 25344, "rv": 0,      bits sent and the redundancy version (where the circular buffer is read from)
     "qm": 1,                  the modulation order the bit interleaver groups by (5.4.2.2; 1: none)
     "ncb": null,              the circular buffer (limited buffer rate matching; default N)
     "iters": 25, "alpha": 0.8}

The parity-check matrix lifts the base graph by Z: an entry V of row i, column j is the Z x Z identity shifted right by
V mod Z (the set of Table 5.3.2-1 that holds Z picks V's column). The codeword is the K bits and the parity, 68 Z bits
(BG2 52 Z), of which the first 2 Z are never sent: N = 66 Z (50 Z) go into the circular buffer. Encoding solves the 4 Z
core parity bits through the core's inverse (once per graph and Z), then every extension row in turn (each ends on an
identity). Decoding: the punctured bits erased, the filler known, layered min-sum (rpps.code.types.ldpc).

encode(): K' bits a block to E. decode(): E soft bits (positive = 0) a block to K' bits; errors -1 where the checks did
not all hold (the transport block's CRC is its caller's). `segment(B, bg)` and `base_graph(A, R)` give what a transport
block's size picks (5.2.2, 7.2.2).
"""
import functools

import numpy as np

from ._code import register, as_llr, word_errors
from .ldpc import ldpc

SETS = [[2, 4, 8, 16, 32, 64, 128, 256], [3, 6, 12, 24, 48, 96, 192, 384], [5, 10, 20, 40, 80, 160, 320],
        [7, 14, 28, 56, 112, 224], [9, 18, 36, 72, 144, 288], [11, 22, 44, 88, 176, 352], [13, 26, 52, 104, 208],
        [15, 30, 60, 120, 240]] # Table 5.3.2-1
SHAPE = {1: (46, 68, 22), 2: (42, 52, 10)} # Rows, columns, systematic columns of each base graph
K0 = {1: (0, 17, 33, 56), 2: (0, 13, 25, 43)} # Table 5.4.2.1-2: k0 = floor(x Ncb / (cols - 2) Z) Z
KNOWN = 1e3

def set_of(Z):
    return next(i for i, s in enumerate(SETS) if Z in s)

def lifting(kp, kb):
    """The smallest Z of any set with kb Z >= K'"""
    return min(z for s in SETS for z in s if kb*z >= kp)

def kb_of(bg, B):
    if bg == 1:
        return 22
    return 10 if B > 640 else 9 if B > 560 else 8 if B > 192 else 6

def base_graph(A, R):
    """The base graph of a transport block of A bits at code rate R (7.2.2)"""
    return 2 if (A <= 292 or (A <= 3824 and R <= 0.67) or R <= 0.25) else 1

def segment(B, bg):
    """Code block segmentation (5.2.2) of B bits (the transport block and its CRC): (C blocks, K' bits each with their
    CRC-24B when C > 1, K bits after the filler, Z)"""
    kcb = 8448 if bg == 1 else 3840
    if B <= kcb:
        C, Bp = 1, B
    else:
        C = -(-B//(kcb - 24))
        Bp = B + 24*C
    kp = -(-Bp//C)
    Z = lifting(kp, kb_of(bg, B))
    return C, kp, SHAPE[bg][2]*Z, Z

def _rot(x, s):
    """I(s) x for blocks x (..., Z): the identity shifted right by s, times x: x[(r + s) mod Z]"""
    return np.roll(x, -s, axis=-1)

@functools.lru_cache(maxsize=32)
def _core_inverse(graph_key, bg, Z):
    """The inverse over GF(2) of the 4Z x 4Z core (rows 0-3, the four columns after the systematic ones)"""
    graph = _GRAPHS[graph_key]
    ks = SHAPE[bg][2]
    ils = set_of(Z)
    B = np.zeros((4*Z, 4*Z), np.uint8)
    for i in range(4):
        for e in graph[i]:
            j = e[0] - ks
            if 0 <= j < 4:
                s = e[1 + ils] % Z
                r = np.arange(Z)
                B[i*Z + r, j*Z + (r + s) % Z] ^= 1
    n = 4*Z # Gauss-Jordan on [B | I]
    A = np.concatenate((B, np.eye(n, dtype=np.uint8)), axis=1)
    for c in range(n):
        p = c + int(np.argmax(A[c:, c]))
        if not A[p, c]:
            raise ValueError(f"nr_ldpc: BG{bg} core at Z {Z} is singular")
        if p != c:
            A[[c, p]] = A[[p, c]]
        rows = np.nonzero(A[:, c])[0]
        rows = rows[rows != c]
        A[rows] ^= A[c]
    return A[:, n:]

_GRAPHS = {}


@register("nr_ldpc")
class nr_ldpc(ldpc):
    """One 5G NR LDPC code block (see the module)"""

    def __init__(self, log, bg, graph, kp, e, z=None, kb=None, rv=0, qm=1, ncb=None, iters=25, alpha=0.8):
        self.bg, self.kp, self.E, self.rv, self.qm = int(bg), int(kp), int(e), int(rv), int(qm)
        rows, cols, ks = SHAPE[self.bg]
        if len(graph) != rows:
            raise ValueError(f"nr_ldpc: base graph {bg} has {rows} rows, the definition {len(graph)}")
        self.Z = int(z) if z else lifting(self.kp, int(kb) if kb is not None else kb_of(self.bg, self.kp))
        Z, ils = self.Z, set_of(self.Z)
        self.K = ks*Z
        if self.kp > self.K:
            raise ValueError(f"nr_ldpc: K' {self.kp} is more than K = {self.K} (Z {Z})")
        self.graph = [[(int(r[0]), int(r[1 + ils]) % Z) for r in row] for row in graph]
        key = (self.bg, hash(str(graph)))
        _GRAPHS.setdefault(key, graph)
        self._key = key
        base = [[-1]*cols for _ in range(rows)]
        for i, row in enumerate(self.graph):
            for j, s in row:
                base[i][j] = s
        from .ldpc import qc_checks
        n, checks = qc_checks(base, Z)
        super().__init__(log, n, checks, self.K, "head", iters, alpha)
        self.num, self.den = self.kp, self.E # (K' bits in, E out: what the operators count)
        self.Nd = (cols - 2)*Z # N: what goes into the buffer, from the codeword's bit 2Z on
        self.ncb = self.Nd if ncb is None else min(int(ncb), self.Nd)
        filler = np.zeros(self.Nd, bool)
        filler[self.kp - 2*Z:self.K - 2*Z] = True # (the filler is in the systematic part, past the punctured 2Z)
        k0 = K0[self.bg][self.rv]*self.ncb//((cols - 2)*Z)*Z
        sel, j = [], 0
        while len(sel) < self.E:
            p = (k0 + j) % self.ncb
            if not filler[p]:
                sel.append(p)
            j += 1
            if j > self.E + 4*self.ncb:
                raise ValueError("nr_ldpc: nothing to send in the circular buffer")
        sel = np.array(sel, np.int64)
        if self.qm > 1: # 5.4.2.2: f[i + j Qm] = e[i E/Qm + j]
            if self.E % self.qm:
                raise ValueError(f"nr_ldpc: E {self.E} is not a multiple of Qm {self.qm}")
            sel = sel.reshape(self.qm, self.E//self.qm).T.reshape(-1)
        self.index = sel + 2*Z # Codeword bit of each bit sent

    @classmethod
    def from_config(cls, log, obj):
        code = cls(log, obj["bg"], obj["graph"], obj["kp"], obj["e"], obj.get("z"), obj.get("kb"), obj.get("rv", 0),
                   obj.get("qm", 1), obj.get("ncb"), obj.get("iters", 25), obj.get("alpha", 0.8))
        code.schedule = obj.get("schedule", "layered")
        code.check = cls.check_of(obj)
        return code

    def codeword(self, c):
        """The 68 Z (52 Z) bits of each row of K bits (filler as 0)"""
        c = np.atleast_2d(np.asarray(c, np.uint8))
        W, Z = len(c), self.Z
        rows, cols, ks = SHAPE[self.bg]
        x = np.zeros((W, cols, Z), np.uint8)
        x[:, :ks] = c.reshape(W, ks, Z)
        lam = np.zeros((W, 4, Z), np.uint8)
        for i in range(4):
            for j, s in self.graph[i]:
                if j < ks:
                    lam[:, i] ^= _rot(x[:, j], s)
        p = (lam.reshape(W, 4*Z).astype(np.int64) @ _core_inverse(self._key, self.bg, Z).T.astype(np.int64)) % 2
        x[:, ks:ks + 4] = p.reshape(W, 4, Z)
        for i in range(4, rows):
            acc = np.zeros((W, Z), np.uint8)
            for j, s in self.graph[i]:
                if j < ks + 4:
                    acc ^= _rot(x[:, j], s)
            x[:, ks + i] = acc # (row i ends on the identity at column ks + i)
        return x.reshape(W, cols*Z)

    def encode(self, bits):
        b = np.asarray(bits).astype(np.uint8)
        m = b[:len(b)//self.kp*self.kp].reshape(-1, self.kp)
        if not len(m):
            return np.zeros(0, bool)
        c = np.zeros((len(m), self.K), np.uint8)
        c[:, :self.kp] = m
        return self.codeword(c)[:, self.index].reshape(-1).astype(bool)

    def codeword_llr(self, llr):
        """(W, E) soft bits to (W, 68 Z) of the codeword: the punctured erased, the filler known, repetitions added"""
        llr = np.atleast_2d(np.asarray(llr, float))
        full = np.zeros((len(llr), self.n))
        np.add.at(full, (slice(None), self.index), llr)
        full[:, self.kp:self.K] = KNOWN
        return full

    def _decode(self, bits, soft=None):
        llr = as_llr(np.asarray(bits), soft, 2.0)
        llr = llr[:len(llr)//self.E*self.E].reshape(-1, self.E)
        if not len(llr):
            return np.zeros(0, bool), np.zeros(0, int)
        cw, converged = self.propagate(self.codeword_llr(llr))
        sent = cw[:, self.index]
        return cw[:, :self.kp].reshape(-1), word_errors(llr < 0, sent, converged)
