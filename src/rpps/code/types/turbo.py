import numpy as np

from ._code import _code, register, word_errors
from .trellis import Trellis, bcjr

def qpp(K, f1, f2):
    """Quadratic permutation polynomial interleaver: pi(i) = (f1 i + f2 i^2) mod K"""
    i = np.arange(K, dtype=np.int64)
    p = (f1*i + f2*i*i) % K
    if len(np.unique(p)) != K:
        raise ValueError(f"QPP ({f1}, {f2}) is not a permutation of {K}")
    return p

def _poly_bits(g, K):
    g = int(g, 0) if isinstance(g, str) else int(g)
    return [(g >> (K - 1 - i)) & 1 for i in range(K)] # Coefficient of D^0 first

class rsc:
    """Recursive systematic convolutional code: feedback g0 and feedforward g1 over K taps (D^0 first, octal as
    usual: LTE g0 = 0o13 = 1 + D^2 + D^3, g1 = 0o15 = 1 + D + D^3). State: the register, newest bit most significant"""
    def __init__(self, g0, g1, K):
        self.K, self.m = K, K - 1
        self.g0, self.g1 = _poly_bits(g0, K), _poly_bits(g1, K)
        S = 1 << self.m
        self.S = S
        self.next = np.zeros((S, 2), dtype=np.int64)
        self.par = np.zeros((S, 2), dtype=np.uint8)
        self.fb = np.zeros(S, dtype=np.uint8) # Feedback of each state (the input that drives it towards zero)
        for s in range(S):
            r = [(s >> (self.m - 1 - i)) & 1 for i in range(self.m)] # r[0] newest
            f = 0
            for i in range(1, K):
                f ^= self.g0[i] & r[i - 1]
            self.fb[s] = f
            for x in (0, 1):
                a = x ^ f
                p = self.g1[0] & a
                for i in range(1, K):
                    p ^= self.g1[i] & r[i - 1]
                self.par[s, x] = p
                self.next[s, x] = (a << (self.m - 1)) | (s >> 1)
        label = np.stack((np.broadcast_to(np.arange(2), (S, 2)), self.par), axis=-1) # (S, 2, 2): systematic, parity
        self.trellis = Trellis(self.next[None], label[None])

    def encode(self, x, terminate=True):
        """(parity, tail systematic bits, tail parity bits); the tail drives the register to zero (m steps)"""
        s, z = 0, np.zeros(len(x), dtype=np.uint8)
        for i, b in enumerate(x):
            z[i] = self.par[s, b]
            s = self.next[s, b]
        tx, tz = [], []
        if terminate:
            for _ in range(self.m):
                b = int(self.fb[s])
                tx.append(b)
                tz.append(int(self.par[s, b]))
                s = self.next[s, b]
        return z, np.array(tx, np.uint8), np.array(tz, np.uint8)

@register("turbo")
class turbo(_code):
    """Parallel concatenated (turbo) code, LTE layout (3GPP TS 36.212 5.1.3.2)

    Two identical RSC codes (feedback g0, feedforward g1, constraint K), the second fed through a QPP interleaver;
    each terminated by m tail steps. Output: three streams d0 (systematic), d1 (parity 1), d2 (parity 2) of length+4
    each, one after the other, with the 12 tail bits placed as 5.1.3.2.2 does.
        length: block size; qpp: [f1, f2], or qpp_table: {"<length>": [f1, f2], ...}
        iters: decoder iterations, scale: extrinsic scaling (max-log), maxlog: max-log-MAP (default) or log-MAP
    decode() takes LLRs (positive = 0) of the 3 (length + 4) bits and returns the length message bits; `llr` keeps
    the final a-posteriori LLRs. `errors`: systematic bits that the decoder changed (a CRC is the real check).
    """
    soft = True

    @classmethod
    def from_config(cls, log, obj):
        K = int(obj["length"])
        f = obj.get("qpp") or (obj.get("qpp_table") or {}).get(str(K))
        if f is None:
            raise ValueError(f"No QPP parameters for block size {K}")
        return cls(log, K, obj.get("g0", "0o13"), obj.get("g1", "0o15"), obj.get("constraint", 4), f,
                   obj.get("iters", 8), obj.get("scale", 0.75), obj.get("maxlog", True))

    def __init__(self, log, length, g0, g1, constraint, qpp_f, iters=8, scale=0.75, maxlog=True):
        self.code = rsc(g0, g1, constraint)
        if self.code.m != 3:
            raise ValueError("The LTE tail layout needs constraint length 4")
        super().__init__(log, None, length, 3*(length + 4))
        self.Kb = length
        self.pi = qpp(length, *qpp_f)
        self.inv = np.argsort(self.pi)
        self.iters, self.scale, self.maxlog = iters, scale, maxlog
        self.llr = None

    def encode(self, bits):
        c = np.asarray(bits).astype(np.uint8)
        K = self.Kb
        out = []
        for blk in c[:len(c)//K*K].reshape(-1, K):
            z, x, zt = self.code.encode(blk)
            z2, x2, zt2 = self.code.encode(blk[self.pi])
            d = np.zeros((3, K + 4), dtype=np.uint8)
            d[0, :K], d[1, :K], d[2, :K] = blk, z, z2
            d[:, K:] = [[x[0], zt[1], x2[0], zt2[1]],    # 5.1.3.2.2
                        [zt[0], x[2], zt2[0], x2[2]],
                        [x[1], zt[2], x2[1], zt2[2]]]
            out.append(d.reshape(-1))
        return np.concatenate(out).astype(bool) if out else np.zeros(0, bool)

    def _constituent(self, Ls, Lp, La, tail_s, tail_p):
        """A-posteriori LLRs of the K message bits from one constituent decoder (trellis from and to state 0)"""
        sys = np.concatenate((Ls + La, tail_s))
        par = np.concatenate((Lp, tail_p))
        app = bcjr(self.code.trellis, np.stack((sys, par), axis=-1)[None], start=0, end=0, maxlog=self.maxlog)[0]
        return app[:len(Ls)]

    def decode(self, bits, soft=None):
        x = np.asarray(bits)
        if soft is None:
            soft = x.dtype.kind == "f"
        llr = x.astype(np.float64) if soft else 2.0*(1.0 - 2.0*x.astype(np.float64))
        K, n = self.Kb, self.den
        words = llr[:len(llr)//n*n].reshape(-1, n)
        out, finals = [], []
        for w in words:
            d = w.reshape(3, K + 4)
            Ls, L1, L2 = d[0, :K], d[1, :K], d[2, :K]
            t = d[:, K:]
            t1s, t1p = np.array([t[0, 0], t[2, 0], t[1, 1]]), np.array([t[1, 0], t[0, 1], t[2, 1]])
            t2s, t2p = np.array([t[0, 2], t[2, 2], t[1, 3]]), np.array([t[1, 2], t[0, 3], t[2, 3]])
            La1 = np.zeros(K)
            for _ in range(self.iters):
                Le1 = self.scale*(self._constituent(Ls, L1, La1, t1s, t1p) - Ls - La1)
                La2 = Le1[self.pi]
                Le2 = self.scale*(self._constituent(Ls[self.pi], L2, La2, t2s, t2p) - Ls[self.pi] - La2)
                La1 = Le2[self.inv]
            final = Ls + La1 + Le1
            finals.append(final)
            out.append(final < 0)
        self.llr = np.concatenate(finals) if finals else np.zeros(0)
        dec = np.array(out).reshape(len(words), K)
        self.errors = word_errors(words[:, :K] < 0, dec)
        return dec.reshape(-1)
