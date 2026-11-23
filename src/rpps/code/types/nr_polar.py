"""5G NR polar code with its rate matching (3GPP TS 38.212 5.3.1, 5.4.1): K bits, CRC included, to E bits

    {"base": "block", "type": "nr_polar", "K": 56, "E": 864,
     "nmax": 9,                    the largest code, 2^nmax bits (9: downlink, 10: uplink)
     "interleave": true,           the input interleaver of 5.3.1.1 (I_IL: downlink)
     "bil": false,                 the coded-bit interleaver of 5.4.1.3 (I_BIL: uplink)
     "crc": "crc/crc24_nr_c",      the CRC the last bits of the K carry (the list decoder picks the path it checks)
     "ones": 24,                   the CRC computed over that many ones before the message (a DCI's)
     "list": 8}

N = 2^n from E and K (5.3.1); the K information positions are the most reliable of the polar sequence (config table
nr/polar_sequence) below N, past those the rate matching punctures or shortens (5.4.1.1); the N coded bits go through the
32-part sub-block interleaver into a circular buffer, read by repetition (E >= N), puncturing (K/E <= 7/16: the first
N - E not sent) or shortening (the last N - E not sent, known zeros). No parity-check bits (n_PC = 0: an uplink control
message of 12 to 19 bits is not covered).

encode(): messages of K - crc bits, the CRC attached (`mask`: XORed onto the last 16 CRC bits, an RNTI), to E bits each.
decode(): E soft bits (positive = 0) a codeword; the punctured positions erased, the shortened known, repetitions added;
then a list decoder (rpps.code.types.polar.scl) and the first path whose CRC checks; errors -1 when none does.
`paths(x)`: every path's K bits (deinterleaved, CRC included) and metric, for a blind search over masks (a PDCCH's RNTIs).
"""
import functools

import numpy as np

from ... import _effort
from ._code import _code, register, as_llr, word_errors
from .polar import encode_u, scl

SUBBLOCK = [0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19, 12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31] # Table 5.4.1.1-1
KNOWN = 1e3 # The LLR of a shortened bit: known to be 0

@functools.lru_cache(maxsize=None)
def _table(name):
    from ... import _config
    return [r[1] for r in _config.read(_config.find("table", name))["rows"]]

def sequence(N):
    """Bit indices below N from least to most reliable (5.3.1.2)"""
    return np.array([q for q in _table("nr/polar_sequence") if q < N], np.int64)

def input_pattern(K):
    """Pi(k) of the input interleaver for K bits (5.3.1.1)"""
    p = _table("nr/polar_interleave")
    if K > len(p):
        raise ValueError(f"nr_polar: the input interleaver takes up to {len(p)} bits, not {K}")
    return np.array([v - (len(p) - K) for v in p if v >= len(p) - K], np.int64)

def code_size(K, E, nmax):
    """n of N = 2^n (5.3.1)"""
    le = int(np.ceil(np.log2(E)))
    n1 = le - 1 if (E <= 9/8*2**(le - 1) and K/E < 9/16) else le
    n2 = int(np.ceil(np.log2(8*K)))
    return max(min(n1, n2, nmax), 5)

def subblock(N):
    """J(n): the coded bit the sub-block interleaver puts at n (5.4.1.1)"""
    n = np.arange(N)
    return np.array(SUBBLOCK)[32*n//N]*(N//32) + n % (N//32)

def triangle(E):
    """The coded-bit interleaver (5.4.1.3): f_k = e_order[k]"""
    T = int(np.ceil((np.sqrt(8*E + 1) - 1)/2))
    v = -np.ones((T, T), np.int64)
    k = 0
    for i in range(T):
        for j in range(T - i):
            if k < E:
                v[i, j] = k
            k += 1
    order = [v[i, j] for j in range(T) for i in range(T - j) if v[i, j] >= 0]
    return np.array(order, np.int64)


@register("nr_polar")
class nr_polar(_code):
    """The 5G NR polar code of K bits sent as E (see the module)"""
    soft = True
    CONTROLS = {"mask": "live"}

    def __init__(self, log, K, E, nmax=9, interleave=True, bil=False, crc=None, ones=0, size=8, mask=0):
        self.K, self.E, self.nmax = int(K), int(E), int(nmax)
        self.crc, self.ones, self.size, self.mask = crc, int(ones), int(size), int(mask)
        self.width = 0 if crc is None else crc.width
        if self.K > self.E or self.K <= self.width:
            raise ValueError(f"nr_polar: K {self.K} needs E >= K and more than the {self.width} CRC bits")
        n = code_size(self.K, self.E, self.nmax)
        N = self.N = 1 << n
        self.J = subblock(N)
        Q = sequence(N)
        frozen_tmp = set()
        if self.E < N:
            if self.K/self.E <= 7/16: # Puncturing
                frozen_tmp |= set(self.J[:N - self.E].tolist())
                top = int(np.ceil(3*N/4 - self.E/2)) if self.E >= 3*N/4 else int(np.ceil(9*N/16 - self.E/4))
                frozen_tmp |= set(range(top))
            else: # Shortening
                frozen_tmp |= set(self.J[self.E:].tolist())
        info = [q for q in Q if q not in frozen_tmp][-self.K:]
        self.info = np.sort(np.array(info, np.int64))
        self.frozen = np.ones(N, bool)
        self.frozen[self.info] = False
        self.pi = input_pattern(self.K) if interleave else np.arange(self.K)
        self.bil = triangle(self.E) if bil else None
        self.punctured = self.E < N and self.K/self.E <= 7/16
        super().__init__(log, f"nr_polar_{self.K}_{self.E}", self.K - self.width, self.E)

    @classmethod
    def from_config(cls, log, obj):
        crc = None
        if obj.get("crc"):
            from ... import _config
            crc = _config.load_ref(obj["crc"]).encoder
        return cls(log, obj["K"], obj["E"], obj.get("nmax", 9), obj.get("interleave", True), obj.get("bil", False), crc,
                   obj.get("ones", 0), obj.get("list", 8), obj.get("mask", 0))

    def control(self, mask=None):
        if mask is not None:
            self.mask = int(mask)

    def _mask_bits(self, mask):
        m = np.zeros(self.width, np.uint8)
        b = min(16, self.width)
        m[self.width - b:] = [(int(mask) >> (b - 1 - i)) & 1 for i in range(b)]
        return m

    def attach(self, msg, mask=None):
        """The K bits: the message and its CRC (over `ones` ones and the message), the mask on the CRC's last 16 bits"""
        msg = np.asarray(msg, np.uint8)
        if self.crc is None:
            return msg
        p = np.asarray(self.crc.field(np.r_[np.ones(self.ones, np.uint8), msg])).astype(np.uint8)
        return np.r_[msg, p ^ self._mask_bits(self.mask if mask is None else mask)]

    def residue(self, c):
        """CRC field XOR what the message's CRC is, per row of K bits (..., K): the mask a codeword carries (0: none)"""
        c = np.asarray(c, np.uint8)
        flat = c.reshape(-1, self.K)
        out = []
        for w in flat:
            p = np.asarray(self.crc.field(np.r_[np.ones(self.ones, np.uint8), w[:self.K - self.width]])).astype(np.uint8)
            r = 0
            for b in (w[self.K - self.width:] ^ p):
                r = (r << 1) | int(b)
            out.append(r)
        return np.array(out).reshape(c.shape[:-1])

    def encode(self, bits, mask=None):
        msg = np.asarray(bits).astype(np.uint8)
        out = []
        for m in msg[:len(msg)//self.num*self.num].reshape(-1, self.num):
            c = self.attach(m, mask)
            u = np.zeros(self.N, np.uint8)
            u[self.info] = c[self.pi]
            y = encode_u(u)[self.J]
            if self.E >= self.N:
                e = y[np.arange(self.E) % self.N]
            elif self.punctured:
                e = y[self.N - self.E:]
            else:
                e = y[:self.E]
            out.append(e if self.bil is None else e[self.bil])
        return (np.concatenate(out) if out else np.zeros(0, np.uint8)).astype(bool)

    def coded_llr(self, llr):
        """(W, E) soft bits to (W, N) soft bits of d, the code's output"""
        llr = np.atleast_2d(np.asarray(llr, float))
        W = len(llr)
        if self.bil is not None:
            e = np.zeros_like(llr)
            e[:, self.bil] = llr
            llr = e
        y = np.zeros((W, self.N))
        if self.E >= self.N:
            np.add.at(y, (slice(None), np.arange(self.E) % self.N), llr)
        elif self.punctured:
            y[:, self.N - self.E:] = llr
        else:
            y[:, :self.E] = llr
            y[:, self.E:] = KNOWN
        d = np.zeros_like(y)
        d[:, self.J] = y
        return d

    def paths(self, x, soft=None):
        """(K bits of every path (W, P, K), deinterleaved, CRC included; metrics (W, P), best first)"""
        llr = as_llr(np.asarray(x), soft)
        llr = llr[:len(llr)//self.E*self.E].reshape(-1, self.E)
        u, pm = scl(self.coded_llr(llr), self.frozen, _effort.scale("list", self.size))
        cp = u[:, :, self.info]
        c = np.zeros_like(cp)
        c[:, :, self.pi] = cp
        return c, pm

    def _decode(self, bits, soft=None):
        x = np.asarray(bits)
        if len(x) < self.E:
            return np.zeros(0, bool), np.zeros(0, int)
        c, _ = self.paths(x, soft)
        W = len(c)
        if self.crc is None:
            pick, ok = np.zeros(W, np.int64), np.ones(W, bool)
        else:
            good = self.residue(c) == self.mask
            ok = good.any(axis=1)
            pick = np.where(ok, np.argmax(good, axis=1), 0)
        best = c[np.arange(W), pick]
        llr = as_llr(x[:W*self.E], soft).reshape(W, self.E)
        sent = np.array([self.encode(b[:self.num], self.mask) for b in best]).reshape(W, self.E) if W else np.zeros((0, self.E), bool)
        return best[:, :self.num].reshape(-1).astype(bool), word_errors(llr < 0, sent, ok)
