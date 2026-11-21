import numpy as np

from ._code import _code, register, word_errors, as_llr
from .trellis import Trellis, bcjr
from ... import _effort

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
    decode() takes LLRs (positive = 0) of the 3 (length + 4) bits and returns the length message bits; app() gives the
    final a-posteriori LLRs of the message bits instead (decode's bits are app() < 0). Status (status=True): systematic
    bits that the decoder changed (a CRC is the real check).
    """
    soft = True

    @classmethod
    def from_config(cls, log, obj):
        if "outputs" in obj:
            return turbo_mux.from_config(log, obj)
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

    def _constituent(self, Ls, Lp, La, tail_s, tail_p, maxlog):
        """A-posteriori LLRs of the K message bits from one constituent decoder (trellis from and to state 0)"""
        sys = np.concatenate((Ls + La, tail_s))
        par = np.concatenate((Lp, tail_p))
        app = bcjr(self.code.trellis, np.stack((sys, par), axis=-1)[None], start=0, end=0, maxlog=maxlog)[0]
        return app[:len(Ls)]

    def app(self, bits, soft=None):
        """The a-posteriori LLRs of the message bits (positive = 0), every word's K one after the other"""
        return self._iterate(bits, soft)[0].reshape(-1)

    def _decode(self, bits, soft=None):
        finals, hard = self._iterate(bits, soft)
        dec = finals < 0
        return dec.reshape(-1), word_errors(hard, dec)

    def _iterate(self, bits, soft):
        """(final LLRs (words, K), the received systematic bits hard decided (words, K))"""
        llr = as_llr(bits, soft, 2.0)
        K, n = self.Kb, self.den
        # The effort level (rpps._effort): its iterations and max-log or log-MAP. The extrinsic scaling corrects max-log
        # only; past the definition's iterations a word stops once two iterations agree
        iters, maxlog = _effort.iters(self.iters), _effort.maxlog(self.maxlog)
        scale = self.scale if maxlog or maxlog == self.maxlog else 1.0
        words = llr[:len(llr)//n*n].reshape(-1, n)
        finals = []
        for w in words:
            d = w.reshape(3, K + 4)
            Ls, L1, L2 = d[0, :K], d[1, :K], d[2, :K]
            t = d[:, K:]
            t1s, t1p = np.array([t[0, 0], t[2, 0], t[1, 1]]), np.array([t[1, 0], t[0, 1], t[2, 1]])
            t2s, t2p = np.array([t[0, 2], t[2, 2], t[1, 3]]), np.array([t[1, 2], t[0, 3], t[2, 3]])
            La1 = np.zeros(K)
            last, same = None, 0
            for it in range(iters):
                Le1 = scale*(self._constituent(Ls, L1, La1, t1s, t1p, maxlog) - Ls - La1)
                La2 = Le1[self.pi]
                Le2 = scale*(self._constituent(Ls[self.pi], L2, La2, t2s, t2p, maxlog) - Ls[self.pi] - La2)
                La1 = Le2[self.inv]
                if it >= self.iters: # (beyond the definition's own: only at a higher effort)
                    now = (Ls + La1 + Le1) < 0
                    same = same + 1 if last is not None and np.array_equal(now, last) else 0
                    last = now
                    if same >= 2:
                        break
            finals.append(Ls + La1 + Le1)
        return np.array(finals).reshape(len(words), K), words[:, :K] < 0


class turbo_mux(_code):
    """Parallel concatenated code with its outputs multiplexed bit time by bit time (CCSDS 131.0-B 6): the layout `turbo`
    takes when a definition gives "outputs"

        length: block size k
        g0: the feedback (backward connection) polynomial, D^0 first ("0b10011", or octal as LTE writes it)
        forward: {"1": g1, "2": g2, ...}: the forward polynomials the outputs name
        constraint: taps of each polynomial
        interleave: an interleave definition giving the second encoder's input order ("block/ccsds_turbo"), with `args`
        outputs: what is sent at each bit time, a list of periods: [["0a", "1a"], ["0a", "1b"]] sends the systematic bit
            and encoder a's parity 1 at even times, the systematic bit and encoder b's parity 1 at odd ones. "0a" is the
            systematic stream (with encoder a's tail), "<p>a" / "<p>b" encoder a's / b's output of forward polynomial p
    Both encoders start at zero and are terminated (constraint - 1 steps with the input from the feedback), so a word is
    (k + constraint - 1) bit times. Decoded by iterative BCJR (max-log by default, extrinsic scaling `scale`).
    """
    soft = True

    @classmethod
    def from_config(cls, log, obj):
        from ..serial import load
        K = int(obj["length"])
        il = obj["interleave"]
        name, args = (il, {}) if isinstance(il, str) else (il["name"], il.get("args", {}))
        perm = load(f"interleave/{name}", **args).encoder.src
        return cls(log, K, obj["g0"], obj["forward"], obj.get("constraint", 5), perm, obj["outputs"],
                   obj.get("iters", 10), obj.get("scale", 0.75), obj.get("maxlog", True))

    def __init__(self, log, length, g0, forward, constraint, perm, outputs, iters=10, scale=0.75, maxlog=True):
        self.Kb, self.K = int(length), int(constraint)
        self.perm = np.asarray(perm, dtype=np.int64)
        if len(self.perm) != self.Kb or not np.array_equal(np.sort(self.perm), np.arange(self.Kb)):
            raise ValueError(f"turbo: the interleaver is not a permutation of {self.Kb}")
        self.inv = np.argsort(self.perm)
        self.polys = {str(p): g for p, g in forward.items()}
        self.outputs = [list(o) for o in outputs]
        for o in (x for period in self.outputs for x in period):
            if o != "0a" and (o[-1] not in "ab" or o[:-1] not in self.polys):
                raise ValueError(f"turbo: output {o!r} names no forward polynomial")
        self.codes = {p: rsc(g0, g, self.K) for p, g in self.polys.items()}
        first = next(iter(self.codes.values()))
        self.m = first.m
        self.T = self.Kb + self.m
        # One trellis per encoder: label = (systematic, parity of each forward polynomial in `order`)
        self.order = sorted(self.polys)
        label = np.stack([np.broadcast_to(np.arange(2), (first.S, 2))] + [self.codes[p].par for p in self.order], axis=-1)
        self.trellis = Trellis(first.next[None], label[None])
        self.slots = [] # (time, stream) of each sent bit, in order
        for t in range(self.T):
            for o in self.outputs[t % len(self.outputs)]:
                self.slots.append((t, o))
        self.iters, self.scale, self.maxlog = iters, scale, maxlog
        super().__init__(log, None, self.Kb, len(self.slots))

    def _run(self, u):
        """Systematic stream (with the tail) and {poly: parity stream} of one encoder, terminated"""
        out = {}
        for p in self.order:
            z, tx, tz = self.codes[p].encode(u)
            out[p] = np.concatenate((z, tz))
        return np.concatenate((u, tx)).astype(np.uint8), out

    def encode(self, bits):
        c = np.asarray(bits).astype(np.uint8)
        words = []
        for blk in c[:len(c)//self.Kb*self.Kb].reshape(-1, self.Kb):
            sa, pa = self._run(blk)
            _, pb = self._run(blk[self.perm])
            streams = {"0a": sa, **{p + "a": v for p, v in pa.items()}, **{p + "b": v for p, v in pb.items()}}
            words.append(np.array([streams[o][t] for t, o in self.slots], dtype=np.uint8))
        return np.concatenate(words).astype(bool) if words else np.zeros(0, bool)

    def _streams(self, w):
        """Channel LLRs of each stream (T,), 0 where a stream was not sent at that time"""
        s = {o: np.zeros(self.T) for o in ["0a"] + [p + e for p in self.order for e in "ab"]}
        for (t, o), v in zip(self.slots, w):
            s[o][t] += v
        return s

    def app(self, bits, soft=None):
        """The a-posteriori LLRs of the message bits (positive = 0), every word's K one after the other"""
        return self._iterate(bits, soft)[0].reshape(-1)

    def _decode(self, bits, soft=None):
        finals, hard = self._iterate(bits, soft)
        dec = finals < 0
        return dec.reshape(-1), word_errors(hard, dec)

    def _iterate(self, bits, soft):
        """(final LLRs (words, K), the received systematic bits hard decided (words, K))"""
        llr = as_llr(bits, soft, 2.0)
        n, K, m = self.den, self.Kb, self.m
        iters, maxlog = _effort.iters(self.iters), _effort.maxlog(self.maxlog)
        scale = self.scale if maxlog or maxlog == self.maxlog else 1.0
        words = llr[:len(llr)//n*n].reshape(-1, n)
        finals, hard = [], []
        for w in words:
            s = self._streams(w)
            Ls = s["0a"][:K]
            pa = np.stack([s[p + "a"] for p in self.order], axis=-1)
            pb = np.stack([s[p + "b"] for p in self.order], axis=-1)
            La1 = np.zeros(K)
            for _ in range(iters):
                sys_a = np.concatenate((Ls + La1, s["0a"][K:]))
                app = bcjr(self.trellis, np.concatenate((sys_a[:, None], pa), axis=1)[None], start=0, end=0, maxlog=maxlog)[0][:K]
                Le1 = scale*(app - Ls - La1)
                La2 = Le1[self.perm]
                sys_b = np.concatenate((Ls[self.perm] + La2, np.zeros(m))) # (encoder b's tail systematic bits are not sent)
                app = bcjr(self.trellis, np.concatenate((sys_b[:, None], pb), axis=1)[None], start=0, end=0, maxlog=maxlog)[0][:K]
                Le2 = scale*(app - Ls[self.perm] - La2)
                La1 = Le2[self.inv]
            finals.append(Ls + La1 + Le1)
            hard.append(Ls < 0)
        return np.array(finals).reshape(len(words), K), np.array(hard).reshape(len(words), K)
