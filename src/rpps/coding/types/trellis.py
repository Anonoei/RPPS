"""Trellis (BCJR / MAP) decoding

A trellis lists, for every section of the code, which state each branch leaves and enters and which coded
bits it carries. `bcjr` runs the forward-backward algorithm over it and returns, for every section, the
a-posteriori log-likelihood ratio of the branch bit. The branch bit is whatever the trellis is built around:

 - convolutional codes (`bcjr` decoder): the information bit. States are the encoder register
 - linear block codes (`trellis` decoder): the coded bit. States are the syndrome so far (Wolf's trellis), so
   it has 2^(n-k) states: practical for short codes (Hamming, small BCH), not for LDPC

Unlike Viterbi it gives soft outputs (usable by an iterative decoder) and it minimises the bit error rate
rather than the sequence error rate. LLRs are log(P(0)/P(1)), positive means 0, as everywhere else. In the
log domain (`maxlog=False`) the inputs have to be true LLRs: scale them wrong and the output is over or
under confident. `maxlog=True` is the cheap approximation (Viterbi-like) and does not care about scale.
"""
import numpy as np

from ._code import _code, register
from .viterbi import viterbi
from ..blocker import block

class Trellis:
    """Sections of a trellis

    next: (P, S, 2) state reached from each state by each branch bit; section t uses next[t % P]
    label: (P, S, 2, L) the L coded bits each branch carries
    Every state needs exactly two branches in, which holds for convolutional and syndrome trellises.
    """
    def __init__(self, next_state, label):
        self.next = np.asarray(next_state, dtype=np.int64)
        self.label = np.asarray(label, dtype=np.uint8)
        self.P, self.S, _ = self.next.shape
        self.L = self.label.shape[-1]
        self.sign = 1.0 - 2.0*self.label.astype(np.float64) # +1 for a 0 bit, -1 for a 1 bit
        # Predecessors of each state: (P, S, 2) previous state and branch bit
        self.pred_state = np.zeros((self.P, self.S, 2), dtype=np.int64)
        self.pred_bit = np.zeros((self.P, self.S, 2), dtype=np.int64)
        for p in range(self.P):
            src = np.repeat(np.arange(self.S), 2)
            bit = np.tile(np.arange(2), self.S)
            order = np.argsort(self.next[p].reshape(-1), kind="stable")
            dst = self.next[p].reshape(-1)[order]
            if not np.array_equal(dst, np.repeat(np.arange(self.S), 2)):
                raise ValueError("Every trellis state needs exactly two incoming branches")
            self.pred_state[p] = src[order].reshape(self.S, 2)
            self.pred_bit[p] = bit[order].reshape(self.S, 2)

    @staticmethod
    def syndrome(H):
        """Syndrome trellis of a parity-check matrix H (m, n): sections are coded bits, states the partial syndrome"""
        H = np.asarray(H, dtype=np.uint8)
        m, n = H.shape
        if m > MAX_STATE_BITS:
            raise ValueError(f"Syndrome trellis needs 2^{m} states: too many (limit 2^{MAX_STATE_BITS})")
        cols = (H.astype(np.int64) * (1 << np.arange(m - 1, -1, -1))[:, None]).sum(axis=0) # Each column as an integer
        S = 1 << m
        states = np.arange(S)
        nxt = np.stack([np.stack((states, states ^ cols[t]), axis=1) for t in range(n)])
        label = np.zeros((n, S, 2, 1), dtype=np.uint8)
        label[:, :, 1, 0] = 1
        return Trellis(nxt, label)

MAX_STATE_BITS = 16

def bcjr(tr, llr, start=0, end=0, maxlog=False):
    """Forward-backward over a trellis

    llr: (W, T, L) channel LLRs of the L coded bits of each of T sections, for W independent blocks
    start/end: the state every block starts / ends in (None: free)
    Returns (W, T) a-posteriori LLRs of the branch bits. Needs memory for 2 floats per state per section.
    """
    llr = np.asarray(llr, dtype=np.float64)
    W, T, L = llr.shape
    S = tr.S
    comb = np.maximum if maxlog else np.logaddexp
    sgn = tr.sign.reshape(tr.P, S, 2, L)

    def gamma(t):
        return 0.5*np.einsum("sil,wl->wsi", sgn[t % tr.P], llr[:, t])  # (W, S, 2)

    alpha = np.full((T+1, W, S), -np.inf)
    if start is None:
        alpha[0] = 0.0
    else:
        alpha[0, :, start] = 0.0
    for t in range(T):
        p = t % tr.P
        g = gamma(t)
        ps, pb = tr.pred_state[p], tr.pred_bit[p]
        a = alpha[t]
        a_next = comb(a[:, ps[:, 0]] + g[:, ps[:, 0], pb[:, 0]], a[:, ps[:, 1]] + g[:, ps[:, 1], pb[:, 1]])
        alpha[t+1] = a_next - a_next.max(axis=1, keepdims=True)

    beta = np.full((W, S), -np.inf)
    if end is None:
        beta[:] = 0.0
    else:
        beta[:, end] = 0.0
    out = np.empty((W, T))
    for t in range(T-1, -1, -1):
        p = t % tr.P
        g = gamma(t)
        nxt = tr.next[p]
        m = alpha[t][:, :, None] + g + beta[:, nxt]  # (W, S, 2)
        reduce = (lambda x: x.max(axis=1)) if maxlog else (lambda x: np.logaddexp.reduce(x, axis=1))
        out[:, t] = reduce(m[:, :, 0]) - reduce(m[:, :, 1])
        b_new = comb(g[:, :, 0] + beta[:, nxt[:, 0]], g[:, :, 1] + beta[:, nxt[:, 1]])
        beta = b_new - b_new.max(axis=1, keepdims=True)
    return np.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

def _hard_to_llr(x, hard_llr):
    return hard_llr*(1.0 - 2.0*np.asarray(x, dtype=np.float64))

@register("bcjr")
class bcjr_conv(viterbi):
    """BCJR (MAP) decoder for feed-forward convolutional codes: same definition as `viterbi`

    maxlog: max-log-MAP (no log-sum-exp, insensitive to the LLR scale)
    hard_llr: the LLR magnitude given to hard input bits (log-MAP needs a guess at the channel; this is p = 5%)
    decode() takes hard bits or LLRs; `app()` returns the a-posteriori LLR of every information bit.
    """
    soft = True
    stream = False # Decodes a whole frame: in a stream the data is collected until it ends

    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["num"], obj["den"], obj["constraint"], np.array(obj["generator"], dtype=bool),
                   obj.get("puncture"), obj.get("terminated", False), obj.get("maxlog", False), obj.get("hard_llr", 2.94))

    def __init__(self, log, num, den, constraint, generator, puncture=None, terminated=False, maxlog=False, hard_llr=2.94):
        super().__init__(log, num, den, constraint, generator, puncture, terminated)
        self.maxlog, self.hard_llr = maxlog, hard_llr
        self.trellis = Trellis(self.next_state[None], self.out_bits[None])

    def app(self, bits, soft=None):
        """A-posteriori LLR of each information bit (the tail of a terminated code is left out)"""
        x = np.asarray(bits)
        if soft is None:
            soft = x.dtype.kind == "f"
        llr = x.astype(np.float64) if soft else _hard_to_llr(x, self.hard_llr)
        llr = self.depuncture(llr)                                   # (T, n)
        out = bcjr(self.trellis, llr[None], start=0, end=0 if self.terminated else None, maxlog=self.maxlog)[0]
        return out[:len(out)-(self.con-1)] if self.terminated else out

    def decode(self, bits, soft=None):
        return self.app(bits, soft) < 0

@register("trellis")
class trellis(_code):
    """Maximum a-posteriori decoder for a linear block code, on its syndrome trellis

    Corrects what the code can correct (a single error for Hamming) from hard bits, and uses LLRs when given.
    The message is the first k bits of each word (systematic). Built from the code's own parity-check matrix.
    """
    soft = True

    @classmethod
    def from_encoder(cls, log, enc, **kw):
        return cls(log, np.asarray(enc.check).T, **kw)

    def __init__(self, log, H, maxlog=False, hard_llr=2.94):
        H = np.asarray(H, dtype=np.uint8)
        m, n = H.shape
        super().__init__(log, None, n - m, n)
        self.maxlog, self.hard_llr = maxlog, hard_llr
        self.trellis = Trellis.syndrome(H)

    def app(self, bits, soft=None):
        """A-posteriori LLRs of the coded bits, (words, n)"""
        x = np.asarray(bits)
        if soft is None:
            soft = x.dtype.kind == "f"
        llr = x.astype(np.float64) if soft else _hard_to_llr(x, self.hard_llr)
        llr = block(llr[:len(llr)//self.den*self.den], self.den)
        return bcjr(self.trellis, llr[:, :, None], start=0, end=0, maxlog=self.maxlog)

    def decode(self, bits, soft=None):
        words = self.app(bits, soft) < 0
        return words[:, :self.num].reshape(-1)
