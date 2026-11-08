import numpy as np

from ._code import _code, register
from .conv import termination_of, generators_of

@register("viterbi")
class viterbi(_code):
    """Viterbi decoder for feed-forward convolutional codes (rate 1/n, any constraint length)

    generator: (n, K) taps. generator[j][i] multiplies the input delayed by i, the same convention as `conv`
    puncture: optional (n, P) 0/1 pattern, 1 = bit was sent. Punctured positions are decoded as erasures
    terminated: the encoder was flushed with K-1 zeros, so the path ends in state 0 (and the tail is dropped).
        Or a termination name: "zero" (the same), "truncated" (no tail: the best end state), "tailbiting" (the
        path ends where it started: decoded on three laps round the frame, keeping the middle one; frames only)

    decode() takes hard bits (0/1) or soft values. Soft values are LLRs, log(P(bit=0)/P(bit=1)): positive means 0.
    Pass `soft=True` to force soft, otherwise integer/bool input is treated as hard bits.

    Without `state` decode() takes a whole frame. With `state` (a dict) it is a stream: the path metrics and the
    survivor decisions carry over, and a bit is released once it is `depth` trellis steps old (it is then settled for
    practical purposes). `final` ends the stream and releases the rest, ending in state 0 if terminated.
    """
    soft = True
    stream = True # decode() can carry on from a previous call

    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["num"], obj["den"], obj["constraint"], generators_of(obj),
                   obj.get("puncture"), termination_of(obj))

    def __init__(self, log, num, den, constraint, generator, puncture=None, terminated=False):
        super().__init__(log, None, num, den)
        self.con = constraint
        self.gen = np.asarray(generator, dtype=np.uint8).reshape(den, constraint)
        self.puncture = None if puncture is None else np.asarray(puncture, dtype=bool)
        self.termination = termination_of({"termination": terminated})
        if self.termination == "tailbiting":
            self.stream = False # Whole frames: a stream is collected until it ends
        self.depth = 10*constraint # Traceback depth of a stream

        K, n = self.con, self.den
        S = 2**(K-1)
        self.n_states = S

        # Register for (state, input): [input, newest past bit, ..., oldest]. State bit K-2 is the newest past bit
        state = np.arange(S)
        reg = np.zeros((S, 2, K), dtype=np.uint8)
        reg[:, :, 0] = np.arange(2)[None, :]
        for i in range(1, K):
            reg[:, :, i] = ((state >> (K-1-i)) & 1)[:, None]
        self.out_bits = (np.einsum("sik,nk->sin", reg, self.gen) & 1).astype(np.uint8) # (S, 2, n)
        pow2 = 2**np.arange(n-1, -1, -1)
        self.out_pat = (self.out_bits.astype(int) * pow2).sum(axis=2) # Code word as an integer (S, 2)
        self.patterns = ((np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(np.float64) # (2^n, n)

        self.next_state = (np.arange(2)[None, :] << (K-2)) | (state[:, None] >> 1) # (S, 2)
        ns = np.arange(S)
        self.prev = np.stack([((ns << 1) & (S-1)) | b for b in (0, 1)], axis=1) # (S, 2) predecessors
        self.prev_in = ns >> (K-2) # Input bit that leads into each state

    def depuncture(self, values):
        """Insert zeros (erasures) for the punctured positions. Returns (T, n) array"""
        n = self.den
        if self.puncture is None or self.puncture.all():
            return values[:len(values)//n*n].reshape(-1, n)
        P = self.puncture.shape[1]
        per_period = int(self.puncture.sum())
        periods = len(values) // per_period
        mask = np.tile(self.puncture, (1, periods)) # (n, T)
        out = np.zeros((n, periods*P))
        out.T[mask.T] = values[:periods*per_period]
        return out.T

    @property
    def terminated(self):
        return self.termination == "zero"

    def decode(self, bits: np.ndarray, soft=None, state=None, final=True):
        x = np.asarray(bits)
        if soft is None:
            soft = x.dtype.kind == "f"
        llr = x.astype(np.float64) if soft else 1.0 - 2.0*x.astype(np.float64) # 0 -> +1, 1 -> -1
        llr = self.depuncture(llr)                                              # (T, n)
        if self.termination == "tailbiting":
            return self._tailbiting(llr)
        T = len(llr)
        S = self.n_states

        # Cost of sending a bit 1 where llr>0 (it says 0), 0 otherwise: sum of llr over the code word's 1 bits
        cost = llr @ self.patterns.T                                            # (T, 2^n)

        if state and "pm" in state:
            pm, past = state["pm"], state["hist"]
        else:
            pm, past = np.full(S, np.inf), np.zeros((0, S), dtype=np.uint8)
            pm[0] = 0.0
        decision = np.zeros((T, S), dtype=np.uint8)
        pat0, pat1 = self.out_pat[self.prev[:, 0], self.prev_in], self.out_pat[self.prev[:, 1], self.prev_in]
        p0, p1 = self.prev[:, 0], self.prev[:, 1]
        for t in range(T):
            c = cost[t]
            m0 = pm[p0] + c[pat0]
            m1 = pm[p1] + c[pat1]
            take1 = m1 < m0
            decision[t] = take1
            pm = np.where(take1, m1, m0)
            pm -= pm.min() # Keep the metrics bounded
        decision = np.concatenate((past, decision))
        T = len(decision)

        end = 0 if (self.terminated and final) else int(np.argmin(pm))
        out = np.zeros(T, dtype=np.uint8)
        s_ = end
        for t in range(T-1, -1, -1):
            out[t] = self.prev_in[s_]
            s_ = self.prev[s_, decision[t, s_]]
        release = T if final else max(0, T - self.depth)
        out = out[:release]
        if state is not None:
            if final:
                state.clear()
            else:
                state["pm"], state["hist"] = pm, decision[release:]
        if self.terminated and final:
            out = out[:max(0, len(out)-(self.con-1))]
        return out.astype(bool)

    def _tailbiting(self, llr):
        """Viterbi over three laps of the circular trellis from no particular state, the middle lap kept: by then the
        path has settled, and it starts and ends in the state the frame's last bits leave"""
        T, S = len(llr), self.n_states
        cost = np.tile(llr, (3, 1)) @ self.patterns.T
        pm = np.zeros(S)
        decision = np.zeros((3*T, S), dtype=np.uint8)
        pat0, pat1 = self.out_pat[self.prev[:, 0], self.prev_in], self.out_pat[self.prev[:, 1], self.prev_in]
        p0, p1 = self.prev[:, 0], self.prev[:, 1]
        for t in range(3*T):
            c = cost[t]
            m0, m1 = pm[p0] + c[pat0], pm[p1] + c[pat1]
            take1 = m1 < m0
            decision[t] = take1
            pm = np.where(take1, m1, m0)
            pm -= pm.min()
        out = np.zeros(3*T, dtype=np.uint8)
        s_ = int(np.argmin(pm))
        for t in range(3*T - 1, -1, -1):
            out[t] = self.prev_in[s_]
            s_ = self.prev[s_, decision[t, s_]]
        return out[T:2*T].astype(bool)
