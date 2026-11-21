import numpy as np

from ... import _accel
from ._code import _code, register, as_llr
from .conv import termination_of, generators_of

@register("viterbi")
class viterbi(_code):
    """Viterbi decoder for feed-forward convolutional codes (rate 1/n, any constraint length)

    generator: (n, K) taps. generator[j][i] multiplies the input delayed by i, the same convention as `conv`
    puncture: optional (n, P) 0/1 pattern, 1 = bit was sent. Punctured positions are decoded as erasures
    invert: optional n 0/1 flags: outputs the encoder sent inverted (their values are turned back before decoding)
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
                   obj.get("puncture"), termination_of(obj), obj.get("invert"))

    def __init__(self, log, num, den, constraint, generator, puncture=None, termination="truncated", invert=None):
        super().__init__(log, None, num, den)
        self.con = constraint
        self.gen = np.asarray(generator, dtype=np.uint8).reshape(den, constraint)
        self.puncture = None if puncture is None else np.asarray(puncture, dtype=bool)
        self.invert = None if invert is None or not any(invert) else np.where(np.asarray(invert, dtype=bool), -1.0, 1.0)
        self.termination = termination_of({"termination": termination})
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
        # The same trellis for rpps._native: branch j into state s comes from prev[s, j] and sends pattern _pat[s, j]
        self._pred = np.ascontiguousarray(self.prev, dtype=np.int32)
        self._pat = np.ascontiguousarray(np.stack([self.out_pat[self.prev[:, j], self.prev_in] for j in (0, 1)], axis=1), dtype=np.int32)

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

    def _decode(self, bits, soft=None, state=None, final=True):
        return self._path(bits, soft, state, final), None # No codewords: no status

    def _path(self, bits: np.ndarray, soft=None, state=None, final=True):
        llr = self.depuncture(as_llr(bits, soft))                              # (T, n)
        if self.invert is not None:
            llr = llr*self.invert[None, :] # An inverted output's 0 means 1 (erasures stay 0)
        S = self.n_states
        if self.termination == "tailbiting":
            # Three laps of the circular trellis from no particular state, the middle one kept: by then the path has
            # settled, and it starts and ends in the state the frame's last bits leave
            T = len(llr)
            decision, pm = self._forward(np.tile(llr, (3, 1)), np.zeros(S))
            return self.prev_in[self._states(decision, int(np.argmin(pm)))][T:2*T].astype(bool)
        if state and "pm" in state:
            pm, past = state["pm"], state["hist"]
        else:
            pm, past = np.full(S, np.inf), np.zeros((0, S), dtype=np.uint8)
            pm[0] = 0.0
        decision, pm = self._forward(llr, pm)
        decision = np.concatenate((past, decision)) if len(past) else decision
        T = len(decision)
        end = 0 if (self.terminated and final) else int(np.argmin(pm))
        out = self.prev_in[self._states(decision, end)]
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

    def decode_batch(self, words, soft=None):
        """Words (W, L), each decoded as a frame of its own: decisions (W, T), the same as decode() on each row. In C++
        (rpps._native.viterbi_batch) on several threads; a loop over decode() without the kernels or for a tail-biting code"""
        words = np.asarray(words)
        if _accel.native is None or self.termination == "tailbiting" or len(words) == 0:
            return np.array([self.decode(w, soft) for w in words], dtype=bool).reshape(len(words), -1)
        llr = self._depuncture_batch(as_llr(words, soft)) # (W, T, n)
        if self.invert is not None:
            llr = llr*self.invert[None, None, :]
        S = self.n_states
        pm0 = np.full(S, np.inf)
        pm0[0] = 0.0
        end = np.full(len(llr), 0 if self.terminated else -1, dtype=np.int64) # -1: the best end state, as decode()'s argmin
        states, _, _ = _accel.native.viterbi_batch(np.ascontiguousarray(llr, dtype=np.float64), self.patterns, 1, self._pred,
                                                   self._pat, pm0, end)
        out = self.prev_in[states]
        if self.terminated:
            out = out[:, :max(0, out.shape[1]-(self.con-1))]
        return out.astype(bool)

    def _depuncture_batch(self, values):
        """depuncture() of each row of values (W, L): (W, T, n)"""
        n = self.den
        if self.puncture is None or self.puncture.all():
            T = values.shape[1]//n
            return values[:, :T*n].reshape(len(values), T, n)
        P = self.puncture.shape[1]
        per_period = int(self.puncture.sum())
        periods = values.shape[1]//per_period
        mask = np.tile(self.puncture, (1, periods)).T # (T, n): the same order depuncture() fills
        out = np.zeros((len(values), periods*P, n))
        out[:, mask] = values[:, :periods*per_period]
        return out

    def _forward(self, llr, pm):
        """Add-compare-select over (T, n) LLRs from path metrics pm: (decisions (T, S), final metrics)"""
        if _accel.native is not None:
            return _accel.native.viterbi_forward(np.ascontiguousarray(llr, dtype=np.float64), self.patterns, 1,
                                                 self._pred, self._pat, pm)
        # Cost of sending a bit 1 where llr>0 (it says 0), 0 otherwise: sum of llr over the code word's 1 bits
        cost = llr @ self.patterns.T                                            # (T, 2^n)
        decision = np.zeros((len(llr), self.n_states), dtype=np.uint8)
        pat0, pat1 = self._pat[:, 0], self._pat[:, 1]
        p0, p1 = self.prev[:, 0], self.prev[:, 1]
        for t in range(len(llr)):
            c = cost[t]
            m0 = pm[p0] + c[pat0]
            m1 = pm[p1] + c[pat1]
            take1 = m1 < m0
            decision[t] = take1
            pm = np.where(take1, m1, m0)
            pm -= pm.min() # Keep the metrics bounded
        return decision, pm

    def _states(self, decision, end):
        """Traceback: the state after each step of the survivor path that ends in `end`"""
        if _accel.native is not None:
            return _accel.native.viterbi_traceback(decision, self._pred, end)[0]
        states = np.zeros(len(decision), dtype=np.int64)
        s_ = end
        for t in range(len(decision)-1, -1, -1):
            states[t] = s_
            s_ = self.prev[s_, decision[t, s_]]
        return states
