import numpy as np

from ._code import _code, register

@register("fsm")
class fsm(_code):
    """Trellis code given as a finite state machine: P25's 1/2 and 3/4 rate trellis codes, ATSC's 8VSB trellis code

    Each step takes k input bits (most significant first) in state s, sends the n-bit word output[s][u] (most
    significant first) and moves to next[s][u]. Tables are (states, 2^k), or flat lists of states*2^k (row = state).
        start: the state the encoder starts in (default 0); null: unknown (the decoder starts from any)
        tail: input words of zeros appended by the encoder (P25: one), decoded and dropped; end: the state the
              path must end in (null: any)
        symbols: {"bits": b, "levels": [...]} to decode received amplitudes instead of bits: each b-bit group of a
              word is sent as levels[group] (ATSC: 3 bits -> -7..7). decode() then takes the amplitudes, n/b per step,
              and the branch metric is the squared distance
    Bit decoding takes hard bits or LLRs (positive = 0). decode() takes one sequence, decode_batch() a (W, ...) array
    of independent ones (all the same length) and returns the input bits of each.
    After decode(), `errors` is the number of received bits (hard decided) that differ from the decoded path's words.
    """
    soft = True

    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["k"], obj["n"], obj["next"], obj["output"], obj.get("start", 0), obj.get("end"),
                   obj.get("tail", 0), obj.get("symbols"))

    def __init__(self, log, k, n, next_state, output, start=0, end=None, tail=0, symbols=None):
        nxt = np.asarray(next_state, dtype=np.int64)
        out = np.asarray(output, dtype=np.int64)
        U = 1 << k
        nxt, out = nxt.reshape(-1, U), out.reshape(-1, U)
        S = len(nxt)
        if out.shape != nxt.shape:
            raise ValueError(f"next {nxt.shape} and output {out.shape} tables differ")
        if nxt.min() < 0 or nxt.max() >= S or out.min() < 0 or out.max() >= 1 << n:
            raise ValueError("Table entries out of range")
        self.k, self.n, self.S, self.U = k, n, S, U
        self.next, self.out = nxt, out
        self.start, self.end, self.tail = start, end, int(tail)
        self.levels = None
        if symbols is not None:
            b = int(symbols["bits"])
            if n % b:
                raise ValueError(f"{n}-bit words don't split into {b}-bit symbols")
            self.sym_bits, self.levels = b, np.asarray(symbols["levels"], dtype=np.float64)
            if len(self.levels) != 1 << b:
                raise ValueError(f"{1 << b} levels needed for {b}-bit symbols")
        steps_out = n
        super().__init__(log, None, k, steps_out)
        # Predecessors of each state: every (state, input) pair that leads there
        pred = [[] for _ in range(S)]
        for s in range(S):
            for u in range(U):
                pred[nxt[s, u]].append((s, u))
        P = max(len(p) for p in pred)
        if min(len(p) for p in pred) == 0:
            raise ValueError("Some state can't be reached")
        self.pred_s = np.array([[p[i % len(p)][0] for i in range(P)] for p in pred]) # (S, P), padded by repeating
        self.pred_u = np.array([[p[i % len(p)][1] for i in range(P)] for p in pred])
        self.out_bits = ((out[..., None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.uint8) # (S, U, n)

    # ------------------------------------------------------------------ encoder
    def encode(self, bits):
        b = np.asarray(bits).astype(np.uint8).reshape(-1)
        b = np.concatenate((b[:len(b)//self.k*self.k], np.zeros(self.tail*self.k, dtype=np.uint8)))
        words = (b.reshape(-1, self.k) * (1 << np.arange(self.k - 1, -1, -1))).sum(axis=1)
        s = 0 if self.start is None else self.start
        out = np.empty((len(words), self.n), dtype=np.uint8)
        for t, u in enumerate(words):
            out[t] = self.out_bits[s, u]
            s = self.next[s, u]
        return out.reshape(-1).astype(bool)

    def modulate(self, bits):
        """Encoded bits -> the amplitudes the `symbols` mapping sends"""
        g = np.asarray(bits).astype(np.int64).reshape(-1, self.sym_bits)
        return self.levels[(g * (1 << np.arange(self.sym_bits - 1, -1, -1))).sum(axis=1)]

    # ------------------------------------------------------------------ decoder
    def _prepare(self, x, soft):
        """Per-step inputs (W, T, m) and a function of one step's inputs -> (W, S, U) branch costs"""
        W = x.shape[0]
        if self.levels is not None:
            m = self.n // self.sym_bits
            r = np.asarray(x, dtype=np.float64).reshape(W, -1, m)
            wl = self.levels[(self.out_bits.reshape(self.S, self.U, m, self.sym_bits) *
                              (1 << np.arange(self.sym_bits - 1, -1, -1))).sum(axis=3)] # (S, U, m)
            return r, lambda v: ((v[:, None, None, :] - wl[None])**2).sum(axis=3)
        llr = np.asarray(x, dtype=np.float64) if soft else 1.0 - 2.0*np.asarray(x, dtype=np.float64)
        ob = self.out_bits.astype(np.float64)
        return llr.reshape(W, -1, self.n), lambda v: np.einsum("wn,sun->wsu", v, ob) # Sum of the LLRs where the word has a 1

    def decode_batch(self, x, soft=None):
        x = np.asarray(x)
        if soft is None:
            soft = x.dtype.kind == "f"
        steps, cost_of = self._prepare(x, soft)            # (W, T, m)
        W, T = steps.shape[:2]
        pm = np.zeros((W, self.S)) if self.start is None else np.full((W, self.S), np.inf)
        if self.start is not None:
            pm[:, self.start] = 0.0
        dec = np.empty((T, W, self.S), dtype=np.int64)
        ps, pu = self.pred_s, self.pred_u
        rows = np.arange(W)[:, None, None]
        for t in range(T):
            cand = pm[:, ps] + cost_of(steps[:, t])[rows, ps, pu] # (W, S, P)
            j = np.argmin(cand, axis=2)
            dec[t] = j
            pm = np.take_along_axis(cand, j[..., None], axis=2)[..., 0]
            pm -= pm.min(axis=1, keepdims=True)
        s = np.full(W, self.end) if self.end is not None else np.argmin(pm, axis=1)
        words = np.empty((W, T), dtype=np.int64)
        path_out = np.empty((W, T), dtype=np.int64)
        ar = np.arange(W)
        for t in range(T - 1, -1, -1):
            j = dec[t, ar, s]
            u, prev = pu[s, j], ps[s, j]
            words[:, t] = u
            path_out[:, t] = self.out[prev, u]
            s = prev
        keep = T - self.tail
        bits = ((words[:, :keep, None] >> np.arange(self.k - 1, -1, -1)) & 1).astype(bool).reshape(W, -1)
        self._path_out = path_out
        if self.levels is None:
            rx = (x.reshape(W, T, self.n) < 0) if soft else x.reshape(W, T, self.n).astype(bool)
            sent = ((path_out[..., None] >> np.arange(self.n - 1, -1, -1)) & 1).astype(bool)
            self.errors = np.count_nonzero(rx != sent, axis=(1, 2))
        else:
            self.errors = None
        return bits

    def decode(self, bits, soft=None):
        x = np.asarray(bits)
        per = self.n // self.sym_bits if self.levels is not None else self.n
        x = x[:len(x)//per*per]
        return self.decode_batch(x[None], soft)[0]
