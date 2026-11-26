from types import SimpleNamespace

import numpy as np

from ..base import BitStage
from ..code.coding import puncture_period, mismatch

class Puncture(BitStage):
    """Find where a punctured convolutional code's puncture pattern starts, and start the stream there

        aligned = llrs + rp.sync.Puncture(rp.code.load("conv/k7_171_133/3_4")) + ...   # then Viterbi

    Block estimation on channel bits: windows of `window` coded bits at fixed stream positions, each offset into the
    puncture period tried on one by decoding it and coding the decision again; the right offset reproduces what was
    received (only channel errors differ), a wrong one does not. rotations: the constellation's turns as bit transforms
    (rp.symbol.mapping.bit_rotations: the carrier's M-fold phase left by an M-th power estimate) tried with every offset;
    the one found is applied to everything after. The first window where the best hypothesis's mismatch is under
    DECISIVE of the best one not equivalent to it (within EQUIVALENT: a code that decodes inverted bits too) decides (the stream's start is where the timing and carrier before were settling:
    there every offset decodes badly, and the pick was chance); after `max_windows` the best summed over them. The bits
    before the offset are dropped, once; after that the stream passes through. Its estimate is in `stream.est[name]`
    (offset, the mismatch rate at each offset in the window that decided, windows).
    """
    DECISIVE = 0.5
    EQUIVALENT = 0.02

    def __init__(self, code, window=4000, name="puncture", max_windows=8, rotations=None):
        self.code, self.window, self.name = code, int(window), name
        self.period = puncture_period(code)
        self.max_windows = int(max_windows)
        self.rotations = list(rotations) if rotations else [None]
        self.k = len(self.rotations[0][0]) if self.rotations[0] is not None else 1

    def __str__(self):
        return f"Puncture({self.code.name})"

    def new_state(self, meta):
        return SimpleNamespace(pending=None, offset=None, rates=None, checked=0, total=None, windows=0, rot=None, held=None)

    def _rate(self, hard):
        k, n = mismatch(self.code, hard)
        return k/n if n else 1.0

    def _turn(self, x, rot):
        """x (a whole number of symbols' bits or LLRs) turned by rot = (perm, flip)"""
        if rot is None or (list(rot[0]) == list(range(self.k)) and not any(rot[1])):
            return x
        perm, flip = rot
        y = np.asarray(x).reshape(-1, self.k)[:, list(perm)]
        if y.dtype.kind == "f":
            return (y*np.where(flip, -1.0, 1.0)).reshape(-1).astype(np.asarray(x).dtype)
        return (y.astype(bool) ^ np.asarray(flip, bool)).reshape(-1).astype(np.asarray(x).dtype)

    def _whole(self, st, x):
        """x after what was held back, cut to whole symbols (the rest held for the next call)"""
        x = x if st.held is None else np.concatenate((st.held, x))
        n = len(x)//self.k*self.k
        st.held = x[n:] if n < len(x) else None
        return x[:n]

    def step(self, x, st, meta):
        x = np.asarray(x)
        if st.offset is not None:
            return self._turn(self._whole(st, x), st.rot) if self.k > 1 else x
        st.pending = x if st.pending is None else np.concatenate((st.pending, x))
        step = np.lcm(self.period, self.k)
        n = self.window//step*step
        hyps = [(o, r) for r in range(len(self.rotations)) for o in range(self.period)]
        while st.offset is None and len(st.pending) - st.checked >= n + self.period:
            c = st.checked # (a multiple of the period and the symbol from the stream's start: alike in every window)
            rates = []
            for o, r in hyps:
                seg = self._turn(st.pending[c:c + n], self.rotations[r])
                seg = np.concatenate((seg, st.pending[c + n:c + n + self.period]))[o:o + n]
                rates.append(self._rate(seg < 0 if seg.dtype.kind == "f" else seg.astype(bool)))
            st.total = np.asarray(rates) if st.total is None else st.total + rates
            st.windows += 1
            low = min(rates)
            # Of the hypotheses as good as the best (a turn a code is transparent to: inverted bits), the first: no turn
            # unless one decodes clearly better (what follows, a sync word's polarity, takes inverted bits)
            best = next(i for i, v in enumerate(rates) if v <= low + self.EQUIVALENT)
            others = [v for v in rates if v > low + self.EQUIVALENT]
            if len(hyps) == 1 or (others and rates[best] < self.DECISIVE*min(others)):
                st.offset, st.rates, st.rot = hyps[best][0], rates, hyps[best][1]
            elif st.windows >= self.max_windows:
                b = next(i for i, v in enumerate(st.total) if v <= min(st.total) + self.EQUIVALENT*st.windows)
                st.offset, st.rates, st.rot = hyps[b][0], list(st.total/st.windows), hyps[b][1]
            else:
                st.checked += n
        if st.offset is None and meta.flush and len(st.pending) >= self.period: # The stream's end: the best so far
            b = int(np.argmin(st.total)) if st.total is not None else 0
            st.offset, st.rot = hyps[b][0], hyps[b][1]
            st.rates = list(st.total/st.windows) if st.total is not None else None
        if st.offset is None:
            return x[:0]
        pending, st.pending = st.pending, None
        rot = self.rotations[st.rot]
        st.rot = rot
        out = self._turn(self._whole(st, pending), rot)[st.offset:] if self.k > 1 else pending[st.offset:]
        meta.stream.est[self.name] = dict(offset=st.offset, windows=st.windows, rotation=None if rot is None else
                                          {"perm": list(rot[0]), "flip": list(rot[1])},
                                          rates=None if st.rates is None else [round(float(r), 4) for r in st.rates])
        return out
