from types import SimpleNamespace

import numpy as np

from ..base import BitStage

class Puncture(BitStage):
    """Find where a punctured convolutional code's puncture pattern starts, and start the stream there

        aligned = llrs + rp.sync.Puncture(rp.code.load("conv", "k7_171_133/3_4")) + ...   # then Viterbi

    Block estimation on channel bits: once `window` coded bits are in, each offset into the puncture period is tried by
    decoding that window and coding the decision again; the right offset reproduces what was received (only channel errors
    differ), a wrong one does not. The bits before the best offset are dropped, once; after that the stream passes through.
    Its estimate is in `stream.est[name]` (offset, the mismatch rate at each offset).
    """
    def __init__(self, code, window=4000, name="puncture"):
        self.code, self.window, self.name = code, int(window), name
        p = getattr(code._dec, "puncture", None)
        self.period = int(p.sum()) if p is not None else code._dec.den

    def __str__(self):
        return f"Puncture({self.code.name})"

    def new_state(self, meta):
        return SimpleNamespace(pending=None, offset=None, rates=None)

    def _rate(self, hard):
        enc, dec = self.code._enc, self.code._dec
        u = dec.decode(hard.astype(np.uint8))
        e = enc.encode(u.astype(bool))
        n = min(len(e), len(hard))
        return float(np.mean(e[:n] != hard[:n].astype(bool))) if n else 1.0

    def step(self, x, st, meta):
        x = np.asarray(x)
        if st.offset is not None:
            return x
        st.pending = x if st.pending is None else np.concatenate((st.pending, x))
        need = self.window + self.period
        if len(st.pending) < need:
            return x[:0]
        hard = st.pending < 0 if st.pending.dtype.kind == "f" else st.pending.astype(bool)
        n = self.window//self.period*self.period
        st.rates = [self._rate(hard[o:o + n]) for o in range(self.period)]
        st.offset = int(np.argmin(st.rates))
        out, st.pending = st.pending[st.offset:], None
        meta.stream.est[self.name] = dict(offset=st.offset, rates=[round(r, 4) for r in st.rates])
        return out
