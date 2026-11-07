from types import SimpleNamespace

import numpy as np

from .sample import Sample, Meta

class Downsample(Sample):
    __slots__ = ("ratio")

class Decimate(Downsample):
    """Keep every `ratio`th sample. The phase carries across blocks, so any block length works"""
    def __init__(self, ratio: int):
        self.ratio = int(ratio)

    def new_state(self, meta):
        return SimpleNamespace(off=0) # Index in the next block of the next sample to keep

    def run(self, samples):
        """One-shot (stateless) decimation"""
        return samples[::self.ratio]

    def step(self, samples, st, meta):
        kept = len(range(st.off, len(samples), self.ratio))
        out = samples[st.off::self.ratio]
        st.off = st.off + kept*self.ratio - len(samples)
        meta.op.Fs /= self.ratio
        meta.op.sps /= self.ratio
        return out

def interpolate_stream(samples, st, step, kind="linear"):
    """Read `samples` at positions st.pos, st.pos + step, ... by interpolation, carrying the position and the unread
    tail in `st` (SimpleNamespace(tail=None, pos=0.0); for "cubic", pos starts at 1.0). step: input samples per output
    sample. kind: "linear", or "cubic" (Catmull-Rom: smoother, needs one sample before and two after)"""
    cubic = kind == "cubic"
    pre = 1 if cubic else 0
    if st.tail is None and pre and len(samples):
        samples = np.concatenate((samples[:1], samples)) # Nothing before the first sample: repeat it
        first = True
    else:
        first = False
    buf = samples if st.tail is None else np.concatenate((st.tail, samples))
    n = len(buf)
    last = n - 1 - (2 if cubic else 0) # Newest position that has what it needs
    count = int(np.floor((last - st.pos)/step)) + 1 if n and st.pos <= last else 0
    pos = st.pos + step*np.arange(count)
    i0 = pos.astype(np.int64)
    mu = (pos - i0).astype(np.float32)
    if not count:
        out = buf[:0]
    elif cubic:
        a, b, c, d = buf[i0 - 1], buf[i0], buf[i0 + 1], buf[i0 + 2]
        out = b + 0.5*mu*(c - a + mu*(2*a - 5*b + 4*c - d + mu*(3*(b - c) + d - a)))
    else:
        out = buf[i0]*(1 - mu) + buf[np.minimum(i0 + 1, n - 1)]*mu
    nxt = st.pos + step*count
    keep = max(0, min(int(nxt) - pre, n)) # Samples before this can't be reached any more
    st.tail = buf[keep:]
    st.pos = nxt - keep
    return out.astype(buf.dtype, copy=False)

class FractionalSpline(Downsample):
    __slots__ = ("ratio", "kind")
    """Fractional resampling by linear interpolation: ratio is the output rate over the input rate

    The read position carries across blocks, so any block length gives the same stream. There is no anti-alias
    filter: low-pass first (`rp.filters.low_pass`) when decimating a signal that isn't oversampled.
    """
    def __init__(self, ratio: float, kind="linear"):
        """kind: "linear", or "cubic" for a smoother result"""
        if not ratio > 0:
            raise ValueError("Resample ratio must be positive")
        if kind not in ("linear", "cubic"):
            raise ValueError("kind must be linear or cubic")
        self.ratio = ratio
        self.kind = kind

    def new_state(self, meta):
        # pos: where the next output sits in tail+block, in input samples
        return SimpleNamespace(tail=None, pos=1.0 if self.kind == "cubic" else 0.0)

    def run(self, samples):
        """One-shot (stateless) resampling"""
        return interpolate_stream(samples, self.new_state(None), 1/self.ratio, self.kind)

    def step(self, samples, st, meta):
        meta.op.Fs *= self.ratio
        meta.op.sps *= self.ratio
        return interpolate_stream(samples, st, 1/self.ratio, self.kind)
