from types import SimpleNamespace

import numpy as np

from .sample import Sample, Meta
from ..filters.impl.designer import designer

class Resample(Sample):
    __slots__ = (
        "up", "down",
        "taps"
    )
    def __init__(self, up, down, taps=None):
        """Rational resampler by up/down
        taps: anti-image/anti-alias FIR length at the up-sampled rate [20*max(up, down)+1]
        """
        self.up = int(up)
        self.down = int(down)
        n = max(self.up, self.down)
        if taps is None:
            taps = 20*n+1
        # Cutoff is the Nyquist of the lower rate, at the up-sampled rate. Gain `up` makes up for the zero stuffing
        self.taps = self.up * designer(taps, 1/n, window=np.hamming, scale=True).astype(np.float64)

    def run(self, samples):
        raise NotImplementedError()

    def step(self, samples, st, meta: Meta):
        out = self.run(samples, st)
        meta.op.Fs *= self.up/self.down
        meta.op.sps *= self.up/self.down
        meta.delay += (len(self.taps)-1) / 2 / (meta.Fs * self.up)
        return out

class Polyphase(Resample):
    """Stateful FIR rational resampler. Output length tracks up/down across blocks"""
    def new_state(self, meta):
        hist = -(-len(self.taps)//self.up) + 1 # Input samples the filter can reach back to
        return SimpleNamespace(hist=None, nh=hist, phase=0)

    def run(self, samples, st=None):
        if st is None:
            st = self.new_state(None)
        up, down, L = self.up, self.down, len(self.taps)
        if st.hist is None:
            st.hist = np.zeros(st.nh, dtype=samples.dtype)
        if len(samples) == 0:
            return samples

        buf = np.concatenate((st.hist, samples))
        # Output m sits at position n = start + m*down of the zero-stuffed stream (input j at j*up). Only every up-th tap
        # meets a real sample: y = sum_i taps[p + i*up] * buf[q - i], q = n // up, p = n % up. So no zero-stuffed copy
        start = st.nh*up + st.phase # The first new sample sits at st.nh*up
        total = len(buf)*up
        n = np.arange(start, total, down)
        q, p = n//up, n % up
        K = -(-L//up)
        poly = np.zeros((up, K), dtype=np.float64) # poly[p, i] = taps[p + i*up]
        for ph in range(up):
            h = self.taps[ph::up]
            poly[ph, :len(h)] = h
        out = np.empty(len(n), dtype=np.result_type(buf.dtype, poly.dtype))
        for a in range(0, len(n), 1 << 16): # In chunks: (outputs, taps per phase) at a time
            idx = q[a:a + (1 << 16), None] - np.arange(K)[None, :]
            out[a:a + (1 << 16)] = np.sum(poly[p[a:a + (1 << 16)]]*buf[np.maximum(idx, 0)]*(idx >= 0), axis=1)

        st.phase = start + len(n)*down - total
        st.hist = buf[len(buf)-st.nh:]
        return out.astype(samples.dtype if np.iscomplexobj(samples) else out.dtype, copy=False)
