import numpy as np
from ..sync import Sync, Meta

from ...sample import up
class MM:
    __slots__ = (
        "mu", "est", "interp",
        "_interp", "outs", "rail"
    )
    def __init__(self, mu, est, interp=32):
        self.mu = mu
        self.est = est
        self.interp = up.Sinc(interp)

        self._interp = interp
        self.outs = np.zeros(3, dtype=np.complex64) # 0, -1, -2
        self.rail = np.zeros(3, dtype=np.complex64) # 0, -1, -2

    def run(self, meta: Meta):
        meta = meta + self.interp # Interpolate meta.obj
        opt_idx = int(self.est*self._interp)
        out = meta.obj[opt_idx]
        rail = int(np.real(out) > 0) + 1j*int(np.imag(out) > 0)
        self._clock(out, rail)

        x = (self.rail[0] - self.rail[2]) * np.conj(self.outs[1])
        y = (self.outs[0] - self.outs[2]) * np.conj(self.rail[1])
        mm_val = np.real(y-x)
        self.est += self.mu*mm_val
        return out

    def burst(self, meta: Meta):
        samples = np.copy(meta.obj)
        Fs = meta.Fs
        sym_out = int(len(samples)//meta.sps)
        out = np.zeros(sym_out, dtype=samples.dtype)
        i_in = meta.sps
        for i in range(1, sym_out, 1):
            meta.obj = samples[i_in-meta.sps:i_in]
            out[i] = self.run(meta)
            i_in += meta.sps + int(np.floor(self.est))
        meta.obj = out
        meta.Fs = Fs / meta.sps
        meta.sps = 1
        return meta

    def _clock(self, out, rail):
        self.outs = np.roll(self.outs, 1)
        self.outs[0] = out

        self.rail = np.roll(self.rail, 1)
        self.rail[0] = rail
