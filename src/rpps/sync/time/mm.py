import numpy as np
from ..sync import Sync, Meta

from ...sample import up

class MM:
    __slots__ = (
        "alpha", "est",
        "interp", "buffer",
        "count",
        "outs", "rail"
    )

    # fig, ax = plt.subplots(1)
    def __init__(self, alpha=0.3, est=0.0, interp=32):
        self.alpha = alpha
        self.est = est
        self.interp = up.Sinc(interp)
        self.buffer = None
        self.count = -2

        self.outs = np.zeros(3, dtype=np.complex64) # 0, -1, -2
        self.rail = np.zeros(3, dtype=np.complex64) # 0, -1, -2

    def run(self, meta: Meta):
        meta = meta + self.interp # Interpolate meta.obj
        opt_idx = int(self.est*self.interp.ratio) + meta.sps//2

        out = meta.obj[opt_idx]
        rail = int(np.real(out) > 0) + 1j*int(np.imag(out) > 0)
        self._clock(out, rail)
        x = (self.rail[0] - self.rail[2]) * np.conj(self.outs[1])
        y = (self.outs[0] - self.outs[2]) * np.conj(self.rail[1])
        mm_val = np.real(y-x)
        self.est += self.alpha*mm_val
        self.count += 1
        # print(f"mm_val: {mm_val} / est: {self.est}")
        return out

    def burst(self, meta: Meta):
        sps = int(round(meta.sps, 0))
        samples = np.copy(meta.obj)
        idx_in = sps + int(np.floor(self.est))
        if self.buffer is not None:
            samples = np.concat((self.buffer, samples), axis=0)
            self.buffer = None
        N = len(samples)
        sym_out = int(len(samples)//sps)
        out = np.zeros(sym_out, dtype=samples.dtype)
        skip = 2 if self.count == -2 else None

        for i in range(0, sym_out, 1):
            if idx_in > N:
                print(f"MM ended early at {i}/{sym_out}")
                out = out[:i]
                break
            elif idx_in-sps < 0:
                print(f"MM started with invalid idx {idx_in}-{sps}")
                break
            meta.obj = samples[idx_in-sps:idx_in]
            # print(f"Running samples[{i_in-sps}:{i_in}] of ({N}) [{meta.obj.shape}]")
            out[i] = self.run(meta)
            idx_in += sps + int(np.floor(self.est))
            self.est = self.est - np.floor(self.est)
        if (idx_in-sps) < len(samples):
            next_idx = idx_in - len(samples)
            self.buffer = samples[-next_idx:]
            # print(f"Next idx is at {next_idx}+{len(self.buffer)} ({idx_in}-{sps}:{len(samples)})")
        meta.op.Fs = meta.Fs / sps
        meta.op.sps = 1
        if skip is not None:
            out = out[skip:]
        meta.obj = out
        return meta

    def _clock(self, out, rail):
        self.outs = np.roll(self.outs, 1)
        self.outs[0] = out

        self.rail = np.roll(self.rail, 1)
        self.rail[0] = rail
