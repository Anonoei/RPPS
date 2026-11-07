from types import SimpleNamespace

import numpy as np
from ...base import Meta, Stage

from ...sample import up

class MM(Stage):
    """Mueller & Muller timing recovery

    The whole block is interpolated once, then the optimal sample is picked from
    the interpolated block. The timing estimate is kept in [0, 1) (fractional sample
    offset); its integer part is moved into the input index.

    Settings live here, the loop state (est, history...) lives in `meta.state_of(mm)`.

    ted: "mm" (Mueller & Muller, decision directed) or "gardner" (needs no decisions, works for any modulation,
        wants at least 2 samples per symbol)
    biphase: Biphase (Manchester) symbols, after their matched filter, have two power
        peaks per symbol and M&M is happy at the weaker one. Use early-late on power,
        and hop half a symbol if the other peak is stronger.
    """
    __slots__ = (
        "alpha", "est",
        "interp", "biphase", "ted"
    )
    GUARD = 8 # Input samples at the block end the sinc interpolation can't be trusted for
    WARMUP = 2 # Outputs dropped while the M&M history fills

    def __init__(self, alpha=0.3, est=0.0, interp=32, biphase=False, ted="mm"):
        if ted not in ("mm", "gardner"):
            raise ValueError(f"ted must be mm or gardner, not {ted!r}")
        self.ted = ted
        self.alpha = alpha
        self.est = est # Initial estimate
        self.interp = up.Sinc(interp)
        self.biphase = biphase

    def new_state(self, meta: Meta):
        return SimpleNamespace(
            est=self.est,
            buffer=None, pos=meta.isps,
            count=0, skip=0 if self.biphase else self.WARMUP,
            outs=np.zeros(3, dtype=np.complex64), # 0, -1, -2
            rail=np.zeros(3, dtype=np.complex64), # 0, -1, -2
            pw=0.0, pw_half=0.0
        )

    def step(self, samples, st, meta: Meta):
        sps = meta.isps
        if sps < 1:
            raise ValueError(f"MM needs meta.sps >= 1, got {meta.sps}")
        ratio = self.interp.ratio
        samples = np.copy(samples)
        if st.buffer is not None:
            samples = np.concatenate((st.buffer, samples), axis=0)
        N = len(samples)
        out = []
        if N > 0:
            interp = self.interp.run(samples) # Interpolate once, not per symbol
            i_in = st.pos
            while i_in + self.GUARD < N:
                opt_idx = i_in * ratio + int(st.est * ratio)
                val = interp[opt_idx]
                pw = abs(val)**2
                st.pw = pw if st.count == 0 else st.pw + 0.02 * (pw - st.pw)
                if self.biphase:
                    half = interp[opt_idx - (sps * ratio) // 2]
                    st.pw_half = abs(half)**2 if st.count == 0 else st.pw_half + 0.02 * (abs(half)**2 - st.pw_half)
                    if st.count > 100 and st.pw_half > 1.3 * st.pw:
                        # Locked to the wrong half-symbol; hop to the stronger peak
                        i_in -= sps // 2
                        st.pw = st.pw_half = max(st.pw, st.pw_half)
                        continue
                out.append(val)
                if self.biphase:
                    # Early-late on power, amplitude independent
                    d = ratio * max(1, sps // 8)
                    early = abs(interp[opt_idx - d])**2
                    late = abs(interp[opt_idx + d])**2
                    err = (late - early) / (early + late + 1e-12)
                elif self.ted == "gardner":
                    mid = interp[opt_idx - (sps * ratio) // 2]
                    self._clock(st, val, 0)
                    err = -np.real((st.outs[0] - st.outs[1]) * np.conj(mid)) / (st.pw + 1e-12)
                else:
                    self._clock(st, val, int(np.real(val) > 0) + 1j*int(np.imag(val) > 0))
                    err = self._error(st) / (np.sqrt(st.pw) + 1e-12) # Error scales with amplitude
                st.est += self.alpha * err
                st.count += 1

                step = int(np.clip(np.floor(st.est), -(sps//2), sps//2)) # Bound a runaway loop
                st.est -= step # Integer part goes in the input index only
                i_in += sps + step

            # Keep the history that the next block's window still needs
            keep = max(0, min(i_in - sps, N))
            st.buffer = samples[keep:]
            st.pos = i_in - keep

        out = np.array(out, dtype=samples.dtype)
        if st.skip and len(out):
            drop = min(st.skip, len(out))
            out = out[drop:]
            st.skip -= drop
        meta.op.Fs = meta.Fs / sps
        meta.op.sps = 1
        return out

    @staticmethod
    def _error(st):
        x = (st.rail[0] - st.rail[2]) * np.conj(st.outs[1])
        y = (st.outs[0] - st.outs[2]) * np.conj(st.rail[1])
        return np.real(y - x)

    @staticmethod
    def _clock(st, out, rail):
        st.outs = np.roll(st.outs, 1)
        st.outs[0] = out

        st.rail = np.roll(st.rail, 1)
        st.rail[0] = rail
