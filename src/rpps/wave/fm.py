from types import SimpleNamespace

import numpy as np

from ..base import Stage

class FM(Stage):
    """Frequency modulator: a shaped level signal becomes a constant-envelope signal whose frequency follows it. The phase is
    continuous across samples and blocks, so FSK, GFSK, GMSK, 4FSK and C4FM are this and a pulse

        levels + Zeros(sps) + pulse + FM(h)          # a held level a is a tone of a*h/step cycles per symbol

    h: modulation index, the frequency spacing of neighbouring levels in cycles per symbol (0.5: MSK; BLE 0.5; 4FSK C4FM 1/3...).
    step: the distance between neighbouring levels of the signal (2 for levels -1 +1 or -3 -1 +1 +3). The signal must be at
    meta.sps >= 1 samples per symbol, and real; its pulse should leave a held level at unit gain (rp.signal.filters shaping `rect`,
    `gaussian`).
    """
    def __init__(self, h=0.5, step=2.0):
        self.h, self.level_step = float(h), float(step) # not `step`: that is the Stage method

    def new_state(self, meta):
        return SimpleNamespace(phase=0.0)

    def step(self, samples, st, meta):
        if meta.sps < 1:
            raise ValueError(f"FM needs samples per symbol >= 1, got {meta.sps}: set sps on the stream or upsample first")
        x = np.asarray(samples)
        if x.dtype.kind == "c":
            x = x.real
        inc = 2*np.pi*self.h/(self.level_step*meta.sps)*x.astype(np.float64)
        ph = st.phase + np.cumsum(inc)
        if len(ph):
            st.phase = float(ph[-1] % (2*np.pi))
        return np.exp(1j*ph).astype(np.complex64)

def fm_gain(h, sps, step=2.0):
    """The `FMDemod` gain that gives back the levels `FM(h, step)` was fed at `sps` samples per symbol"""
    return -step*sps/(2*np.pi*h)

class FMDemod(Stage):
    """Quadrature FM demodulator as a stream stage: the last sample carries across blocks"""
    __slots__ = ("gain",)
    def __init__(self, gain=0.5):
        self.gain = gain

    def new_state(self, meta):
        return SimpleNamespace(last=None)

    def step(self, samples, st, meta):
        if len(samples) == 0:
            return np.zeros(0, dtype=np.float32)
        last = samples[0] if st.last is None else st.last
        st.last = samples[-1]
        z = np.concatenate(([last], samples))
        return self.gain * np.angle(z[:-1] * np.conj(z[1:]))
