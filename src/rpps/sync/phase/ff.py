from types import SimpleNamespace

import numpy as np

from ...base import Stage, Meta
from ...signal.filters.filter import FIR

class FFCarrier(Stage):
    """Feed-forward carrier correction for M-fold symmetric (PSK) symbols. No loop

    Takes the modulation order M, the constellation phase, the frequency offset and the gain
    from the detection in Meta (meta.stream.det). Then per symbol:
        frequency: phase-continuous oscillator at the detected offset
        phase: Viterbi & Viterbi, the angle of the sum of y^M over the last `window` symbols, divided by M
        amplitude: the detected gain

    window: symbols averaged for the phase. Longer is quieter, shorter follows drift better
    """
    __slots__ = ("window",)

    def __init__(self, window=64):
        self.window = window

    def new_state(self, meta: Meta):
        return SimpleNamespace(n=0, last=None, hist=None, cm=0.0, mod=None)

    def step(self, y, st, meta: Meta):
        det = meta.stream.det
        if det is None or len(y) == 0: # Still acquiring
            return y[:0]
        if st.mod != det.mod: # Detection can change while it refines
            from ...analysis.modulation import candidates
            st.cm = np.angle(np.mean(candidates([det.mod])[det.mod]**det.M))
            st.mod = det.mod

        y = y / det.gain
        n = st.n + np.arange(len(y))
        st.n += len(y)
        y = y * np.exp(-2j*np.pi*det.cfo/det.baud*n) # Frequency
        if det.M == 1:
            return y

        p = y**det.M
        taps = np.ones(self.window)
        s, st.hist = FIR.stream(p, taps, st.hist)
        ang = np.angle(s)
        if st.last is not None:
            ang = np.unwrap(np.r_[st.last, ang])[1:]
        else:
            ang = np.unwrap(ang)
        st.last = ang[-1]
        return y * np.exp(-1j*(ang - st.cm)/det.M)
