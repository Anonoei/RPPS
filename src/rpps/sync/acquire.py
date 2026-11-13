"""Acquire: streaming modulation/baud/cfo detection (the estimators are in rpps.analysis.modulation)"""
from types import SimpleNamespace

import numpy as np

from ..base import Stage
from ..analysis.rate import find_baud
from ..analysis.modulation import detect

class Acquire(Stage):
    """Detect as soon as the signal is there, then keep refining as the stream grows

    The matrix matched filter and the likelihood only need a few hundred symbols, so there is no
    fixed wait: blocks are held only until the symbol-rate line is significant (`lock` times the
    local floor, noise peaks at about 3-4), which can be the first block. Then everything held
    is released at once, and the estimates are re-run on the growing history (doubling each time, up
    to `history` samples) so baud and cfo tighten. Refinements retest only the detected
    modulation and roll-off, which is much cheaper.

    Calibrates meta.sps (to the detected one), so following stages get the fractional sps.

    lock: symbol-rate line strength required to detect
    max_wait: samples after which to detect regardless
    """
    __slots__ = ("lock", "max_wait", "history", "names", "betas")
    def __init__(self, lock=5.0, max_wait=2**17, history=2**16, names=None, betas=(0.2, 0.25, 0.35)):
        self.lock = lock
        self.max_wait = max_wait
        self.history = history
        self.names = names
        self.betas = betas

    def new_state(self, meta):
        return SimpleNamespace(chunks=[], count=0, next=0, done=False, held=True)

    def step(self, samples, st, meta):
        if st.done or len(samples) == 0:
            return samples
        st.chunks.append(samples)
        st.count += len(samples)
        x = np.concatenate(st.chunks) if len(st.chunks) > 1 else st.chunks[0]
        st.chunks = [x]
        det = meta.stream.det
        if det is None:
            if st.count < self.max_wait and find_baud(x, meta.Fs)[1] < self.lock:
                return samples[:0] # Nothing there (yet)
            det = detect(x, meta.Fs, self.names, self.betas)
            st.next = 2*st.count
        elif st.count >= st.next: # Refine: same modulation and roll-off
            scores = det.scores
            det = detect(x, meta.Fs, [det.mod], (det.beta,))
            det.scores = scores
            st.next = 2*st.count
        else:
            return samples
        meta.stream.det = det
        meta.calibrate_sps(det.sps)
        if st.count >= self.history:
            st.done = True
            st.chunks = []
        if st.held: # First detection: release what was held
            st.held = False
            return x
        return samples
