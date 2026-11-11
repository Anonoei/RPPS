from types import SimpleNamespace

import numpy as np

from ...base import Meta
from .filter import Filter, FIR

def _cdtype(samples):
    """Complex type that keeps the precision of the samples (real ones become complex64)"""
    return np.result_type(samples.dtype, np.complex64)

class ShiftFreq(Filter):
    """Shift the spectrum by `fo` Hz. The oscillator phase carries across blocks"""
    __slots__ = ("fo")
    def __init__(self, fo):
        self.fo = fo

    def new_state(self, meta):
        return SimpleNamespace(phase=0.0)

    def run(self, samples, Fs):
        """One-shot (stateless) shift"""
        t = np.arange(len(samples))/Fs
        return (samples * np.exp(2j*np.pi*self.fo*t)).astype(_cdtype(samples), copy=False)

    def step(self, samples, st, meta):
        inc = 2*np.pi*self.fo/meta.Fs
        out = samples * np.exp(1j*(st.phase + inc*np.arange(len(samples))))
        st.phase = (st.phase + inc*len(samples)) % (2*np.pi)
        meta.op.cf += self.fo
        return out.astype(_cdtype(samples), copy=False)

class ShiftFreqIdx(ShiftFreq):
    """Shift by `fo` cycles per sample (fo is normalised to the sample rate)"""
    def run(self, samples, idx=None):
        """One-shot shift. idx: sample indices of `samples` (default: 0..N-1)"""
        t = np.arange(len(samples)) if idx is None else np.asarray(idx)
        return (samples * np.exp(2j*np.pi*self.fo*t)).astype(_cdtype(samples), copy=False)

    def step(self, samples, st, meta):
        inc = 2*np.pi*self.fo
        out = samples * np.exp(1j*(st.phase + inc*np.arange(len(samples))))
        st.phase = (st.phase + inc*len(samples)) % (2*np.pi)
        meta.op.cf += self.fo*meta.Fs
        return out.astype(_cdtype(samples), copy=False)

class ShiftTime(FIR):
    """Delay by a fraction of a sample (windowed-sinc fractional delay), carrying its history across blocks

    delay: samples, normally in [0, 1). The filter's own latency is (taps-1)/2 samples
    """
    __slots__ = ("pulse",)
    def __init__(self, delay, taps=21):
        pulse = np.arange(taps) - (taps-1)/2
        pulse = np.sinc(pulse - delay) # calc filter taps
        pulse *= np.hamming(taps)
        pulse /= np.sum(pulse)  # unity gain
        self.pulse = pulse

    def make_taps(self, meta):
        return self.pulse

    def run(self, samples):
        """One-shot (stateless) shift, drops the edges"""
        return np.convolve(samples, self.pulse, mode="valid")
