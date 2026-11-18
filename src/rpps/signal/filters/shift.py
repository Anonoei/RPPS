from types import SimpleNamespace

import numpy as np

from ... import _accel

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

    def run(self, samples, Fs=1.0):
        """One-shot (stateless) shift; Fs 1: fo is in cycles per sample"""
        t = np.arange(len(samples))/Fs
        return (samples * np.exp(2j*np.pi*self.fo*t)).astype(_cdtype(samples), copy=False)

    def step(self, samples, st, meta):
        inc = 2*np.pi*self.fo/meta.Fs
        meta.op.cf += self.fo
        if _accel.native is not None and samples.dtype == np.complex64:
            out = np.empty_like(samples) # a rotating phasor in C++, on several threads: no exponential per sample
            st.phase = _accel.native.rotate_into(np.ascontiguousarray(samples), out, st.phase, inc, 0)
            return out
        out = samples * np.exp(1j*(st.phase + inc*np.arange(len(samples))))
        st.phase = (st.phase + inc*len(samples)) % (2*np.pi)
        return out.astype(_cdtype(samples), copy=False)

class ShiftFreqIdx(ShiftFreq):
    """Shift by `fo` cycles per sample (fo is normalised to the sample rate)"""
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
