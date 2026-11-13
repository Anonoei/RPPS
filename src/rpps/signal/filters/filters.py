import numpy as np

from .impl.designer import designer
from .impl.remez import remez as _remez
from .filter import FIR

class ConvFilter(FIR):
    __slots__ = ("filt")
    def __init__(self, filt):
        self.filt = filt

    def make_taps(self, meta):
        return self.filt

    def run(self, samples):
        """One-shot (stateless) filtering, drops the edges"""
        return np.convolve(samples, self.filt, mode="valid")

def low_pass(taps, cutoff, fs=None, window=np.hamming):
    """Windowed-sinc low pass: `taps` coefficients, cut off at `cutoff` Hz of sample rate `fs` (None: normalised, Nyquist is 1)
    window: a function of the length, e.g. np.hamming (default), np.blackman, or `kaiser_window(beta)`"""
    return ConvFilter(designer(taps, cutoff, fs=fs, pass_zero=True, window=window))

def high_pass(taps, cutoff, fs=None, window=np.hamming):
    return ConvFilter(designer(taps, cutoff, fs=fs, pass_zero=False, window=window))

def band_pass(taps, cut1, cut2, fs=None, window=np.hamming):
    return ConvFilter(designer(taps, [cut1, cut2], fs=fs, pass_zero=False, window=window))

def band_stop(taps, cut1, cut2, fs=None, window=np.hamming):
    return ConvFilter(designer(taps, [cut1, cut2], fs=fs, pass_zero=True, window=window))

def kaiser_window(beta):
    """A Kaiser window of shape `beta`, as a function of the length"""
    return lambda n: np.kaiser(n, beta)

def kaiser_params(atten_db, width):
    """Taps and beta for a Kaiser-window FIR with `atten_db` stopband attenuation and a transition `width` wide
    (a fraction of the sample rate, e.g. 0.05)"""
    if atten_db > 50:
        beta = 0.1102*(atten_db - 8.7)
    elif atten_db >= 21:
        beta = 0.5842*(atten_db - 21)**0.4 + 0.07886*(atten_db - 21)
    else:
        beta = 0.0
    taps = int(np.ceil((atten_db - 7.95)/(2.285*2*np.pi*width))) + 1
    return taps | 1, beta

def low_pass_kaiser(cutoff, transition, atten_db=60, fs=1.0):
    """Low pass that meets a specification: stopband `atten_db` down, the transition (Hz) centred on `cutoff`"""
    taps, beta = kaiser_params(atten_db, transition/fs)
    return low_pass(taps, cutoff, fs, kaiser_window(beta))

def high_pass_kaiser(cutoff, transition, atten_db=60, fs=1.0):
    taps, beta = kaiser_params(atten_db, transition/fs)
    return high_pass(taps, cutoff, fs, kaiser_window(beta))

def remez(taps, bands, desired, weight=None, fs=1.0, type="bandpass"):
    """Equiripple (Parks-McClellan) FIR: the smallest possible maximum weighted error for `taps` coefficients

        remez(101, [0, 1000, 1500, 5000], [1, 0], fs=10e3)   # low pass: pass 0-1 kHz, stop 1.5 kHz up

    bands: band edges in Hz of `fs`, two per band; desired: one gain per band; weight: one per band (a heavier band gets less ripple);
    type: "bandpass", "hilbert" or "differentiator". Needs no scipy
    """
    return ConvFilter(np.asarray(_remez(taps, bands, desired, weight=weight, fs=fs, type=type), dtype=np.float32))
