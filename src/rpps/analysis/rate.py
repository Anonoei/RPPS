"""Rate processing"""
import numpy as np

from ..math.interp import parabolic_offset

def find_baud(x, Fs, lo=None, hi=None, pad=4):
    """Find the symbol rate from IQ samples using the |x|^2 cyclostationary line

    lo/hi: search range in Hz [Fs/16, Fs/2.5]
    returns (baud, strength). Strength is the line over the local floor, a few is noise
    """
    n = len(x)
    if n < 256:
        return 0.0, 0.0
    lo = Fs/16 if lo is None else lo
    hi = Fs/2.5 if hi is None else hi
    z = np.abs(x)**2
    z = (z - z.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(z, pad*n))
    freq = np.fft.rfftfreq(pad*n, 1/Fs)
    k = min(pad*64+1, len(spec)//2*2 - 1)
    floor = np.convolve(spec, np.ones(k)/k, mode="same")[:len(spec)]
    ratio = spec/(floor + 1e-30)
    band = np.where((freq >= lo) & (freq <= hi), ratio, 0.0)
    k = int(np.argmax(band))
    d = parabolic_offset(*np.log(spec[k-1:k+2] + 1e-30)) if 0 < k < len(spec)-1 else 0.0
    return (k + d)*Fs/(pad*n), float(band[k])

def find_rate(syms):
    """Find symbol rate (as a fraction of the sample rate) from IQ samples"""
    return find_baud(syms, 1.0)[0]
