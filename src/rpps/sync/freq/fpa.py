import numpy as np

from ...math.interp import peak

def fpa(samples, Ts, order=4, pad=4):
    """Frequency offset of M-PSK samples from the M-th power line. Returns the corrected samples and the offset [Hz]"""
    n = len(samples)
    spec = np.abs(np.fft.fft(samples**order * np.hanning(n), pad*n))
    k, d = peak(spec, circular=True)
    f = (k + d)/(pad*n)
    f = (f - (1 if f > 0.5 else 0)) / order # cycles/sample
    return samples * np.exp(-2j*np.pi*f*np.arange(n)), f/Ts
