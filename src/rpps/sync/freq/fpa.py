import numpy as np

def fpa(samples, Ts, order=4, pad=4):
    """Frequency offset of M-PSK samples from the M-th power line. Returns the corrected samples and the offset [Hz]"""
    n = len(samples)
    spec = np.abs(np.fft.fft(samples**order * np.hanning(n), pad*n))
    k = int(np.argmax(spec))
    a, b, c = np.log(spec[(k-1) % len(spec)] + 1e-30), np.log(spec[k] + 1e-30), np.log(spec[(k+1) % len(spec)] + 1e-30)
    den = a - 2*b + c
    f = (k + (0.0 if den == 0 else 0.5*(a - c)/den))/(pad*n)
    f = (f - (1 if f > 0.5 else 0)) / order # cycles/sample
    return samples * np.exp(-2j*np.pi*f*np.arange(n)), f/Ts
