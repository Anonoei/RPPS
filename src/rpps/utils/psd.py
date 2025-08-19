import numpy as np
from ..filters.impl import gaussian

def psd(samps, Fs: float = 1, vbw_hz=None):
    """Perform Power Spectral Density"""
    y = np.abs(np.fft.fft(samps))**2 / (len(samps)*Fs)
    y = np.fft.fftshift(10.0*np.log10(y))
    if vbw_hz is not None:
        smooth = vbw_calc(len(samps), Fs, vbw_hz)
        y = vbw(y, smooth)
    return y

def vbw_calc(N: int, Fs: float, vbw_hz: float):
    """Convert Hz to gaussian sigma"""
    return vbw_hz/(Fs / N)

def vbw(samps, smooth: float):
    """Perform gaussian smoothing"""
    return gaussian.gaussian(samps, gaussian.fwhm2sigma(smooth))
