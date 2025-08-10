import numpy as np
from ..filters.impl import gaussian

def psd(samps, Fs=1, vbw_hz=None):
    y = np.abs(np.fft.fft(samps))**2 / (len(samps)*Fs)
    y = np.fft.fftshift(10.0*np.log10(y))
    if vbw_hz is not None:
        y = vbw(y, Fs, vbw_hz)
    return y

def vbw(samps, Fs, vbw):
    smooth = vbw/(Fs / len(samps))
    sigma = gaussian.fwhm2sigma(smooth)
    return gaussian.gaussian(samps, sigma)
