import numpy as np
from .. import filters

def psd(samps, Fs=1, vbw_hz=None):
    y = np.abs(np.fft.fft(samps))**2 / (len(samps)*Fs)
    y = np.fft.fftshift(10.0*np.log10(y))
    if vbw_hz is not None:
        y = vbw(y, Fs, vbw_hz)
    return y

def vbw(samps, Fs, vbw):
    smooth = vbw/(Fs / len(samps))
    sigma = filters.gaussian.fwhm2sigma(smooth)
    return filters.gaussian.gaussian(samps, sigma)
