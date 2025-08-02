import numpy as np

def psd(samps, Fs=1):
    y = np.abs(np.fft.fft(samps))**2 / (len(samps)*Fs)
    y = np.fft.fftshift(10.0*np.log10(y))
    return y
