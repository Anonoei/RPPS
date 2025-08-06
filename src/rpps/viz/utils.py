import numpy as np

from .. import filters

def window(samps):
    return samps * np.hamming(len(samps))  # apply a Hamming window

def bins(count):
    np.arange(0, count) - count/2

def fft(samps):
    return np.fft.fftshift(np.fft.fft(samps))

def freq_axis(count, sr, cf=0):
    return np.arange(-sr/2.0, sr/2.0, sr/count) + cf
