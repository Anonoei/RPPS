import numpy as np

from .. import filters

def window(samps):
    return samps * np.hamming(len(samps))  # apply a Hamming window

def bins(count):
    return np.arange(0, count) - count/2

def fft(samps):
    return np.fft.fftshift(np.fft.fft(samps))

def freq_axis(count, Fs=2.0, cf=0.0):
    return np.arange(-Fs/2.0, Fs/2.0, Fs/count) + cf
