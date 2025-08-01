import numpy as np

def psd(samps, sr):
    y = np.abs(np.fft.fft(samps))
    y = y**2 / (len(samps) * sr) # Convert to power
    y = np.fft.fftshift(y) # shift
    y = 10.0 * np.log10(y) # Convert to dB
    return y
