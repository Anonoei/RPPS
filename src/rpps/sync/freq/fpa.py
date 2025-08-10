import numpy as np

def fpa(samples, Ts):
    Fs = 1/Ts
    l = len(samples)
    frq = np.fft.fftshift(np.abs(np.fft.fft(samples**4)))
    f_max = np.argmax(frq)

    t = np.arange(0, Ts*l, Ts)
    return samples * np.exp(-1j*2*np.pi*f_max*t/4.0)
