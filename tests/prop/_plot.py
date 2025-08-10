import numpy as np

def phasor(syms, ax):
    snip = syms / np.real(np.max(syms))
    ax.plot(snip.real, snip.imag, ".")
    ax.grid(True)
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.set_aspect("equal")

def psd(t, samples, ax):
    _psd = np.abs(np.fft.fft(samples))**2 / (len(samples))
    _psd = np.fft.fftshift(10*np.log10(_psd))
    ax.plot(t, _psd)

def time(t, samples, ax):
    ax.plot(t, samples.real+samples.imag)

def timeIQ(t, samples, ax):
    ax.plot(t, samples.real)
    ax.plot(t, samples.imag)
