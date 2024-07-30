import numpy as np
import matplotlib.pyplot as plt

from . import Meta

def _window(symbols):
    return symbols * np.hamming(len(symbols))  # apply a Hamming window

def _time_to_freq(symbols):
    Fs = 300  # sample rate
    Ts = 1 / Fs  # sample period
    N = len(symbols)  # number of samples to simulate

    x = _window(symbols)

    PSD = np.abs(np.fft.fft(x)) ** 2 / (N * Fs)
    PSD_log = 10.0 * np.log10(PSD)
    PSD_shifted = np.fft.fftshift(PSD_log)

    f = np.arange(Fs / -2.0, Fs / 2.0, Fs / N)  # start, stop, step

    return f, PSD_shifted

def psd(symbols, meta: Meta, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    f, PSD_shifted = _time_to_freq(symbols)

    plt.plot(f, PSD_shifted)

    plt.grid(True)
    plt.title(f"PSD")
    plt.title("Freq Domain", loc="left")
    plt.title(f"{len(symbols)} symbols", loc="right")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)


def magnitude(symbols, meta: Meta, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    Fs = 1  # Hz
    N = len(symbols)
    S = _window(np.fft.fftshift(np.fft.fft(symbols)))
    mag = np.abs(S)
    f = np.arange(Fs / -2, Fs / 2, Fs / N)

    plt.plot(f, mag, label="Magnitude")

    plt.title("Magnitude")
    plt.title("Freq Domain", loc="left")
    plt.title(f"{N} symbols", loc="right")
    plt.xlabel("Index")
    plt.ylabel("Magnitude")


def phase(symbols, meta: Meta, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    Fs = 1  # Hz
    N = len(symbols)
    S = _window(np.fft.fftshift(np.fft.fft(symbols)))
    pha = np.angle(S)
    f = np.arange(Fs / -2, Fs / 2, Fs / N)

    plt.plot(f, pha, label="Phase")

    plt.title("Phase")
    plt.title("Freq Domain", loc="left")
    plt.title(f"{N} symbols", loc="right")
    plt.xlabel("Index")
    plt.ylabel("Phase")
