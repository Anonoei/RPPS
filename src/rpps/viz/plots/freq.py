"""Frequency domain visualizations"""
import numpy as np
import matplotlib.pyplot as plt

from ... import utils
from ..utils import fft, vbw, freq_axis, bins

def psd(ax, samps, sr=1, vbw_hz=None):
    if ax is None:
        fig, ax = plt.subplots()
    x = freq_axis(len(samps), sr)
    y = utils.psd(samps, sr)

    if vbw_hz is not None:
        y = vbw(y, sr, vbw_hz)

    ax.plot(x, y)
    ax.grid(True)
    ax.set_title("PSD")
    ax.set_title("Freq Domain", loc="left")
    ax.set_title(f"{len(samps)} samples", loc="right")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    return x, y

def magnitude(ax, samps):
    """Plot FFT Magnitude"""
    if ax is None:
        fig, ax = plt.subplots()
    x = bins(len(samps))
    y = np.abs(fft(samps))

    ax.plot(x, y, label="Magnitude")
    ax.title("Magnitude")
    plt.title("Freq Domain", loc="left")
    plt.title(f"{len(samps)} samples", loc="right")
    plt.xlabel("Index")
    plt.ylabel("Magnitude")


def phase(ax, samps):
    """Plot FFT phase"""
    if ax is None:
        fig, ax = plt.subplots()
    x = bins(len(samps))
    y = np.angle(fft(samps))

    ax.plot(x, y, label="Phase")

    ax.title("Phase")
    ax.title("Freq Domain", loc="left")
    ax.title(f"{len(samps)} samples", loc="right")
    ax.xlabel("Bin")
    ax.ylabel("Phase")

def mag_phase(samps):
    fig, ax = plt.subplots(1,2)
    magnitude(ax[0], samps)
    phase(ax[1], samps)
