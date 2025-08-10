"""Frequency domain visualizations"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from ... import utils

from ..utils import fft, window, freq_axis, bins

def spectrogram(ax, samp_list, Fs=1, vbw_hz=None):
    """Plot spectrogram"""
    if ax is None:
        fig, ax = plt.subplot()
    x = freq_axis(len(samp_list[0]), Fs)

    zs = np.zeros((len(samp_list), len(x)))
    z_min = np.inf
    z_max = -np.inf
    for i, samps in enumerate(samp_list):
        zs[i] = utils.psd(window(samps), Fs, vbw_hz)
        z_min = min(np.append(zs[i], z_min))
        z_max = max(np.append(zs[i], z_max))

    ax.set_title("Spectrogram")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Time")
    ax.imshow(
        zs, aspect="auto", extent=x,
        cmap=colormaps.get("plasma"), shading="nearest",
        vmin=z_min, vmax=z_max
    )
