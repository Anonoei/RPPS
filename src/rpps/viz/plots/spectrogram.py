"""Frequency domain visualizations"""
import numpy as np
from matplotlib import colormaps

from ... import analysis

from ..utils import window, freq_axis
from ..base import axes

def spectrogram(ax, samp_list, Fs=1, vbw_hz=None):
    """Plot spectrogram"""
    ax = axes(ax)
    x = freq_axis(len(samp_list[0]), Fs)

    zs = np.array([analysis.psd(window(samps), Fs, vbw_hz) for samps in samp_list])

    ax.set_title("Spectrogram")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Time")
    ax.imshow(
        zs, aspect="auto", extent=(x[0], x[-1], len(samp_list), 0), # one row per block, the first on top
        cmap=colormaps.get("plasma"), interpolation="nearest", vmin=zs.min(), vmax=zs.max()
    )
