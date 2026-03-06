"""Frequency domain visualizations"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches # for draw_signals

from ... import utils
from ..utils import fft, freq_axis, bins

def psd(ax, samps, Fs=1.0, cf=0.0, vbw_hz=None):
    if ax is None:
        fig, ax = plt.subplots()
    N = len(samps)
    x = freq_axis(N, Fs, cf)
    y = utils.psd(samps, Fs, vbw_hz)

    if isinstance(ax, mpl.lines.Line2D):
        ax.set_data(x, y)
        line = ax
        ax = line.axes
    else:
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=10)
        ax.grid(True, alpha=0.2)
        ax.set_title("PSD")
        ax.set_title("Freq", loc="left")
        ax.set_title(f"{len(samps)} N", loc="right")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        line, = ax.plot(x, y, c="y")
    ax.set_xlim(x[0], x[-1])
    ymin = np.min(y)
    ymax = np.max(y)
    if ymin < np.min(line.get_ydata()) or ymax > np.max(line.get_ydata()):
        ax.set_ylim(np.min(y), np.max(y))
        ax.relim()
    return line

def psd_signals(ax, samps, signals, Fs=1.0, cf=0.0, vbw_hz = None):
    if ax is None:
        fig, ax = plt.subplots()
    x = freq_axis(len(samps), Fs, cf)
    y = utils.psd(samps, Fs, vbw_hz)
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=10)
    ax.grid(True, alpha=0.2)
    ax.set_title("PSD")
    ax.set_title("Freq", loc="left")
    ax.set_title(f"{len(samps)} N", loc="right")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_xlim(x[0], x[-1])
    line, = ax.plot(x, y, c="y", zorder=1)
    for signal in signals:
        left = cf + (signal.mid - signal.baud/2)
        # print(f"Drawing signal ({x[il]:0.3f}, {bottom:0.3f}),({x[il]+width:0.3f}, {bottom+height:0.3f})")
        ax.add_patch(patches.Rectangle((left, signal.bottom), signal.baud, signal.height, facecolor='blue', alpha=0.5, zorder=2))

def psdreal(ax, samps, Fs=1, cf=0, vbw_hz=None):
    if ax is None:
        fig, ax = plt.subplots()
    y = utils.psd(samps.real, Fs, vbw_hz)
    y = y[len(y)//2:]
    x = freq_axis(len(y), Fs, cf)
    # x = np.arange(0, Fs, Fs/len(y)) + cf

    if isinstance(ax, mpl.lines.Line2D):
        ax.set_data(x, y)
        line = ax
    else:
        line, = ax.plot(x, y, c="y")
        ax.grid(True)
        ax.set_title("PSD")
        ax.set_title(f"Freq [{Fs} Hz]", loc="left")
        ax.set_title(f"{len(samps)} N", loc="right")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_xlim(x[0], x[-1])
    return line

def psd_minmax(ax, samps, sr=1, vbw_hz=None, cache={}):
    if isinstance(ax, tuple):
        line, line_min, line_max = ax
        line = psd(line, samps, sr, vbw_hz)
    else:
        line = psd(ax, samps, sr, vbw_hz)
    x = line.get_xdata()
    y = line.get_ydata()
    if cache.get("min", None) is None:
        cache["min"] = np.copy(y)
    if cache.get("max", None) is None:
        cache["max"] = np.copy(y)
    mask = cache["min"]>y
    cache["min"][mask] = y[mask]
    mask = cache["max"]<y
    cache["max"][mask] = y[mask]

    if isinstance(ax, tuple):
        line_min.set_data(x, cache["min"])
        line_max.set_data(x, cache["max"])
    else:
        line_min, = ax.plot(x, cache["min"], c="b")
        line_max, = ax.plot(x, cache["max"], c="r")
    return line, line_min, line_max

def mag(ax, samps):
    """Plot FFT Magnitude"""
    if ax is None:
        fig, ax = plt.subplots()
    x = bins(len(samps))
    y = np.abs(fft(samps))
    if isinstance(ax, mpl.lines.Line2D):
        ax.set_data(x, y)
        line = ax
    else:
        line, = ax.plot(x, y, label="Magnitude")
        ax.title("Magnitude")
        plt.title("Freq", loc="left")
        plt.title(f"{len(samps)} N", loc="right")
        plt.xlabel("Index")
        plt.ylabel("Magnitude")
    return line

def pha(ax, samps):
    """Plot FFT phase"""
    if ax is None:
        fig, ax = plt.subplots()
    x = bins(len(samps))
    y = np.angle(fft(samps))

    if isinstance(ax, mpl.lines.Line2D):
        ax.set_data(x, y)
        line = ax
    else:
        line, = ax.plot(x, y, label="Phase")

        ax.title("Phase")
        ax.title("Freq", loc="left")
        ax.title(f"{len(samps)} N", loc="right")
        ax.xlabel("Bin")
        ax.ylabel("Phase")
    return line

def mag_phase(ax_mag, ax_pha, samps):
    fig, ax = plt.subplots(1,2)
    line_mag = mag(ax_mag, samps)
    line_pha = pha(ax_pga, samps)
    return line_mag, line_pha

def draw_signals(ax, signals, samps, Fs, cf):
    x = freq_axis(len(samps), Fs, cf)
