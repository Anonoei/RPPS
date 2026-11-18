"""Frequency domain visualizations"""
import numpy as np
import matplotlib.patches as patches

from ... import analysis
from ..base import line
from ..utils import fft, freq_axis, bins

def psd(ax, samps, Fs=1.0, cf=0.0, vbw_hz=None):
    """Power spectral density [dB] against frequency; pass the returned line back to update it"""
    from matplotlib.lines import Line2D
    x = freq_axis(len(samps), Fs, cf)
    y = analysis.psd(samps, Fs, vbw_hz)
    new = not isinstance(ax, Line2D)
    ln = line(ax, x, y, "PSD", "Freq", f"{len(samps)} N", "Frequency [Hz]", "Magnitude [dB]", grid=0.2, c="y", zorder=1)
    if new:
        ln.axes.locator_params(axis="x", nbins=5)
        ln.axes.locator_params(axis="y", nbins=10)
    lo, hi = ln.axes.get_ylim()
    if np.min(y) < lo or np.max(y) > hi:
        ln.axes.set_ylim(np.min(y), np.max(y))
    return ln

def psd_signals(ax, samps, signals, Fs=1.0, cf=0.0, vbw_hz=None):
    """The PSD with a box over each signal found (rpps.analysis.find_signals)"""
    ln = psd(ax, samps, Fs, cf, vbw_hz)
    for signal in signals:
        left = cf + (signal.mid - signal.baud/2)
        ln.axes.add_patch(patches.Rectangle((left, signal.bottom), signal.baud, signal.height, facecolor="blue", alpha=0.5, zorder=2))
    return ln

def psdreal(ax, samps, Fs=1, cf=0, vbw_hz=None):
    """PSD of the real part: the positive frequencies"""
    y = analysis.psd(samps.real, Fs, vbw_hz)
    y = y[len(y)//2:]
    return line(ax, freq_axis(len(y), Fs, cf), y, "PSD", f"Freq [{Fs} Hz]", f"{len(samps)} N", "Frequency [Hz]",
                "Magnitude [dB]", grid=1.0, c="y")

def psd_minmax(ax, samps, sr=1, vbw_hz=None, cache={}):
    if isinstance(ax, tuple):
        line, line_min, line_max = ax
        line = psd(line, samps, sr, vbw_hz=vbw_hz)
    else:
        line = psd(ax, samps, sr, vbw_hz=vbw_hz)
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
    """FFT magnitude"""
    return line(ax, bins(len(samps)), np.abs(fft(samps)), "Magnitude", "Freq", f"{len(samps)} N", "Bin", "Magnitude",
                label="Magnitude")

def pha(ax, samps):
    """FFT phase"""
    return line(ax, bins(len(samps)), np.angle(fft(samps)), "Phase", "Freq", f"{len(samps)} N", "Bin", "Phase",
                label="Phase")

def mag_phase(ax_mag, ax_pha, samps):
    """Magnitude and phase of samps on two axes"""
    return mag(ax_mag, samps), pha(ax_pha, samps)
