"""Frequency domain visualizations"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from . import Meta
from . import Format

def _window(symbols):
    return symbols * np.hamming(len(symbols))  # apply a Hamming window

def _pre_process(symbols, meta: Meta):
    sample_rate = meta.freq.fields.get("SampleRate", 300)  # sample rate
    samps = len(symbols)  # number of samples to simulate

    y = _window(symbols)
    y = np.fft.fft(y)

    x = np.arange(sample_rate / -2.0, sample_rate / 2.0, sample_rate / samps)  # start, stop, step
    if meta.freq.fields.get("CenterFreq", None) is not None:
        x = x + meta.freq.fields["CenterFreq"]
    return x, y

def psd(symbols, meta: Meta, ax=None, _cache={}):
    """Plot Power-Spectral-Density"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    x, y = _pre_process(symbols, meta)

    y = np.abs(np.fft.fftshift(y)) **2 / (len(symbols) * (meta.freq.fields.get("SampleRate", 300))) # Convert to power
    y = 10.0 * np.log10(y) # Convert to log

    if _cache.get("x", None) == (x[0], x[-1]):
        _cache["min"] = min(np.append(y, np.array([_cache.get("min", np.inf)])))
        _cache["max"] = max(np.append(y, np.array([_cache.get("max", -np.inf)])))
    else:
        _cache["x"] = (x[0], x[-1])
        _cache["min"] = min(y)
        _cache["max"] = max(y)
    plt.plot(x, y)

    ax.set_ylim(_cache["min"], _cache["max"])

    plt.grid(True)
    ax.set_title("PSD")
    ax.set_title("Freq Domain", loc="left")
    ax.set_title(f"{len(symbols)} symbols", loc="right")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    plt.grid(True)
    return x, y


def magnitude(symbols, meta: Meta, ax=None):
    """Plot FFT Magnitude"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    x, y = _pre_process(symbols, meta)

    y = np.abs(y)
    plt.plot(x, y, label="Magnitude")

    plt.title("Magnitude")
    plt.title("Freq Domain", loc="left")
    plt.title(f"{len(symbols)} symbols", loc="right")
    plt.xlabel("Index")
    plt.ylabel("Magnitude")


def phase(symbols, meta: Meta, ax=None):
    """Plot FFT phase"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    x, y = _pre_process(symbols, meta)

    y = np.angle(y)
    plt.plot(x, y, label="Phase")

    plt.title("Phase")
    plt.title("Freq Domain", loc="left")
    plt.title(f"{len(symbols)} symbols", loc="right")
    plt.xlabel("Index")
    plt.ylabel("Phase")

def spectrogram(symbol_list, meta: Meta, fmt: Format, ax=None):
    """Plot spectrogram"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    x, _ = _pre_process(symbol_list[0], meta)
    y = [t * fmt.block_time for t in range(0, fmt.blocks)]
    z = []
    z_min = np.inf
    z_max = -np.inf
    for sym in symbol_list:
        _, z_item = _pre_process(sym, meta)
        z_item = np.abs(z_item) ** 2 / (len(sym) * (meta.freq.fields.get("SampleRate", 300)))
        z_item = 10.0 * np.log10(z_item)
        z.append(z_item)
        z_min = min(np.append(z_item, z_min))
        z_max = max(np.append(z_item, z_max))

    ax.set_title("Spectrogram")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Time")
    ax.pcolormesh(
        x, y, z,
        rasterized=True, cmap=colormaps.get("plasma"), shading="nearest",
        vmin=z_min, vmax=z_max)
