"""Time domain viz helpers"""
import numpy as np
import matplotlib.pyplot as plt

from . import Meta


def phasor(symbols, meta: Meta, ax=None):
    """Plot phasor diagram"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.scatter(np.real(symbols), np.imag(symbols), s=5, c="b")

    ax.grid(True)
    ax.set_title(f"Phasor Diagram ({meta.mod.short()})")
    ax.set_title("Time Domain", loc="left")
    ax.set_title(f"{len(symbols)} symbols", loc="right")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_xlim(1.2, -1.2)
    ax.set_ylim(1.2, -1.2)
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def quadrature(symbols, meta: Meta, ax=None):
    """Plot I and Q"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    real = np.real(symbols)
    imag = np.imag(symbols)

    base_i = np.cos(np.arange(len(symbols))) * 0.2
    base_q = np.sin(np.arange(len(symbols))) * 0.2

    ax.plot(base_i + base_q, "-", label="Carrier")
    ax.plot(real + imag, ".-", label="Constructed")

    plt.grid(True)
    ax.set_title(f"{meta.mod.short()} Quadrature Signal")
    ax.set_title("Time Domain", loc="left")
    ax.set_title(f"{len(symbols)} symbols", loc="right")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-5, 5)
    plt.legend()
    return fig, ax
