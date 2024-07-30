import numpy as np
import matplotlib.pyplot as plt

from . import Meta


def phasor(symbols, meta: Meta, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.scatter(np.real(symbols), np.imag(symbols), s=5, c="b")

    plt.grid(True)
    plt.title(f"Phasor Diagram ({meta.mod.short()})")
    plt.title("Time Domain", loc="left")
    plt.title(f"{len(symbols)} symbols", loc="right")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.xlim(1.2, -1.2)
    plt.ylim(1.2, -1.2)
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def quadrature(symbols, meta: Meta, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    real = np.real(symbols)
    imag = np.imag(symbols)

    base_i = np.cos(np.arange(len(symbols)))
    base_q = np.sin(np.arange(len(symbols)))

    # plt.plot(base_i, label="I")
    # plt.plot(base_q, label="Q")
    ax.plot(base_i + base_q, "-", label="Carrier")
    # plt.plot(real, label="Real")
    # plt.plot(imag, label="Imag")
    ax.plot(real + imag, ".-", label="Constructed")

    plt.grid(True)
    plt.title(f"{meta.mod.short()} Quadrature Signal")
    plt.title("Time Domain", loc="left")
    plt.title(f"{len(symbols)} symbols", loc="right")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.ylim(-5, 5)
    return fig, ax
