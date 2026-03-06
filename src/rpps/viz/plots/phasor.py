import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

def phasor(ax, syms):
    if ax is None:
        fig, ax = plt.subplots()
    # norm = np.max([syms.real, syms.imag])
    # snip = syms / norm
    snip = syms

    if isinstance(ax, Line2D):
        line = ax
        line.set_data(np.real(snip), np.imag(snip))
        ax = line.axes
    else:
        line, = ax.plot(np.real(snip), np.imag(snip), ".", c="g")
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=5)

        ax.grid(True)
        ax.set_title("Phasor")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        # ax.set_xlim(1.2, -1.2)
        # ax.set_ylim(1.2, -1.2)
        ax.set_aspect("equal", adjustable="box")
    # ax.set_title(f"{norm:.3f}", loc="left")
    amax = np.max((snip.real, snip.imag, -snip.real, -snip.imag))
    if amax > ax.get_xlim()[1] or amax > ax.get_ylim()[1]:
        ax.set_xlim(-amax, amax)
        ax.set_ylim(-amax, amax)
    ax.set_title(f"{len(snip)} N", loc="right")
    return line
