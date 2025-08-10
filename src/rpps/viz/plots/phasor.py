import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def phasor(ax, syms):
    if ax is None:
        fig, ax = plt.subplots()
    snip = syms / np.max(syms.real)

    if isinstance(ax, mpl.lines.Line2D):
        ax.set_data(np.real(snip), np.imag(snip))
        line = ax
    else:
        line, = ax.plot(np.real(snip), np.imag(snip), ".", c="g")
        # ax.axhline(y=0, color="k")
        # ax.axvline(x=0, color="k")

        ax.grid(True)
        ax.set_title(f"Phasor Diagram")
        ax.set_title("Time Domain", loc="left")
        ax.set_title(f"{len(snip)} symbols", loc="right")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_xlim(1.2, -1.2)
        ax.set_ylim(1.2, -1.2)
        ax.set_aspect("equal", adjustable="box")
    return line
