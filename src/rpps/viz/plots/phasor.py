import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

LIM = 1.6 # Axis limit: the unit-rms symbols sit near 1 with noise around them, and the margin keeps their markers (which have a size) from clipping

def phasor(ax, syms, points=None):
    """Symbols on the I/Q plane, scaled to unit rms. points: the ideal constellation, drawn as crosses (once)"""
    if ax is None:
        fig, ax = plt.subplots()
    norm = np.sqrt(np.mean(np.abs(syms)**2)) if len(syms) else 1.0 # Rms, so a few noise outliers don't shrink the rest
    snip = syms / (norm or 1.0)
    dots = dict(ms=3, alpha=0.4) if len(snip) > 500 else dict(alpha=1.0)

    if isinstance(ax, Line2D):
        line = ax
        line.set_data(np.real(snip), np.imag(snip))
        ax = line.axes
    else:
        line, = ax.plot(np.real(snip), np.imag(snip), ".", c="g", **dots)
        if points is not None:
            pts = np.asarray(points)
            pts = pts/np.sqrt(np.mean(np.abs(pts)**2))
            ax.plot(pts.real, pts.imag, "x", c="w", ms=8, zorder=3)
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
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    # if amax > ax.get_xlim()[1] or amax > ax.get_ylim()[1]:
    ax.set_title(f"{len(snip)} N", loc="right")
    return line
