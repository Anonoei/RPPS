import numpy as np
import matplotlib.pyplot as plt

def survey(ax, sv, chosen=None):
    """The spectrum of a rp.analysis.Survey with every signal shaded and numbered (the chosen one stands out)"""
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    ax.clear()
    x = (sv.freq + sv.cf)/1e6
    ax.plot(x, sv.db, "-", c="c", lw=0.8)
    ax.axhline(sv.floor, c="gray", ls=":", lw=0.8)
    top = np.max(sv.db) + 3
    for i, s in enumerate(sv):
        picked = s is chosen
        ax.axvspan((s.lo + sv.cf)/1e6, (s.hi + sv.cf)/1e6, color="y" if picked else "m", alpha=0.35 if picked else 0.15)
        ax.annotate(f"{i}", ((s.mid + sv.cf)/1e6, top), ha="center", va="top", c="y" if picked else "w", weight="bold")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(sv.floor - 5, top + 2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(f"{len(sv)} signals, {sv.bin_hz/1e3:.1f} kHz bins")
    return ax
