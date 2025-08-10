from . import matrix

import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl

from . import colors

def Persistent(ax, psds, x=1001, y=600, style="cvec", cmap="hot"):
    if style == "dot":
        mat, (amp_min, amp_max) = matrix.dot(x, y, psds)
    elif style == "cdot":
        mat, (amp_min, amp_max) = matrix.cdot(x, y, psds)
    elif style == "vec":
        mat, (amp_min, amp_max) = matrix.vec(x, y, psds)
    elif style == "cvec":
        mat, (amp_min, amp_max) = matrix.cvec(x, y, psds)
    elif style == "svec":
        mat, (amp_min, amp_max) = matrix.svec(x, y, psds)
    mat = mat / np.max(mat)

    if isinstance(ax, mpl.image.AxesImage):
        ax.set_data(mat) # Only update the data if able
        im = ax
    else:
        cmap = colors.get(cmap)
        im = ax.imshow(mat, cmap=cmap,
            origin="lower", vmin=0, vmax=1,
            aspect="auto",
            interpolation="nearest", resample=False,
            animated=True, rasterized=True
        )
        ax.figure.colorbar(im, pad=0.01)
        x_mul = [0.0,0.25,0.5,0.75,1.0]
        y_mul = [0.0,0.2,0.4,0.6,0.8,1.0]

        amp_rng = abs(abs(amp_max) - abs(amp_min))
        amp_off = amp_min if amp_min < 0 else -amp_min
        x_tick = [x*m for m in x_mul]
        x_text = [f"{m-x/2:.1f}" for m in x_tick]
        y_tick = [y*m for m in y_mul]
        y_text = [f"{(amp_rng*m)+amp_off:.1f}" for m in y_mul]
        ax.set_xticks(x_tick, x_text)
        ax.set_yticks(y_tick, y_text)
        ax.grid(True, alpha=0.2)
    return im
