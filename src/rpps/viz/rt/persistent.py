from . import matrix

import numpy as np
import matplotlib.colors as colors

from . import colors

def Persistent(ax, psds, x=1001, y=60, style="vec"):
    if style == "dot":
        mat, (amp_min, amp_max) = matrix.dot(x, y, psds)
    elif style == "vec":
        mat, (amp_min, amp_max) = matrix.vec(x, y, psds)
    elif style == "svec":
        mat, (amp_min, amp_max) = matrix.svec(x, y, psds)

    x_mul = [0.0,0.25,0.5,0.75,1.0]
    y_mul = [0.0,0.2,0.4,0.6,0.8,1.0]

    amp_rng = abs(abs(amp_max) - abs(amp_min))
    amp_off = amp_min if amp_min < 0 else -amp_min
    x_tick = [x*m for m in x_mul]
    x_text = [f"{m-x/2:.1f}" for m in x_tick]
    y_tick = [y*m for m in y_mul]
    y_text = [f"{(amp_rng*m)+amp_off:.1f}" for m in y_mul]

    mat = mat / np.max(mat)

    im = ax.imshow(mat, cmap=colors.hot,
        origin="lower", vmin=0, vmax=1,
        aspect="auto",
        interpolation="none", resample=False,
        animated=True
    )
    # ax.figure.colorbar(im, pad=0.01)
    ax.set_xticks(x_tick, x_text)
    ax.set_yticks(y_tick, y_text)
    ax.grid(True, alpha=0.2)
    return im
