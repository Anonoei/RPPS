import numpy as np
import matplotlib.pyplot as plt

from . import Meta
from . import identify

def draw(symbols, meta: Meta):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(np.real(symbols), np.imag(symbols), s=10, c="b")

    plt.grid(True)
    plt.title(f"{meta.mod} encoded data")
    plt.xlim(1.2, -1.2)
    plt.ylim(1.2, -1.2)
    ax.set_aspect('equal', adjustable='box')
