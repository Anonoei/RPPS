import numpy as np
import matplotlib.pyplot as plt

from meta import Meta

def draw(data, meta: Meta):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(np.real(data), np.imag(data), '.')

    plt.grid(True)
    plt.title(f"{meta.mod} encoded data")
    plt.xlim(1.2, -1.2)
    plt.ylim(1.2, -1.2)
    ax.set_aspect('equal', adjustable='box')

    # Draw unit circle
    angle = np.linspace( 0 , 2 * np.pi , 150 )
    radius = 1
    x = radius * np.cos( angle )
    y = radius * np.sin( angle )
    plt.plot(x, y)
