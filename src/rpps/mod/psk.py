import numpy as np
import matplotlib.pyplot as plt

from . import Modulation

class PSK(Modulation):
    def draw_refs(self, points: bool=True, ref: bool=True):
        if points:
            x = self.points.real()
            y = self.points.imag()
            plt.scatter(x=x, y=y, s=200, c="r")
            labels = self.maps[self.map].arr
            # Add labels using annotate()
            for i, label in enumerate(labels):
                plt.annotate(bin(label)[2:].zfill(self.constellation._bps), (x[i], y[i]), fontsize=20)

        if ref:
            angle = np.linspace(0, 2 * np.pi, 150)
            radius = 1
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            plt.plot(x, y, "g")
