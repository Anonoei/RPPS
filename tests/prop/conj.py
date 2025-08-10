import numpy as np
import matplotlib.pyplot as plt

import _mod
import _plot

num_sym = 64
points = 4

bits = _mod.get_bits(num_sym, points)
s = _mod.mod_bits(bits, points)
s = _mod.awgn(s)
s = _mod.phase_noise(s)

fig, axs = plt.subplots(2,2)
t = np.arange(0, num_sym) - (num_sym//2)

spc = s + np.conj(s)

_plot.phasor(s,  axs[0,0])
_plot.phasor(spc, axs[0,1])

_plot.time(t,s,  axs[1,0])
_plot.time(t,spc, axs[1,1])

plt.show()
