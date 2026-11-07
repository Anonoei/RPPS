import numpy as np
import matplotlib.pyplot as plt

import _mod
import _plot

num_sym = 1024
sps = 1
points = 4
fs = 250_000

bits = _mod.get_bits(num_sym, points)
s = _mod.mod_bits(bits, points)

sn = _mod.awgn(s,0.1)
sp = _mod.phase_noise(s,0.1)
spn = _mod.phase_noise(sn,0.3)

fig, axs = plt.subplots(2,2)
fig.tight_layout()
t = np.arange(0, len(s)) - (len(s)//2)

_plot.phasor(s,   axs[0,0])
_plot.phasor(spn, axs[0,1])
_plot.phasor(sn,  axs[1,0])
_plot.phasor(sp,  axs[1,1])

axs[0,0].set_title("S")
axs[0,1].set_title("S+AWGN+Phase Noise")
axs[1,0].set_title("S+AWGN")
axs[1,1].set_title("S+Phase Noise")

plt.show()
