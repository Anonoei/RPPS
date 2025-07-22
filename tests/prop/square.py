import numpy as np
import matplotlib.pyplot as plt

import _mod
import _plot

num_sym = 1024
points = 8

bits = _mod.get_bits(num_sym, points)
s = _mod.mod_bits(bits, points)
s = _mod.awgn(s,0.05)
s = _mod.phase_noise(s,0.05)

fig, axs = plt.subplots(2,4)
t = np.arange(0, num_sym) - (num_sym//2)

s2 = s**2
s4 = s2**2
s8 = s4**2

_plot.phasor(s,  axs[0,0])
_plot.phasor(s2, axs[0,1])
_plot.phasor(s4, axs[0,2])
_plot.phasor(s8, axs[0,3])

_plot.psd(t,s,  axs[1,0])
_plot.psd(t,s2, axs[1,1])
_plot.psd(t,s4, axs[1,2])
_plot.psd(t,s8, axs[1,3])

print(f"s0: {np.var(s)}")
print(f"s2: {np.var(s2)}")
print(f"s4: {np.var(s4)}")
print(f"s8: {np.var(s8)}")

axs[0,0].set_title("S")
axs[0,1].set_title("S^2")
axs[0,2].set_title("S^4")
axs[0,3].set_title("S^8")

plt.show()
