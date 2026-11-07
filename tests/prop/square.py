import numpy as np
import matplotlib.pyplot as plt

import _mod
import _plot

num_sym = 32
sps = 1
points = 4
fs = 250_000

bits = _mod.get_bits(num_sym, points)
s = _mod.mod_bits(bits, points)
s = _mod.awgn(s,0.05)
s = _mod.phase_noise(s,0.05)

# bits = _mod.get_bits(num_sym, points)
# s = _mod.map_bits(bits, points, sps)

# ps_taps = 101
# ps_beta = .35
# pulse = _mod.rrc(ps_taps, sps, ps_beta)
# s = np.convolve(s, pulse, mode="same")

# s = _mod.delay(s, 0.4)
# s = _mod.offset(s, 13000, 1/fs)

# s = np.convolve(s, pulse, mode="same")

fig, axs = plt.subplots(3,4)
fig.tight_layout()
t = np.arange(0, len(s)) - (len(s)//2)

s2 = s**2
s4 = s2**2
s8 = s4**2

_plot.phasor(s[::sps],  axs[0,0])
_plot.phasor(s2[::sps], axs[0,1])
_plot.phasor(s4[::sps], axs[0,2])
_plot.phasor(s8[::sps], axs[0,3])

_plot.psd(t,s,  axs[1,0])
_plot.psd(t,s2, axs[1,1])
_plot.psd(t,s4, axs[1,2])
_plot.psd(t,s8, axs[1,3])

_plot.timeIQ(t,s,  axs[2,0])
_plot.timeIQ(t,s2, axs[2,1])
_plot.timeIQ(t,s4, axs[2,2])
_plot.timeIQ(t,s8, axs[2,3])

print(f"s0: {np.var(s)}")
print(f"s2: {np.var(s2)}")
print(f"s4: {np.var(s4)}")
print(f"s8: {np.var(s8)}")

axs[0,0].set_title("S")
axs[0,1].set_title("S^2")
axs[0,2].set_title("S^4")
axs[0,3].set_title("S^8")

plt.show()
