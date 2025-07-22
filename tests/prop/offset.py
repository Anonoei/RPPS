import numpy as np
import matplotlib.pyplot as plt

import _mod
import _plot

num_sym = 32
sps = 16
points = 4
fs = 250_000

bits = _mod.get_bits(num_sym, points)
s = _mod.map_bits(bits, points, sps)

ps_taps = 101
ps_beta = 0
pulse = _mod.rrc(ps_taps, sps, ps_beta)
s = np.convolve(s, pulse, mode="same")

sd = _mod.delay(s, 1.2)
sdf = _mod.offset(sd, 13000, 1/fs)

fig, axs = plt.subplots(2,2)
# t = np.arange(0, len(s)) - (len(s)//2)
t = np.arange(0, len(s))

_plot.phasor(s[::sps], axs[0,0])
_plot.timeIQ(t,s, axs[0,1])

_plot.phasor(sdf[::sps], axs[1,0])
_plot.timeIQ(t,sdf, axs[1,1])

# _plot.psd(t,s, axs[2])
# for i in range(num_sym):
#     x_idx = i*sps+ps_taps//2
#     y_val = s[x_idx]
#     axs[1].plot([x_idx,x_idx], [0, y_val.real], c="blue", alpha=0.5)
#     axs[1].plot([x_idx,x_idx], [0, y_val.imag], c="red", alpha=0.5)
plt.show()
