import numpy as np
import matplotlib.pyplot as plt

import _mod
import _plot

num_sym = 1024*16
sps = 8
points = 4
fs = 250_000

bits = _mod.get_bits(num_sym, points)
s = _mod.map_bits(bits, points, sps)
print(s.dtype)

ps_taps = 101
ps_beta = 0.35
pulse = _mod.rrc(ps_taps, sps, ps_beta)
s = np.convolve(s, pulse, mode="same")
print(s.dtype)
s = _mod.delay(s, 0.4)
s = _mod.offset(s, 10, 1/fs)
print(s.dtype)

s = s[::sps]

fig, axs = plt.subplots(1)
t = np.arange(0, len(s))
# _plot.timeIQ(t[:32], s[:32], axs)
# _plot.phasor(s[::sps], axs)
# plt.show()
print(f"Saving {s[:8]}")
v = np.zeros(len(s)*2, dtype=np.float32)
v[::2] = np.real(s).astype(np.float32)
v[1::2] = np.imag(s).astype(np.float32)
print(f"s: {s.shape} {s.dtype} / v: {v.shape} {v.dtype}")
v.tofile("qpsk_fs250000.cf64")
