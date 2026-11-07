import numpy as np
import matplotlib.pyplot as plt

import _mod
import _plot

num_sym = 1024*16
sps = 16
points = 4
fs = 250_000

s = np.fromfile("data/qpsk_fs250000.cf64", dtype=np.float32).view(np.complex64)

s = s[:32]
fig, axs = plt.subplots(1)
t = np.arange(0, len(s))

_plot.timeIQ(t, s, axs)
plt.show()
