import numpy as np
import matplotlib.pyplot as plt

import _mod
import _plot

num_sym = 2048
sps = 4
points = 4
Fs = 1_000_000

def _sigmoid(taps):
    t = np.arange(-taps/2, taps/2)/(taps/16)
    f = 1/(1+np.exp(t))
    return f

def sigmoid_map(samples, taps, clip=0.0):
    amp_max = np.max(np.abs(samples))
    amp_max = np.max([np.max(samples.real), np.max(samples.imag), -np.min(samples.real), -np.min(samples.imag)])
    amp_max = amp_max * (1-clip)

    sigmoid = _sigmoid(taps)
    plt.plot(sigmoid)
    plt.show()

    y_ratio = taps/amp_max
    print(f"max: {amp_max}, y_ratio: {y_ratio}")
    # exit()
    y_idx = np.floor(np.abs(samples)*y_ratio).astype(int)
    y_idx = np.clip(y_idx, 0, taps-1)

    out = np.zeros(len(samples), dtype=samples.dtype)
    for i, sample in enumerate(samples):
        print(f"{i}, {sample}: {y_idx[i]} *= {sigmoid[y_idx[i]]}")
        out[i] = sample*sigmoid[y_idx[i]]
    return out

bits = _mod.get_bits(num_sym, points)
s = _mod.map_bits(bits, points, sps)

ps_taps = 101
ps_beta = .35
pulse = _mod.rrc(ps_taps, sps, ps_beta)
s = s / np.max(s)
s = np.convolve(s, pulse, mode="same")
s *= 1.4

sc = np.clip(np.copy(s.real), -1,1)+1j*np.clip(np.copy(s.imag), -1,1)

s_max = np.max([np.max(s.real), np.max(s.imag)])
sc_max = np.max([np.max(sc.real), np.max(sc.imag)])
print(f"s_max: {s_max}")
print(f"sc_max: {sc_max}")
# exit()

# s = _mod.awgn(s,0.1)
# sc = _mod.awgn(sc,0.1)

t = np.arange(0, len(s))
x = t - (len(s)//2)

fig, axs = plt.subplots(3)
fig.tight_layout()

_plot.timeIQ(t[:512//sps], s[:512//sps], axs[0])
_plot.timeIQ(t[:512//sps], sc[:512//sps],axs[1])

_plot.psd(x,s,  axs[2], Fs, Fs//50)
_plot.psd(x,sc, axs[2], Fs, Fs//50)

# axs[0,0].set_title("S")
axs[0].set_title(f"Non-Linear {s_max:.2f} -> 1")

plt.show()
