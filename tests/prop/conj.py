import numpy as np
import matplotlib.pyplot as plt

import _mod
import _plot

num_sym = 8
sps = 18
points = 4
fs = 250_000

# bits = _mod.get_bits(num_sym, points)
# s = _mod.mod_bits(bits, points)
# s = _mod.awgn(s)
# s = _mod.phase_noise(s)

bits = _mod.get_bits(num_sym, points)
s = _mod.map_bits(bits, points, sps)

ps_taps = 101
ps_beta = .35
pulse = _mod.rrc(ps_taps, sps, ps_beta)
s = np.convolve(s, pulse, mode="same")
s = _mod.awgn(s, 0.2)

ang = np.angle((s**4))
ang = ang[::sps]
s = s[::sps]

ang_err = (1/4)*ang

print(ang)
print(ang_err)

# plt.plot(s.real+s.imag)
# plt.plot(np.angle(s))
# plt.plot(ang_err)

e_min = np.min(ang_err)
e_max = np.max(ang_err)

print(f"min: {e_min:.3f}, max: {e_max:.3f}, var: {np.var(ang_err):.3f}")

dif = e_max + np.abs(e_min)
print(f"dif: {dif:.3f}")
print(f"pi: {dif/np.pi:.3f}")
print(f"1/pi: {1/np.pi:.3f}")


# plt.plot(ang_err)
# plt.plot(s.real+s.imag)
# plt.show()
exit()

# s = _mod.moving_average(s, 2)
# sps = 2

# s = _mod.delay(s, 0.4)
# s = _mod.offset(s, 13000, 1/fs)

fig, axs = plt.subplots(2,2)
t = np.arange(0, len(s)) - (len(s)//2)

spc = s**2 * np.conj(s)

_plot.phasor(s[::sps],  axs[0,0])
_plot.phasor(spc[::sps], axs[0,1])

_plot.timeIQ(t,s,  axs[1,0])
_plot.timeIQ(t,spc, axs[1,1])

plt.show()
