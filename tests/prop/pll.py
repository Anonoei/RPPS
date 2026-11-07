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

s = _mod.delay(s, 1.2)
s = _mod.offset(s, 13000, 1/fs)

class PLL_PI:
    """
    proportional, e_F,1[n] = Kp * e_D[n]
    integrator,   e_F,2[n] = e_F,2[n-1] + Ki * e_D[n]
    loop filter: e_F[n] = e_F,1[n] + e_F,2[n]
    accumulator: E[n] = E[n-1] + K0 * e_F[n-1] % 2pi

    Df: dampening factor
      = 1/sqrt(2) = 0.707 (0.5-2.0)
      @< 1, overshoot/undershoots (underdamped)
      @> 1, oscillation disappears (overdamped)
      @= 1, critically damped
    Bn: equivalent noise bandwidth
      = Fs * (0.01-0.05)
    K0: loop gain
      = 1
    Kd: error gain
    Kp: proportional
        = ( 1/(Kd*K0) ) * ( ( 4*Df )/( Df+( 1/(4*Df) ) ) ) * (Bn/Fs)
    Ki: integrator
        = ( 1/(Kd*K0) ) * ( 4/( Df+4( 1/(4*Df) ) )**2 ) * (Bn/Fs)**2
    """
    #__slots__ = ("Kp", "Ki")

    def __init__(self):
        k = 1
        N = 15
        Kp = 0.2667
        Ki = 0.0178
        K0 = 1

        input_s = np.zeros(100)

        integrator_out = 0
        ph_est = np.zeros(100)
        e_D = [] # Error detector output
        e_F = [] # loop filter output
        sin_out = np.zeros(100)
        cos_out = np.zeros(100)

        for n in range(99):
            input_s[n] = np.cos(2*np.pi*(k/N)*n + np.pi)

            # phase detectpr
            e_D.append(input_s[n] * sin_out[n])

            # loop filter
            integrator_out += Ki * e_D[n]
            e_F.append(Kp * e_D[n] + integrator_out)

            ph_est[n+1] = ph_est[n] + K0 * e_F[n]


fig, axs = plt.subplots(2,2)
# t = np.arange(0, len(s)) - (len(s)//2)
t = np.arange(0, len(s))

_plot.phasor(s[::sps], axs[0,0])
_plot.timeIQ(t,s, axs[0,1])

plt.show()
