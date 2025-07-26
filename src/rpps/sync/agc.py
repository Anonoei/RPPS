"""Automatic Gain Control

R = reference level

x[n] = input signal
a[n] = AGC output
z[n] = controlled output
|z[n]| = sqrt(zI**2 + zQ**2) = |a[n-1] * x[n]|
e[n] = R - |a[n-1]*x[n]|
a[n] = a[n-1] + μ * e[n]
    a[n] = (1-μB) * a[n-1] + μ * R
        a[n] = { a[0] - R/B }(1 - μB)**n + R/B
        z[n] = R/B * B = R
"""

import numpy as np

class AGC_lin:
    def __init__(self, R, mu):
        self.R = R
        self.mu = mu
        self.ref = 0.0
        self.err = 0.0

    def run(self, sample):
        e = self.R - self.ref*np.abs(sample)
        a = self.ref + self.mu*e
        self.err = e
        self.ref = a
        return self.ref*sample

class AGC_log:
    def __init__(self, R, mu):
        self.R = R
        self.mu = mu
        self.ref = 1.0
        self.err = 0.0

    def run(self, sample):
        e = np.log(self.R) - np.log(abs(self.ref*np.abs(sample)))
        a = np.exp(np.log(self.ref) + self.mu*e)
        self.err = e
        self.ref = a
        return self.ref*sample

class AGC_FF:
    def __init__(self, R):
        self.R = R

    def run(self, samples, n=30):
        B = np.abs(samples)
        avg = np.convolve(B, np.ones(n), mode="same")/n
        a = self.R/avg
        return a*samples

if __name__ == "__main__":
    agc = AGC_log(1, 0.1)
    t = np.arange(0,1000,1)
    y = np.exp(2j*np.pi*(1/20)*t)
    y[251:500] *= 0.1
    y[500:750] *= 1.5
    y[750:] *= 0.4

    out = np.zeros(len(y), dtype=y.dtype)
    gain = np.zeros(len(y))
    err = np.zeros(len(y))
    for i, amp in enumerate(y):
        out[i] = agc.run(amp)
        gain[i] = agc.ref
        err[i] = agc.err
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3,1)
    axs[0].plot(t, y.real)
    axs[0].plot(t, out.real)
    axs[1].plot(t, gain)
    axs[2].plot(t, err)
    plt.show()
