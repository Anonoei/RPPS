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

class _AGC:
    __slots__ = (
        "target",
        "gain", "err",
        "max", "min"
    )

    def __init__(self, target, min_gain=-np.inf, max_gain=np.inf):
        self.target = target
        self.gain = 0.0
        self.err = 0.0
        self.max = max_gain
        self.min = min_gain

    def _run(self, sample):
        ...

    def run(self, sample):
        self._run(sample)
        self.norm()
        return self.gain*sample

    def burst(self, samples, n=30):
        out = np.zeros(len(samples))
        avg = np.convolve(samples, np.ones(n), mode="same")/n
        for i, sample in enumerate(avg):
            out[i] = self.run(sample)
        return out

    def norm(self):
        self.gain = self.gain if self.gain < self.max else self.max
        self.gain = self.gain if self.gain > self.min else self.min
        # self.gain = np.clip(gain, self.min, self.max)
        # self.gain = np.min(self.gain, self.max)
        # self.gain = np.max(self.gain, self.min)

class AGC_lin(_AGC):
    __slots__ = ("mu")
    def __init__(self, target, mu):
        super().__init__(target)
        self.mu = mu

    def _run(self, sample):
        pwr = np.abs(sample)
        self.err = self.target - self.gain*pwr
        self.gain += self.mu*self.err

class AGC_log(_AGC):
    __slots__ = ("mu")
    def __init__(self, target, mu):
        super().__init__(target)
        self.mu = mu
        self.gain = 1.0

    def _run(self, sample):
        pwr = np.abs(sample)
        self.err = np.log(self.target) - np.log(abs(self.gain*pwr))
        self.gain = np.exp(np.log(self.gain) + self.mu*self.err)

class AGC2(_AGC):
    __slots__ = ("agc")
    def __init__(self, target, mu):
        super().__init__(target)
        # self.mu = mu
        self.agc = AGC_log(target, mu)

    def _run(self, sample):
        self.agc._run(sample)
        self.agc._run(sample)
        self.gain = self.agc.gain
        self.err = self.agc.err

if __name__ == "__main__":
    agc = AGC2(1, 0.5)
    t = np.arange(0,1000,1)
    y = 1*np.exp(2j*np.pi*(1/20)*t)
    y[251:500] *= 0.1
    y[500:750] *= 1.5
    y[750:] *= 0.4

    out = np.zeros(len(y), dtype=y.dtype)
    gain = np.zeros(len(y))
    err = np.zeros(len(y))
    for i, amp in enumerate(y):
        out[i] = agc.run(amp)
        gain[i] = agc.gain
        err[i] = agc.err
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3,1)
    axs[0].plot(t, y.real) # type: ignore
    axs[0].plot(t, out.real) # type: ignore
    axs[1].plot(t, gain)
    axs[2].plot(t, err)
    plt.show()
