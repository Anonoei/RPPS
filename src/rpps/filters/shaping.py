import numpy as np

from .filter import Filter, Meta

class ShapingFilter(Filter):
    __slots__ = ("pulse")
    def run(self, samples):
        return np.convolve(samples, self.pulse, mode="same")
    def __radd__(self, meta: Meta):
        meta.obj = self.run(meta.obj)
        return meta

class ShapingRRC(ShapingFilter):
    def __init__(self, taps, sps, beta=0.35, gain=1.0):
        """Root-Raised Cosine shaping
        taps: number of taps for FIR
        sps: samples per symbol (sample width for 0 ISI)
        beta: roll off factor [0.35,0>=beta<=1]
        gain: filter gain [1]
        """
        t = np.arange(-taps//2,0)/sps # only generate half the pulse

        sin_arg = np.sin(np.pi*t*(1-beta))
        cos_arg = 4*beta*t*np.cos(np.pi*t*(1+beta))
        denom = np.pi*t*(1-(4*beta*t)**2)
        h = (sin_arg+cos_arg)/denom

        h = np.concat((h, [1+beta*(4/np.pi -1)], h[::-1]))
        self.pulse = gain*h

class ShapingRC(ShapingFilter):
    def __init__(self, taps, sps, beta=0.35, gain=1.0):
        """Raised Cosine shaping
        taps: number of taps for FIR
        sps: samples per symbol (sample width for 0 ISI)
        beta: roll off factor [0.35,0>=beta<=1]
        gain: filter gain [1]
        """
        t = np.arange(-taps//2+1,1)/sps # only generate half the pulse

        numer = np.cos((np.pi*beta*t))
        denom = 1 - (2*beta*t)**2
        h = np.sinc(t) * (numer/denom)

        h = np.concat((h[:-1], h[::-1]))
        self.pulse = gain*h

class ShapingSinc(ShapingFilter):
    def __init__(self, taps, sps, gain=1.0):
        t = np.arange(-taps//2,1)/sps
        h = np.sinc(t)
        h = np.concat((h[:-1], h[::-1]))
        self.pulse = gain*h

class ShapingRect(ShapingFilter):
    def __init__(self, taps, sps):
        self.pulse = np.ones(taps)
