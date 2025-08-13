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
        # # Correct (slow) implementation
        # t = np.arange(-taps//2,taps//2)
        # i0 = t==0
        # i1 = (t==(sps/(4*beta))) | (t==-(sps/(4*beta)))
        # i2 = ~(i0 | i1)

        # h = np.zeros(taps)
        # h[i0] = 1/sps*(1+beta*(4/np.pi - 1))
        # h[i1] = beta/(sps*np.sqrt(2)) * (
        #             (1+2/np.pi) * np.sin(np.pi/(4*beta)) + \
        #             (1-2/np.pi) * np.cos(np.pi/(4*beta))
        #         )
        # h[i2] = 1/sps*((
        #             np.sin(np.pi*t[i2]/sps * (1-beta)) \
        #             + 4*beta*t[i2]/sps * \
        #             np.cos(np.pi*t[i2]/sps * (1+beta))) / \
        #             (np.pi*t[i2]/sps* (1 - (4*beta*t[i2]/sps)**2))
        #         )


        t = np.arange(-taps//2,0)/sps # only generate half the pulse
        sin_arg = np.sin(np.pi*t*(1-beta))
        cos_arg = 4*beta*t*np.cos(np.pi*t*(1+beta))
        denom = np.pi*t*(1-(4*beta*t)**2)
        h = 1/sps*(sin_arg+cos_arg)/denom

        h = np.concat((h, [1/sps*(1+beta*(4/np.pi -1))], h[::-1]))
        h = h * np.hanning(len(h))
        hc = np.zeros(len(h), dtype=np.complex64)
        hc = h+1j*h
        self.pulse = gain*hc

class ShapingRC(ShapingFilter):
    def __init__(self, taps, sps, beta=0.35, gain=1.0):
        """Raised Cosine shaping
        taps: number of taps for FIR
        sps: samples per symbol (sample width for 0 ISI)
        beta: roll off factor [0.35,0>=beta<=1]
        gain: filter gain [1]
        """
        t = np.arange(-taps//2+1,1) # only generate half the pulse

        numer = np.cos((np.pi*beta*t)/sps)
        denom = 1 - (2*beta*t/sps)**2
        h = np.sinc(t/sps) * numer/denom

        h = np.concat((h[:-1], h[::-1]))
        h = h * np.hanning(len(h))
        hc = np.zeros(len(h), dtype=np.complex64)
        hc = h+1j*h

        self.pulse = gain*hc

class ShapingSinc(ShapingFilter):
    def __init__(self, taps, sps, gain=1.0):
        t = np.arange(-taps//2,1)/sps
        h = np.sinc(t)
        h = np.concat((h[:-1], h[::-1]))
        h = h * np.hanning(len(h))
        hc = np.zeros(len(h), dtype=np.complex64)
        hc = h+1j*h
        self.pulse = gain*hc

class ShapingRect(ShapingFilter):
    def __init__(self, taps, sps, gain=1.0):
        h = np.ones(taps)
        hc = np.zeros(len(h), dtype=np.complex64)
        hc = h+1j*h
        self.pulse = gain*hc


class ShapingMatRRC(ShapingFilter):
    __slots__ = ("N", "sps")
    def __init__(self, N, sps, poff=0, beta=0.35):
        self.N = N
        self.sps = sps
        tmat = self.calc_tmat(N, sps)
        self.pulse = self.pulse_rrc(tmat, poff, beta)

    def calc_tmat(self, N, sps):
        """
        N: the sample count at 1 SPS
        sps: desired SPS
        """
        num_out = N*sps
        vmat = np.ones((num_out, num_out)) * np.arange(num_out)
        tmat = vmat / sps - vmat.T
        tmat = tmat[0:num_out//sps, :]
        return tmat

    def pulse_rrc(self, tmat, poff=0, beta=0.35):
        """
        tmat: time matrix to generated pulses
        poff: pulse offset
        beta: beta rolloff

        returns (N, N*sps) RRC matrix
        """
        t = tmat - poff - 1e-9
        sin_arg = np.sin(np.pi * t * (1.0 - beta))
        cos_arg = 4.0 * beta * t * np.cos(np.pi * t * (1 + beta))
        denom = np.pi * t * (1.0 - 16.0 * (beta * t) * (beta * t))
        return (sin_arg + cos_arg) / denom

    def run(self, samples):
        return np.matmul(samples, self.pulse)

    def burst(self, samples):
        out = np.zeros(len(samples)*self.sps, dtype=samples.dtype)
        for i in range(0, len(samples)//self.N, 1):
            s_idx = i*self.N
            if s_idx+self.N <= len(samples):
                segment = samples[s_idx:s_idx+self.N]
                out[i*(self.N*self.sps):(i+1)*(self.N*self.sps)] = self.run(segment)
            else:
                segment = np.zeros(self.N)
                num_pulse = len(samples)-s_idx
                segment[:num_pulse] = samples[s_idx:]
                print(f"Pulsing {len(segment)}")
                pulsed = self.run(segment)
                out[i*(self.N*self.sps):] = pulsed[:num_pulse*self.sps]
        return out
