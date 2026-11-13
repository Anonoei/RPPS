import numpy as np

from .filter import FIR

def rrc_pulse(t, beta=0.35):
    """Root-raised-cosine pulse, t in symbols. Handles the t=0 and |t|=1/(4*beta) limits"""
    t = np.asarray(t, dtype=np.float64)
    if beta == 0:
        return np.sinc(t)
    out = np.empty_like(t)
    at0 = np.abs(t) < 1e-9
    atS = np.abs(np.abs(t) - 1/(4*beta)) < 1e-9
    rest = ~(at0 | atS)
    out[at0] = 1.0 + beta*(4/np.pi - 1)
    out[atS] = beta/np.sqrt(2) * (
        (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) + (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
    )
    tr = t[rest]
    out[rest] = (np.sin(np.pi*tr*(1 - beta)) + 4*beta*tr*np.cos(np.pi*tr*(1 + beta))) \
        / (np.pi*tr*(1 - (4*beta*tr)**2))
    return out

class ShapingFilter(FIR):
    __slots__ = ("pulse")
    def make_taps(self, meta):
        return self.pulse
    def run(self, samples):
        """One-shot (stateless) filtering"""
        return np.convolve(samples, self.pulse, mode="same")

def _centred(taps, sps):
    """Time axis, in symbols, of `taps` samples centred on the middle of the pulse"""
    return (np.arange(taps) - (taps - 1)/2)/sps

def _default_taps(sps, span=25):
    taps = int(round(sps*span))
    return taps | 1 # Odd, so there is a centre tap

def rc_pulse(t, beta=0.35):
    """Raised-cosine pulse, t in symbols. Handles the |t| = 1/(2*beta) limit"""
    t = np.asarray(t, dtype=np.float64)
    if beta == 0:
        return np.sinc(t)
    den = 1 - (2*beta*t)**2
    sing = np.abs(den) < 1e-9
    safe = np.where(sing, 1.0, den)
    out = np.sinc(t)*np.cos(np.pi*beta*t)/safe
    out[sing] = np.pi/4*np.sinc(1/(2*beta))
    return out

class ShapingRRC(ShapingFilter):
    def __init__(self, taps, sps, beta=0.35, gain=1.0):
        """Root-Raised Cosine shaping
        taps: number of taps for FIR (None: 25 symbols)
        sps: samples per symbol (sample width for 0 ISI)
        beta: roll off factor [0.35, 0 <= beta <= 1]
        gain: filter gain [1]
        """
        taps = _default_taps(sps) if taps is None else int(taps)
        self.pulse = gain/sps*rrc_pulse(_centred(taps, sps), beta)

class ShapingRC(ShapingFilter):
    def __init__(self, taps, sps, beta=0.35, gain=1.0):
        """Raised Cosine shaping
        taps: number of taps for FIR (None: 25 symbols)
        sps: samples per symbol (sample width for 0 ISI)
        beta: roll off factor [0.35, 0 <= beta <= 1]
        gain: filter gain [1]
        """
        taps = _default_taps(sps) if taps is None else int(taps)
        self.pulse = gain*rc_pulse(_centred(taps, sps), beta)

class ShapingSinc(ShapingFilter):
    def __init__(self, taps, sps, gain=1.0):
        t = np.arange(-taps//2,taps//2)/sps
        h = np.sinc(t)
        # h = np.concat((h[:-1], h[::-1]))
        # h = h * np.hanning(len(h))
        # hc = np.zeros(len(h), dtype=np.complex64)
        # hc = h+1j*h
        # hc = h+1j*h
        self.pulse = gain*h

class ShapingRect(ShapingFilter):
    def __init__(self, taps, sps, gain=1.0):
        h = np.ones(taps)
        # hc = np.zeros(len(h), dtype=np.complex64)
        # hc = h+1j*h
        self.pulse = gain*h


class ShapingHalfSine(ShapingFilter):
    def __init__(self, sps, gain=1.0):
        """Half-sine pulse lasting one symbol: sin(pi*n/sps) for n = 0..sps, so the ends are zero and, for an even sps, the centre
        is a whole sample (sps + 1 taps, peak `gain`). On both rails of O-QPSK (staggered by half a symbol) it is MSK
        sps: samples per symbol"""
        sps = int(round(sps))
        if sps < 2:
            raise ValueError(f"A half-sine pulse needs at least 2 samples per symbol, got {sps}")
        self.pulse = gain*np.sin(np.pi*np.arange(sps + 1)/sps)

class ShapingGaussian(ShapingFilter):
    def __init__(self, sps, bt=0.5, span=4, gain=1.0):
        """Gaussian-filtered rectangle: the frequency pulse of GMSK and GFSK, from a held level of one symbol
        sps: samples per symbol. bt: 3 dB bandwidth times the symbol period (BLE 0.5, GSM 0.3). span: symbols the Gaussian covers
        The taps sum to `sps`, so a level held for a symbol comes out as that level: zero-stuff each symbol to `sps` samples,
        then filter, and the shaped signal integrates (rp.wave.FM) to the phase the modulation index asks for"""
        sps = int(round(sps))
        sigma = np.sqrt(np.log(2))/(2*np.pi*bt)*sps # Samples: the Gaussian whose 3 dB bandwidth is bt/T
        n = int(round(span*sps)) | 1
        g = np.exp(-0.5*((np.arange(n) - (n - 1)/2)/sigma)**2)
        g /= g.sum()
        self.pulse = gain*np.convolve(np.ones(sps), g)

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
        return rrc_pulse(tmat - poff, beta)

    def run(self, samples):
        return np.matmul(samples, self.pulse)

    def step(self, samples, st, meta):
        return self.run(samples)

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
                pulsed = self.run(segment)
                out[i*(self.N*self.sps):] = pulsed[:num_pulse*self.sps]
        return out


class ShapingBiphase(ShapingFilter):
    """Matched filter for biphase (Manchester) symbols: first half +1, second half -1

    One symbol is meta.isps samples, so this takes sps from Meta when the stream starts.
    """
    def __init__(self, gain=1.0):
        self.gain = gain

    def make_taps(self, meta):
        sps = meta.isps
        h = np.ones(sps)
        h[sps//2:] = -1.0
        return self.gain * h / sps
