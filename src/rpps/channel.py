"""Channel impairments as stages, to test a receiver against: `signal + rp.channel.AWGN(10) + rp.channel.CFO(300)`

Every stage carries its state (noise generator, oscillator phase, fading) in the stream, so the result doesn't
depend on how the signal is cut into blocks. Seeds make a run repeatable.
"""
from types import SimpleNamespace

import numpy as np

from .base import Stage
from .filters.shift import ShiftFreq, ShiftTime
from .filters.filter import FIR
from .sample.down import interpolate_stream

def _cplx(rng, n):
    """n unit-variance complex gaussian samples. Drawn as one stream, so splitting n doesn't change them"""
    return (rng.standard_normal(2*n).view(np.complex128)/np.sqrt(2))

class AWGN(Stage):
    """Additive white gaussian noise at a signal-to-noise ratio

    Give one of:
        snr_db: signal power over noise power in the sample bandwidth
        esn0_db: symbol energy over noise density: the SNR plus 10*log10(sps) (needs sps on the data)
        ebn0_db: bit energy over noise density (needs `bits_per_symbol`)
    power: the signal power the SNR refers to (1.0 for the unit-power constellations). None measures each block
    """
    __slots__ = ("snr", "esn0", "ebn0", "bps", "power", "seed")
    def __init__(self, snr_db=None, esn0_db=None, ebn0_db=None, bits_per_symbol=1, power=1.0, seed=None):
        if sum(v is not None for v in (snr_db, esn0_db, ebn0_db)) != 1:
            raise ValueError("Give exactly one of snr_db, esn0_db, ebn0_db")
        self.snr, self.esn0, self.ebn0, self.bps = snr_db, esn0_db, ebn0_db, bits_per_symbol
        self.power, self.seed = power, seed

    def new_state(self, meta):
        return SimpleNamespace(rng=np.random.default_rng(self.seed))

    def snr_db(self, sps):
        """The SNR [dB] in the sample bandwidth at this sps"""
        if self.snr is not None:
            return self.snr
        esn0 = self.esn0 if self.esn0 is not None else self.ebn0 + 10*np.log10(self.bps)
        return esn0 - 10*np.log10(max(sps, 1.0))

    def step(self, samples, st, meta):
        if not len(samples):
            return samples
        p = np.mean(np.abs(samples)**2) if self.power is None else self.power
        var = p / 10**(self.snr_db(meta.sps)/10)
        noise = np.sqrt(var) * _cplx(st.rng, len(samples))
        return (samples + noise).astype(np.result_type(samples.dtype, np.complex64), copy=False)

class CFO(ShiftFreq):
    """Carrier frequency offset [Hz]: an oscillator whose phase carries across blocks (the centre frequency is not touched)"""
    def step(self, samples, st, meta):
        cf = meta.op.cf
        out = super().step(samples, st, meta)
        meta.op.cf = cf
        return out

class PhaseOffset(Stage):
    """Constant phase rotation [rad]"""
    __slots__ = ("phase",)
    def __init__(self, phase):
        self.phase = phase
    def step(self, samples, st, meta):
        return samples * np.exp(1j*self.phase).astype(np.complex64)

class Gain(Stage):
    """Amplitude gain in dB"""
    __slots__ = ("gain",)
    def __init__(self, db):
        self.gain = 10**(db/20)
    def step(self, samples, st, meta):
        return samples * np.float32(self.gain)

class TimingOffset(ShiftTime):
    """Fractional-sample delay (windowed sinc, carried across blocks)"""

class ClockDrift(Stage):
    """Sample clock error [ppm]: the receiver samples slightly fast (positive) or slow. Rates in Meta are untouched"""
    __slots__ = ("ppm",)
    def __init__(self, ppm):
        self.ppm = ppm
    def new_state(self, meta):
        return SimpleNamespace(tail=None, pos=0.0)
    def step(self, samples, st, meta):
        return interpolate_stream(samples, st, 1 + self.ppm*1e-6)

class Multipath(FIR):
    """Static multipath: a tap at each delay (in samples, whole numbers) with a complex gain"""
    __slots__ = ("taps",)
    def __init__(self, gains, delays):
        n = int(max(delays)) + 1
        taps = np.zeros(n, dtype=np.complex64)
        for g, d in zip(gains, delays):
            taps[int(d)] += g
        self.taps = taps
    def make_taps(self, meta):
        return self.taps

class Fading(Stage):
    """Flat fading by a sum of sinusoids (Jakes): Rayleigh, or Rician with `k_db` (line of sight over scattered power)

    doppler_hz: the maximum Doppler shift. The fading has unit mean power, so it doesn't change the average level
    """
    __slots__ = ("fd", "paths", "k", "seed")
    def __init__(self, doppler_hz, paths=24, k_db=None, seed=None):
        self.fd, self.paths, self.seed = doppler_hz, paths, seed
        self.k = None if k_db is None else 10**(k_db/10)

    def new_state(self, meta):
        rng = np.random.default_rng(self.seed)
        n = self.paths
        return SimpleNamespace(
            n=0,
            wd=2*np.pi*self.fd*np.cos(2*np.pi*rng.random(n))/meta.Fs, # Doppler of each path, rad per sample
            ph=2*np.pi*rng.random(n), los_ph=2*np.pi*rng.random())

    def gain(self, st, count):
        idx = st.n + np.arange(count)
        scat = np.exp(1j*(np.outer(idx, st.wd) + st.ph)).sum(axis=1) / np.sqrt(self.paths)
        st.n += count
        if self.k is None:
            return scat
        los = np.exp(1j*st.los_ph) # A fixed line of sight component
        return (np.sqrt(self.k)*los + scat) / np.sqrt(self.k + 1)

    def step(self, samples, st, meta):
        return (samples * self.gain(st, len(samples))).astype(np.result_type(samples.dtype, np.complex64), copy=False)
