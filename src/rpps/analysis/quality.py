"""Signal quality of recovered symbols: Es/N0, Eb/N0, EVM

The symbols must be locked (timing, frequency, phase and gain corrected) so they sit on the constellation points.
The noise is found the way detection finds it: the variance of Gaussian clusters around equiprobable points, by EM
(rp.math.prob.mixture_variance), which stays right when the clusters touch and the nearest point is sometimes wrong.
"""
import numpy as np

from ..math.prob import mixture_variance

class Quality:
    """Es/N0, Eb/N0 and friends of a set of symbols (see `measure`)

    esn0_db: symbol energy over noise density. ebn0_db: bit energy over noise density (uncoded, or the information
    bits of a code with `rate`). snr_db: signal over noise power in the sample bandwidth (None without `sps`).
    evm_pct: rms error vector against the nearest point, as a percentage of the rms symbol. sigma: noise per real axis
    (unit-rms symbols). n: symbols used.
    """
    __slots__ = ("esn0_db", "ebn0_db", "snr_db", "evm_pct", "sigma", "bits", "rate", "n")
    def __init__(self, esn0_db, ebn0_db, snr_db, evm_pct, sigma, bits, rate, n):
        self.esn0_db, self.ebn0_db, self.snr_db = esn0_db, ebn0_db, snr_db
        self.evm_pct, self.sigma, self.bits, self.rate, self.n = evm_pct, sigma, bits, rate, n

    def __str__(self):
        return f"Es/N0 {self.esn0_db:.1f} dB, Eb/N0 {self.ebn0_db:.1f} dB, EVM {self.evm_pct:.1f}%"

    __repr__ = __str__

def measure(y, points, bits=None, rate=1.0, sps=None):
    """Es/N0, Eb/N0 and EVM of locked symbols `y` around the constellation `points`

    bits: bits per symbol (default log2 of the number of points). rate: code rate, if the bits are information bits
    of a code (Eb/N0 is per information bit). sps: samples per symbol of the data they were taken from, for snr_db.
    """
    y = np.asarray(y)
    pts = np.asarray(points, dtype=np.complex128)
    pts = pts/np.sqrt(np.mean(np.abs(pts)**2))
    if len(y) < 8:
        raise ValueError("Need at least 8 symbols")
    y = y/np.sqrt(np.mean(np.abs(y)**2))
    for _ in range(3): # The rms includes the noise, so the signal is a little smaller than a point: fit its amplitude
        dec = pts[np.argmin(np.abs(y[:, None] - pts[None, :]), axis=1)]
        y = y/(np.sum((y*np.conj(dec)).real)/np.sum(np.abs(dec)**2))
    d2 = np.abs(y[:, None] - pts[None, :])**2
    var = mixture_variance(d2) # Per real axis
    esn0 = 10*np.log10(1/(2*var))
    bits = float(np.log2(len(pts))) if bits is None else bits
    ebn0 = esn0 - 10*np.log10(bits*rate)
    snr = None if sps is None else esn0 - 10*np.log10(max(sps, 1.0))
    evm = 100*np.sqrt(np.mean(np.min(d2, axis=1)))
    return Quality(float(esn0), float(ebn0), None if snr is None else float(snr), float(evm), float(np.sqrt(var)), bits, rate, len(y))

class Track:
    """Running Es/N0 estimate over the last `history` symbols: `update(block)` returns the current Quality (None until enough)"""
    def __init__(self, points, bits=None, rate=1.0, history=4096, minimum=256):
        self.points, self.bits, self.rate = points, bits, rate
        self.history, self.minimum = history, minimum
        self.buf = np.zeros(0, dtype=np.complex64)

    def update(self, y, sps=None):
        self.buf = np.concatenate((self.buf, np.asarray(y, dtype=np.complex64)))[-self.history:]
        if len(self.buf) < self.minimum:
            return None
        return measure(self.buf, self.points, self.bits, self.rate, sps)
