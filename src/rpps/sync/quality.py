"""Signal quality of recovered symbols: Es/N0, Eb/N0, EVM, measured once (`measure`), over a running window (`Track`)
or on the stream (`Monitor`, a stage that publishes it); each symbol's error (`errors`) and an analyser's error summary
(`summary`: EVM, MER, magnitude and phase error, IQ offset, gain imbalance, quadrature error)

The symbols must be locked (timing, frequency, phase and gain corrected) so they sit on the constellation points.
The noise is found the way detection finds it: the variance of Gaussian clusters around equiprobable points, by EM
(rp.math.prob.mixture_variance), which stays right when the clusters touch and the nearest point is sometimes wrong.
"""
from types import SimpleNamespace

import numpy as np

from ..math.prob import mixture_variance
from ..base import Stage

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

def decide(y, points):
    """The nearest of `points` to each of y"""
    pts = np.asarray(points)
    y = np.asarray(y)
    return pts[np.argmin(np.abs(y[:, None] - pts[None, :]), axis=1)]

class Errors:
    """Per symbol, against the nearest point (see `errors`): vector = y - ideal; evm_pct = |vector|, mag_pct = |y| - |ideal|
    (both as a percentage of the constellation's rms, as a vector signal analyser normalises them); phase_deg = the
    angle from ideal to y [degrees] (0 where the ideal is the origin)"""
    __slots__ = ("ideal", "vector", "evm_pct", "mag_pct", "phase_deg")
    def __init__(self, ideal, vector, evm_pct, mag_pct, phase_deg):
        self.ideal, self.vector, self.evm_pct, self.mag_pct, self.phase_deg = ideal, vector, evm_pct, mag_pct, phase_deg

    def __len__(self):
        return len(self.vector)

def errors(y, points):
    """The error of each symbol of y (locked: corrected and at the constellation's scale) against the nearest of `points`.
    Returns Errors (arrays of len(y))"""
    y = np.asarray(y, dtype=np.complex128)
    pts = np.asarray(points, dtype=np.complex128)
    rms = float(np.sqrt(np.mean(np.abs(pts)**2))) or 1.0
    d = decide(y, pts)
    e = y - d
    ph = np.where(np.abs(d) > 0, np.degrees(np.angle(y*np.conj(d))), 0.0)
    return Errors(d, e, 100*np.abs(e)/rms, 100*(np.abs(y) - np.abs(d))/rms, ph)

class Summary:
    """A vector signal analyser's error summary of locked symbols (see `summary`)

    evm_rms_pct, evm_peak_pct (at symbol evm_peak_at); mer_db: ideal symbol power over error power; mag_err_rms_pct,
    phase_err_rms_deg; from a fit of the symbols as kI I + kQ Q + c on the ideal points' I and Q:
    iq_offset_db: |c| over the rms of the rest [dB] (floored at -200); gain_imbalance_db: |kI| / |kQ| [dB], I over Q;
    quad_error_deg: the angle between the I and Q axes less 90 [degrees] (positive: more than 90 apart). The last two are
    None for a constellation on one line (BPSK, ASK). n: symbols"""
    __slots__ = ("evm_rms_pct", "evm_peak_pct", "evm_peak_at", "mer_db", "mag_err_rms_pct", "phase_err_rms_deg",
                 "iq_offset_db", "gain_imbalance_db", "quad_error_deg", "n")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self):
        g = "" if self.gain_imbalance_db is None else (f", gain imbalance {self.gain_imbalance_db:+.2f} dB, quadrature "
                                                      f"error {self.quad_error_deg:+.2f} deg")
        return (f"EVM {self.evm_rms_pct:.2f}% rms, {self.evm_peak_pct:.2f}% peak (symbol {self.evm_peak_at}), MER "
                f"{self.mer_db:.1f} dB, IQ offset {self.iq_offset_db:.1f} dB{g}, {self.n} symbols")

def summary(y, points):
    """EVM, MER, magnitude and phase error, IQ offset, gain imbalance and quadrature error of locked symbols y against
    the nearest of `points` (Summary). Like an analyser, the decisions are the reference: at low SNR, where some are
    wrong, the errors read low"""
    y = np.asarray(y, dtype=np.complex128)
    if len(y) < 8:
        raise ValueError("Need at least 8 symbols")
    E = errors(y, points)
    d, e = E.ideal, E.vector
    k = int(np.argmax(E.evm_pct))
    with np.errstate(divide="ignore"):
        mer = 10*np.log10(np.sum(np.abs(d)**2)/np.sum(np.abs(e)**2)) if np.any(e) else np.inf
    I, Q = d.real, d.imag
    planar = np.std(I) > 1e-9 and np.std(Q) > 1e-9 and abs(np.corrcoef(I, Q)[0, 1]) < 0.99
    A = np.stack([I, Q, np.ones(len(y))], axis=1) if planar else np.stack([d, np.ones(len(y))], axis=1)
    coef = np.linalg.lstsq(A.astype(np.complex128), y, rcond=None)[0]
    c = coef[-1]
    rest = A[:, :-1] @ coef[:-1]
    off = 20*np.log10(max(abs(c)/max(float(np.sqrt(np.mean(np.abs(rest)**2))), 1e-300), 1e-10))
    gi = qe = None
    if planar:
        kI, kQ = coef[0], coef[1]
        gi = float(20*np.log10(abs(kI)/abs(kQ)))
        qe = float((np.degrees(np.angle(kQ/kI)) - 90 + 180) % 360 - 180)
    return Summary(evm_rms_pct=float(np.sqrt(np.mean(E.evm_pct**2))), evm_peak_pct=float(E.evm_pct[k]), evm_peak_at=k,
                   mer_db=float(mer), mag_err_rms_pct=float(np.sqrt(np.mean(E.mag_pct**2))),
                   phase_err_rms_deg=float(np.sqrt(np.mean(E.phase_deg**2))), iq_offset_db=float(off),
                   gain_imbalance_db=gi, quad_error_deg=qe, n=len(y))

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

class Monitor(Stage):
    """Es/N0, Eb/N0 and EVM of the locked symbols going by, published to `meta.stream.est[name]` as a `Quality` (None until
    `minimum` symbols): the symbols pass unchanged. The constellation is `points`, or the detected one (`meta.det`,
    from `Acquire`) when none is given. A profile names it `{"sync": "quality"}`."""
    __slots__ = ("points", "bits", "rate", "history", "minimum", "name")
    def __init__(self, points=None, bits=None, rate=1.0, history=4096, minimum=256, name="quality"):
        self.points = None if points is None else np.asarray(points, dtype=np.complex128)
        self.bits, self.rate, self.history, self.minimum, self.name = bits, rate, int(history), int(minimum), name

    def __str__(self):
        return "Quality"

    def new_state(self, meta):
        return SimpleNamespace(track=None, mod=None)

    def step(self, y, st, meta):
        pts = self.points
        if pts is None:
            det = meta.det
            if det is None:
                return y
            if st.mod != det.mod:
                from .acquire import candidates
                st.track, st.mod = None, det.mod
                pts = candidates([det.mod])[det.mod]
            elif st.track is not None:
                pts = st.track.points
            else:
                pts = candidates([det.mod])[det.mod]
        if st.track is None:
            st.track = Track(pts, self.bits, self.rate, self.history, self.minimum)
        if len(y):
            meta.stream.est[self.name] = st.track.update(y)
        return y
