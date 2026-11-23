"""Signal quality of recovered symbols: Es/N0, Eb/N0, EVM, measured once (`measure`), over a running window (`Track`)
or on the stream (`Monitor`, a stage that publishes it); each symbol's error (`errors`) and an analyser's error summary
(`summary`: EVM, MER, magnitude and phase error, IQ offset, gain imbalance, quadrature error)

The symbols must be locked (timing, frequency, phase and gain corrected) so they sit on the constellation points.
The noise is found the way detection finds it: the variance of Gaussian clusters around equiprobable points, by EM
(rp.math.prob.mixture_variance), which stays right when the clusters touch and the nearest point is sometimes wrong.

Before anything is locked, three estimates need no constellation and no noise floor measured apart (each an `SNR`),
and a fourth needs only a constellation to assume (`m2m4(y, points)`: the symbols' second and fourth moments under that
constellation's kurtosis, no decisions; rpps.sync.detect scores each candidate at its own):

- `from_sequence(y, ref)`: from a known sequence in the signal. With a CAZAC one (rp.sequence.zadoff_chu: LTE/NR PSS,
  PRACH, SRS), the circular correlation of one received period with it has one peak (the channel, multipath taps
  included within `guard`) and noise alone at every other lag: signal and noise from the same correlation, in band.
  Any other known sequence, aligned: the least-squares gain and its residual.
- `cyclic(x, sps)`: from the signal's own cyclostationarity, in band. Shifted by -1/2T and +1/2T, the two copies of a
  linearly modulated signal correlate (its spectral correlation at alpha = 1/T); noise doesn't. Where both copies carry
  signal (the roll-off's overlap) they are fully coherent whatever the pulse: per frequency (Sa - N0)(Sb - N0) = |C|^2,
  and one N0 is fitted to that over the overlap. Es is the folded spectrum over the signal's band less N0. The roll-off
  (which bins are the overlap) comes from fitting an RRC's correlation shape, and is given too. Any colour of noise
  outside the signal's band is irrelevant, and nothing but the symbol rate (searched +-2%) and independent symbols is
  assumed: a stream whose symbols aren't yet random (an interleaver filling from zero) reads low. Above about 15 dB
  the window's smoothing takes the coherence down and it reads low (17-19 at 20 dB), still high. None: no correlation
  found; noise alone reads None or under -3 dB (100 of 100 blocks).
- `blind(x, sps)`: from the band the signal leaves empty: the eigenvalues of the covariance of `L`-sample windows. The
  signal takes at most 2/sps of them (roll-off <= 1, MDL finds how many), the rest are the noise. Precise in white
  noise, but noise a front-end filter has shaped is not white: where some passband noise shows past the signal the
  eigenvalues are not flat and it says None, but where the filter's edge is right at the signal's band all it sees is
  the stopband, flat, and it reads 26-38 dB high. Use it only where the noise is white across Fs.

Measured (2026-11-23, error mean / std in dB; cyclic and blind on 2048 symbols at 4 sps, roll-off 0.35):

    Es/N0                     -10          -5           0            10           20           noise through a front-end filter
    from_sequence, ZC 63      -0.3 / 3.0                +0.1 / 1.1                +0.15 / 0.6  within 0.5
    from_sequence, ZC 839      0.0 / 0.7                 0.0 / 0.3                 0.0 / 0.16
    cyclic (RRC 0.2-0.5)      unusable     -0.2..-1.5   -0.1..-0.3   -0.1..-0.5   -1.1..-2.8   the same as white
                                           / 0.7-1.3    / 0.2-0.6    / 0.1-0.3    / 0.1-0.2
    blind                     None         -0.2 / 0.3   -0.0 / 0.2   +0.0 / 0.1   +0.0 / 0.04  None, or 26-38 dB high
"""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from ..math.prob import mixture_variance
from ..base import Stage

FLAT = 0.9 # blind: noise eigenvalues flatter than this (geometric / arithmetic mean) are white noise

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

class SNR:
    """A signal-to-noise estimate before anything is locked (`from_sequence`, `blind`)

    snr_db: signal over noise power per sample. esn0_db: per symbol (snr_db + 10 log10 sps). noise_var: noise power per
    sample (complex). gain: the complex channel gain (from_sequence). delay: the lag of the peak (from_sequence,
    periodic). dims: the signal's eigenvalues out of L (blind). beta: the fitted roll-off, sps: the samples a symbol the
    correlation was strongest at (cyclic). n: samples used."""
    __slots__ = ("snr_db", "esn0_db", "noise_var", "gain", "delay", "dims", "beta", "sps", "n")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        return f"SNR {self.snr_db:.1f} dB, Es/N0 {self.esn0_db:.1f} dB"

    __repr__ = __str__

def _db(sig, noise):
    return float(10*np.log10(max(sig, 1e-30)/max(noise, 1e-30)))

def m2m4(y, points):
    """Es/N0 (dB) of symbols y were they of the constellation `points`, from their second and fourth moments alone
    (Pauluzzi and Beaulieu, 2000): M2 = S + N, M4 = k S^2 + 4 S N + 2 N^2 with k = E|p|^4 of the unit-rms points (1 for
    any PSK; under 2 for any digital constellation), so S = sqrt((2 M2^2 - M4)/(2 - k)). No decisions, no carrier phase,
    no pulse: rpps.sync.detect scores each candidate at its own. It fades for QAM at high SNR. Clamped to -30..60 dB"""
    y = np.asarray(y)
    pts = np.asarray(points, dtype=np.complex128)
    k = float(np.mean(np.abs(pts)**4)/np.mean(np.abs(pts)**2)**2)
    a2 = np.abs(y)**2
    M2, M4 = float(np.mean(a2)), float(np.mean(a2**2))
    S = np.sqrt(max(2*M2**2 - M4, 0.0)/(2 - k)) if k < 2 else 0.0
    return float(np.clip(10*np.log10(max(S, 1e-30)/max(M2 - S, 1e-30)), -30.0, 60.0))

def from_sequence(y, ref, periodic=True, guard=2, sps=1.0):
    """SNR of samples `y` carrying the known sequence `ref` (one sample per element, `sps` samples a symbol for esn0)

    periodic: y is one period of a CAZAC sequence (rp.sequence.zadoff_chu) at any cyclic shift and carrier phase, as an
        OFDM symbol or a cyclic-prefixed preamble gives it: the peak of the circular correlation and the `guard` lags
        each side of it are the signal (the channel's taps), every other lag noise alone. Needs len(y) == len(ref).
    otherwise: y and ref aligned; the least-squares gain, and the residual as the noise."""
    y = np.asarray(y, dtype=np.complex128)
    ref = np.asarray(ref, dtype=np.complex128)
    N = len(ref)
    if len(y) != N or N < 8:
        raise ValueError(f"from_sequence needs y as long as ref (at least 8): {len(y)} and {N}")
    ref = ref/np.sqrt(np.mean(np.abs(ref)**2))
    if periodic:
        if 2*guard + 1 >= N//2:
            raise ValueError(f"guard {guard} leaves too few noise lags in {N}")
        c = np.fft.ifft(np.fft.fft(y)*np.conj(np.fft.fft(ref))) # c[k] = mean_n y[n] conj(ref[n - k])
        p = np.abs(c)**2
        k = int(np.argmax(p))
        taps = (k + np.arange(-guard, guard + 1)) % N
        off = np.ones(N, bool)
        off[taps] = False
        lag = float(np.mean(p[off])) # E|c|^2 of a noise lag: noise_var / N
        sig = float(np.sum(p[taps])) - len(taps)*lag
        noise = lag*N
        gain, delay = complex(c[k]), k
    else:
        gain = complex(np.vdot(ref, y)/N)
        noise = float(np.sum(np.abs(y - gain*ref)**2)/(N - 1))
        sig, delay = abs(gain)**2, None
    snr = _db(sig, noise)
    return SNR(snr_db=snr, esn0_db=snr + 10*np.log10(sps), noise_var=noise, gain=gain, delay=delay, n=N)

_BETAS = np.round(np.arange(0.05, 1.001, 0.025), 3) # the roll-offs cyclic fits
_OVER = 16 # cyclic: the fine grid the window's smoothing is computed on
_VAR = 1.05 # cyclic: |cross spectrum|^2 noise alone leaves, over Sa Sb / segments (half-overlapped Hann; measured 1.04-1.08)

def _shifted_spectra(x, sps, nseg):
    """Welch spectra of x shifted by -1/2T (u) and +1/2T (v), and their cross spectrum, on half-overlapped Hann
    segments; frequencies in symbol-rate units. Returns (f, Su, Sv, cross, segments)"""
    n = np.arange(len(x))
    u, v = x*np.exp(-1j*np.pi*n/sps), x*np.exp(1j*np.pi*n/sps)
    w = np.hanning(nseg)
    hop = nseg//2
    K = (len(x) - nseg)//hop + 1
    idx = np.arange(nseg)[None, :] + hop*np.arange(K)[:, None]
    U, V = np.fft.fft(u[idx]*w, axis=1), np.fft.fft(v[idx]*w, axis=1)
    return (np.fft.fftfreq(nseg)*sps, np.mean(np.abs(U)**2, axis=0), np.mean(np.abs(V)**2, axis=0),
            np.mean(U*np.conj(V), axis=0), K)

@lru_cache(maxsize=64)
def _templates(nseg, sps):
    """The correlation |E U V*|^2 at Es = 1 for each roll-off, after the window's smoothing, on the bins |f| < 1/2T"""
    M = nseg*_OVER
    kernel = np.abs(np.fft.fft(np.hanning(nseg), M))**2
    kernel = np.fft.fft(kernel/kernel.sum())
    ff = np.fft.fftfreq(M)*sps
    band = np.abs(np.fft.fftfreq(nseg)*sps) < 0.5
    out = []
    for b in _BETAS:
        c = np.where(np.abs(ff) < b/2, np.cos(np.pi*ff/b)/2, 0.0)
        c = np.real(np.fft.ifft(np.fft.fft(c)*kernel))[::_OVER][band] # smoothed, back on the bins
        out.append(c**2)
    return band, np.array(out)

def _coherent_noise(a, b, c):
    """The one noise density N that best makes (a - N)(b - N) = c on every bin (least squares): a linearly modulated
    signal's two shifted copies are fully coherent where both are there, whatever its pulse. The minimum is a root of a
    cubic, in [0, the mean of min(a, b)] (not their minimum: at low SNR the least of many noisy bins is well under N)"""
    if len(a) == 0:
        return 0.0
    s, p = a + b, a*b - c
    n = len(a)
    roots = np.roots([2*n, -3*np.sum(s), np.sum(s**2 + 2*p), -np.sum(p*s)])
    top = float(np.mean(np.minimum(a, b)))
    cands = [0.0, top] + [float(r.real) for r in roots if abs(r.imag) < 1e-9*max(1.0, abs(r)) and 0 < r.real < top]
    cost = lambda N: float(np.sum(((a - N)*(b - N) - c)**2)) # noqa: E731
    return min(cands, key=cost)

def cyclic(x, sps, nseg=None, search=0.02, step=0.005):
    """Es/N0 of samples of one linearly modulated signal (RRC, any roll-off) from its spectral correlation at the symbol
    rate (see the module). None when there is no correlation to fit; esn0_db inf when N0 came out <= 0 (high SNR).

    x: complex samples, the signal centred; sps: samples a symbol (more than 1.25). nseg: the Welch segment (default
    about 32 bins a symbol rate). search / step: the cyclic frequency is looked for within +-search of 1/sps (relative)
    in these steps, the strongest correlation kept: a rate a few tenths of a percent off turns the two copies half a
    cycle apart over a segment and the correlation, and Es with it, fades (sps in the result: the rate it was found at)"""
    x = np.asarray(x, dtype=np.complex128)
    if sps <= 1.25:
        return None
    nseg = int(nseg or 2**int(np.ceil(np.log2(32*sps))))
    if len(x) < 8*nseg:
        return None
    best = None
    for rel in np.arange(-search, search + step/2, step) if search else (0.0,):
        s = sps/(1 + rel)
        f, Sa, Sb, X, K = _shifted_spectra(x, s, nseg)
        e = np.sum(np.abs(X)**2 - Sa*Sb/(K/1.9))
        if best is None or e > best[0]:
            best = (e, s, f, Sa, Sb, X, K)
    _, sps, f, Sa, Sb, X, K = best
    band, T = _templates(nseg, float(round(sps, 6)))
    f, Sa, Sb, X = f[band], Sa[band], Sb[band], X[band]
    C2 = np.abs(X)**2 - _VAR*Sa*Sb/K # noise alone leaves |X|^2 of about this
    norms = np.sum(T**2, axis=1)
    es2 = np.maximum(T @ C2/np.where(norms > 0, norms, 1), 0.0) # least squares Es^2 for each roll-off
    resid = np.sum((C2[None, :] - es2[:, None]*T)**2, axis=1)
    resid[norms == 0] = np.inf
    i = int(np.argmin(resid))
    if es2[i] <= 0:
        return None
    beta = float(_BETAS[i])
    N = _coherent_noise(Sa[np.abs(f) < beta/2], Sb[np.abs(f) < beta/2], C2[np.abs(f) < beta/2])
    # Es: the folded spectrum over the signal's band, less the noise (f + 1/2T in the band for Sa, f - 1/2T for Sb)
    Es = (np.sum(Sa[f < beta/2] - N) + np.sum(Sb[f > -beta/2] - N))/len(f)
    esn0 = float(10*np.log10(Es/N)) if N > 0 and Es > 0 else np.inf if Es > 0 else -np.inf
    return SNR(snr_db=esn0 - 10*np.log10(sps), esn0_db=esn0, beta=beta, sps=float(sps), n=len(x))

def blind(x, sps, L=32, max_samples=16384, flat=FLAT):
    """SNR of oversampled samples of one signal, from the band it leaves empty (see the module). None when it can't tell

    x: complex samples, the signal centred; sps: samples a symbol (at least 1.3). L: the window, eigenvalues of its
    covariance; max_samples: the most used. flat: how flat the noise eigenvalues must be (geometric over arithmetic mean)"""
    x = np.asarray(x, dtype=np.complex128)[:max_samples]
    n = len(x) - L + 1
    if sps < 1.3 or n < 8*L:
        return None
    W = np.lib.stride_tricks.sliding_window_view(x, L)
    lam = np.sort(np.linalg.eigvalsh(W.T @ W.conj()/n))[::-1].clip(min=1e-30) # (L, L) covariance, largest first
    kmax = min(int(np.ceil(2*L/sps)), L - 4) # the signal's band: at most (1 + roll-off)/sps <= 2/sps of the samples
    best, k = np.inf, 0
    for j in range(kmax + 1): # MDL (Wax and Kailath), j signal eigenvalues
        t = lam[j:]
        mdl = -n*len(t)*(np.mean(np.log(t)) - np.log(np.mean(t))) + 0.5*j*(2*L - j)*np.log(n)
        if mdl < best:
            best, k = mdl, j
    if k == 0:
        return None
    t = lam[k:] # the noise: every eigenvalue past the signal's
    if np.exp(np.mean(np.log(t)))/np.mean(t) < flat:
        return None
    noise = float(np.mean(t))
    snr = _db(float(np.mean(lam)) - noise, noise)
    return SNR(snr_db=snr, esn0_db=snr + 10*np.log10(sps), noise_var=noise, dims=k, n=len(x))

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
