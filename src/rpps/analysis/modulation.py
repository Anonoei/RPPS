"""Modulation detection and feed-forward parameter estimation

Everything here is feed-forward: symbol rate and timing from the matrix matched filter,
frequency/phase from the M-th power of the symbols, modulation from the likelihood of the
corrected symbols under each candidate constellation.
"""
import numpy as np

from ..dsp.rate import find_baud
from ..math.interp import peak
from ..math.prob import mixture_variance
from ..sync.time.mat import matrix_mf, centres_in, best_tau
from .quality import measure

class Detection:
    """Result of `detect`. Stored in meta.stream.det"""
    __slots__ = (
        "mod", "baud", "sps", "beta", "tau",
        "cfo", "phase", "M", "gain",
        "scores", "n", "esn0", "ebn0"
    )
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        q = "" if self.esn0 is None else f", Es/N0 {self.esn0:.1f} dB, Eb/N0 {self.ebn0:.1f} dB"
        return (f"{self.mod}: baud {self.baud:.1f}, sps {self.sps:.4f}, beta {self.beta}, "
                f"cfo {self.cfo:.1f} Hz, margin {self.margin:.3f} nats/sym{q}")

    __repr__ = __str__

    @property
    def margin(self):
        """Score lead of the detected modulation over the runner-up"""
        s = sorted(self.scores.values(), reverse=True)
        return s[0] - s[1] if len(s) > 1 else np.inf

def symmetry(points, tol=1e-3, max_order=16):
    """Order M of the rotational symmetry of a point set (the M-th power removes the modulation)"""
    pts = np.asarray(points)
    for M in range(max_order, 1, -1):
        rot = pts * np.exp(2j*np.pi/M)
        if all(np.min(np.abs(rot[i] - pts)) < tol for i in range(len(pts))):
            return M
    return 1

def candidates(names=None):
    """Load candidate constellations from config/mod: {name: unit-rms points}"""
    from .. import mod # Late: mod imports after sync/analysis
    if names is None:
        names = [n for stems in mod.ls().values() for n in stems if not n.startswith("dvbs2_")] # Labelling variants are not separate modulations
    out = {}
    for name in names:
        pts = np.asarray(mod.load(name).points.arr, dtype=np.complex128)
        out[name] = pts / np.sqrt(np.mean(np.abs(pts)**2))
    return out

def correct(y, pts, M, pad=4):
    """Feed-forward frequency + phase correction for a candidate. Returns (symbols, cfo cycles/symbol, phase)"""
    n = len(y)
    cfo = 0.0
    if M > 1:
        spec = np.abs(np.fft.fft(y**M * np.hanning(n), pad*n))
        k, d = peak(spec, circular=True)
        f = (k + d)/(pad*n)
        cfo = (f - (1 if f > 0.5 else 0))/M
    y1 = y * np.exp(-2j*np.pi*cfo*np.arange(n))
    phase = 0.0
    if M > 1:
        phase = (np.angle(np.sum(y1**M)) - np.angle(np.mean(pts**M)))/M
    return y1 * np.exp(-1j*phase), cfo, phase

def score(y, pts, iters=8):
    """Mean log-likelihood per symbol under equiprobable points with Gaussian noise (sigma estimated)"""
    d2 = np.abs(y[:, None] - pts[None, :])**2
    var = mixture_variance(d2, iters)
    logp = -d2/(2*var)
    m = logp.max(axis=1)
    ll = m + np.log(np.sum(np.exp(logp - m[:, None]), axis=1)) - np.log(len(pts)) - np.log(2*np.pi*var)
    return float(np.mean(ll))

def refine_timing(x, sps, beta, span, rounds=2):
    """Refine sps from the drift of the best timing between the start and end of x. Returns (sps, tau)"""
    n = len(x)
    seg = max(n//8, int(60*sps))
    if seg*2 < n:
        for _ in range(rounds):
            a, _e = best_tau(x, sps, beta, span, 0, seg)
            b, _e = best_tau(x, sps, beta, span, n-seg, n)
            drift = (b - a + sps/2) % sps - sps/2 # Timing moved by this much...
            sps += drift/((n - seg)/sps) # ...over this many symbols
    tau, _e = best_tau(x, sps, beta, span)
    return sps, tau

def detect(x, Fs, names=None, betas=(0.2, 0.25, 0.35), span=8):
    """Find symbol rate, timing, modulation, frequency offset and phase from baseband samples

    x: complex baseband at Fs, with the signal of interest centred and filtered
    names: candidate modulations, default everything in config/mod
    betas: RRC roll-offs to try
    Returns a Detection; its tau is relative to x[0].
    """
    cands = candidates(names)
    baud0, _ = find_baud(x, Fs)
    best = None
    for beta in betas:
        sps, tau = refine_timing(x, Fs/baud0, beta, span)
        c = centres_in(len(x), tau + np.ceil((span*sps + 5 - tau)/sps)*sps, sps, span)
        y = matrix_mf(x, c, sps, beta, span)
        gain = np.sqrt(np.mean(np.abs(y)**2))
        y = y / gain
        scores, fits = {}, {}
        for name, pts in cands.items():
            M = symmetry(pts)
            yc, cfo, phase = correct(y, pts, M)
            scores[name] = score(yc, pts)
            fits[name] = (M, cfo, phase)
        top = max(scores, key=scores.get)
        if best is None or scores[top] > best[0]:
            M, cfo, phase = fits[top]
            q = measure(correct(y, cands[top], M)[0], cands[top])
            baud = Fs/sps
            best = (scores[top], Detection(
                mod=top, baud=baud, sps=sps, beta=beta, tau=tau,
                cfo=cfo*baud, phase=phase, M=M, gain=gain,
                scores=scores, n=len(y), esn0=q.esn0_db, ebn0=q.ebn0_db
            ))
    return best[1]
