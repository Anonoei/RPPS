"""Acquisition: symbol rate, timing, modulation, frequency offset and phase from baseband samples: once (`detect`), as
snapshots of separate blocks (`Lock`: a display that seeks, pauses and retunes) and as the stream arrives (`Acquire`)

Everything here is feed-forward: symbol rate and timing from the matrix matched filter,
frequency/phase from the M-th power of the symbols, modulation from the likelihood of the
corrected symbols under each candidate constellation.

The likelihood needs the noise. Fitted per candidate (each its own variance, the points at the symbols' rms, noise
included), a denser constellation explains noise better, so at low SNR the densest candidate won (noise alone was named
32APSK, QPSK at 4 dB 64QAM): the generalised likelihood's known bias to higher orders. So each candidate is scored at
its own Es/N0 from the symbols' moments (rpps.sync.quality.m2m4: what that candidate's kurtosis says the signal and
noise are; the quasi-hybrid likelihood test), unless the fit's own winner reads over FIT_MIN dB: there the fit is
right and M2M4 fades (for QAM), so the fit; at the caller's Es/N0 when it gives one (`esn0=`, from a known sequence: rpps.sync.quality.from_sequence
on a Zadoff-Chu). It needs nothing but the symbols. With the rate given, BPSK is named 7 dB lower, QPSK from 1.75 dB
(it was 4.75), 32QAM and the APSKs 1.75 dB lower; 64QAM's 10 dB is honest now (tools/modclass_bench.py).

`min_evidence` makes detect say it doesn't know (mod None, the leader in `best`) rather than name a guess: when the
leader doesn't beat Gaussian noise of the same power by NULL_MARGIN nats in all (`Detection.over_noise`: is there a
constellation at all), or doesn't lead the runner-up by min_evidence nats (`Detection.evidence`: which one). 10 nats
never named noise; it named 8PSK as 16PSK and 16QAM as 64QAM now and then at 8 dB, and at 0-4 dB names 9-22%.
"""
from types import SimpleNamespace

import numpy as np

from ..base import Stage
from ..errors import NoLock
from ..math.interp import peak
from ..math.prob import mixture_variance
from .rate import find_baud
from .time.mat import matrix_mf, centres_in, best_tau
from .quality import measure, decide, errors, summary, m2m4

FIT_MIN = 12.5     # dB: the fit's winner reading over this is scored by the fit, else by M2M4 (measured: 12 dips at
                   # 8 dB, where the leaning fit over-reads, 14 at 14-16 dB, where M2M4 fades)
NULL_MARGIN = 10.0 # nats: what the leader must beat Gaussian noise by for min_evidence (noise alone: 1 block in 60)

class Detection:
    """Result of `detect`. Acquire stamps it on its blocks: meta.det

    line: strength of the symbol-rate line the rate came from, over its local floor (noise peaks at about 3-4; a rate
        taken from a weaker line is a guess)
    mod: the modulation, None when detect was asked for `min_evidence` and the evidence is short (`best` the leader)
    how: how the candidates were scored: "given" (at the caller's esn0, `given`), "moments" (each at its M2M4 Es/N0) or
        "fit" (each its own variance). null: the score of Gaussian noise of the symbols' power (see over_noise)"""
    __slots__ = (
        "mod", "baud", "sps", "beta", "tau",
        "cfo", "phase", "M", "gain",
        "scores", "n", "esn0", "ebn0", "line",
        "best", "how", "given", "null"
    )
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        q = "" if self.esn0 is None else f", Es/N0 {self.esn0:.1f} dB, Eb/N0 {self.ebn0:.1f} dB"
        return (f"{self.mod if self.mod is not None else f'unclear ({self.best}?)'}: baud {self.baud:.1f}, sps {self.sps:.4f}, beta {self.beta}, "
                f"cfo {self.cfo:.1f} Hz, margin {self.margin:.3f} nats/sym{q}")

    __repr__ = __str__

    @property
    def margin(self):
        """Score lead of the detected modulation over the runner-up"""
        s = sorted(self.scores.values(), reverse=True)
        return s[0] - s[1] if len(s) > 1 else np.inf

    @property
    def evidence(self):
        """Total log-likelihood lead of the leader over the runner-up (nats): margin times symbols"""
        return self.margin*(self.n or 0)

    @property
    def over_noise(self):
        """Total log-likelihood lead of the leader over Gaussian noise of the same power (nats): a constellation at all"""
        return (self.scores[self.best] - self.null)*(self.n or 0) if self.null is not None else np.inf

def symmetry(points, tol=1e-3, max_order=16):
    """Order M of the rotational symmetry of a point set (the M-th power removes the modulation)"""
    pts = np.asarray(points)
    for M in range(max_order, 1, -1):
        rot = pts * np.exp(2j*np.pi/M)
        if all(np.min(np.abs(rot[i] - pts)) < tol for i in range(len(pts))):
            return M
    return 1

_CANDIDATES = {} # (config search path, name) -> unit-rms points (read-only), None -> the default names

def candidates(names=None):
    """Load candidate constellations from config/constellation: {name: unit-rms points}. Each is read once per config
    search path (a Lock rechecks them every block: finding the files would cost more than scoring them)"""
    from .. import symbol  # Late: symbol imports after sync
    from .. import _config
    where = tuple(_config._EXTRA)
    if names is None: # The PSK family's own point sets: a child (a labelling or a rotation of a parent) is not another modulation
        names = _CANDIDATES.get((where, None))
        if names is None:
            names = _CANDIDATES[(where, None)] = tuple(
                n for n in symbol.ls(examples=False).get("psk", []) if "extends" not in _config.read_raw(_config.find("constellation", f"psk/{n}")))
    out = {}
    for name in names:
        pts = _CANDIDATES.get((where, name))
        if pts is None:
            pts = np.asarray(symbol.load_or_generate(name).points, dtype=np.complex128)
            pts = pts / np.sqrt(np.mean(np.abs(pts)**2))
            pts.flags.writeable = False
            _CANDIDATES[(where, name)] = pts
        out[name] = pts
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

def score(y, pts, iters=8, esn0=None):
    """Mean log-likelihood per symbol of unit-rms y under equiprobable points with Gaussian noise: the variance fitted
    (EM), or, given esn0 (dB), the points scaled to the signal's share of the power and the noise the rest"""
    return _score(y, pts, iters, esn0)[0]

def _score(y, pts, iters=8, esn0=None):
    """`score` and the variance per real axis it used"""
    if esn0 is None:
        d2 = np.abs(y[:, None] - pts[None, :])**2
        var = mixture_variance(d2, iters)
    else:
        r = 10**(esn0/10)
        d2 = np.abs(y[:, None] - np.sqrt(r/(1 + r))*pts[None, :])**2
        var = 1/(2*(1 + r)) # per real axis
    logp = -d2/(2*var)
    m = logp.max(axis=1)
    ll = m + np.log(np.sum(np.exp(logp - m[:, None]), axis=1)) - np.log(len(pts)) - np.log(2*np.pi*var)
    return float(np.mean(ll)), float(var)

def score_all(ycs, esn0=None):
    """Scores of {name: (corrected unit-rms symbols, points)} and how they were made: at the caller's esn0 ("given"); each
    fitting its own variance when the fit's winner reads over FIT_MIN dB ("fit"); else each at its own M2M4 Es/N0
    ("moments")"""
    if esn0 is not None:
        return {n: score(y, p, esn0=esn0) for n, (y, p) in ycs.items()}, "given"
    fits = {n: _score(y, p) for n, (y, p) in ycs.items()}
    fit = {n: f[0] for n, f in fits.items()}
    var = fits[max(fit, key=fit.get)][1] # per axis, unit-rms y: signal 1 - 2 var
    if 10*np.log10(max(1 - 2*var, 1e-12)/(2*var)) > FIT_MIN:
        return fit, "fit"
    return {n: score(y, p, esn0=m2m4(y, p)) for n, (y, p) in ycs.items()}, "moments"

def null_score(y):
    """`score` of unit-rms y as Gaussian noise of the same power (complex, unit variance)"""
    return float(-np.mean(np.abs(y)**2) - np.log(np.pi))

def track_timing(x, sps, beta, span=8, rounds=2):
    """Refine sps from the drift of the best timing between the start and end of x. Returns (sps, tau)

    A rate a little off makes the timing walk across x: the walk between its first and last eighth (at least 60 symbols each)
    corrects sps, `rounds` times. The walk is seen modulo a symbol, so sps must be right to within about sps/(2 len(x))
    relative (40 ppm for 60000 samples at 4 sps): a lock follows a clock, a search gets close first. tau is on the absolute
    lattice tau + k*sps from x[0]. x too short for two such parts keeps sps."""
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

def detect(x, Fs, names=None, betas=(0.2, 0.25, 0.35), span=8, baud=None, esn0=None, min_evidence=None):
    """Find symbol rate, timing, modulation, frequency offset and phase from baseband samples

    x: complex baseband at Fs, with the signal of interest centred and filtered
    names: candidate modulations, default everything in config/constellation
    betas: RRC roll-offs to try
    baud: the symbol rate when already known (a channel plan, rpps.analysis.spectrum.cyclic): timing refines it, no search
    esn0: the Es/N0 when known (rpps.sync.quality.from_sequence on a known sequence): every candidate is scored at it
    min_evidence: nats the winner must lead the runner-up by in all (Detection.evidence), with it NULL_MARGIN over
        Gaussian noise (Detection.over_noise), or mod is None and `best` holds the leader (see the module)
    Returns a Detection; its tau is relative to x[0].
    """
    cands = candidates(names)
    baud0, line = (float(baud), None) if baud else find_baud(x, Fs)
    best = None
    for beta in betas:
        sps, tau = track_timing(x, Fs/baud0, beta, span)
        c = centres_in(len(x), tau + np.ceil((span*sps + 5 - tau)/sps)*sps, sps, span)
        y = matrix_mf(x, c, sps, beta, span)
        gain = np.sqrt(np.mean(np.abs(y)**2))
        y = y / gain
        ycs, fits = {}, {}
        for name, pts in cands.items():
            M = symmetry(pts)
            yc, cfo, phase = correct(y, pts, M)
            ycs[name] = (yc, pts)
            fits[name] = (M, cfo, phase)
        scores, how = score_all(ycs, esn0)
        top = max(scores, key=scores.get)
        if best is None or scores[top] > best[0]:
            M, cfo, phase = fits[top]
            q = measure(correct(y, cands[top], M)[0], cands[top])
            baud = Fs/sps
            best = (scores[top], Detection(
                mod=top, baud=baud, sps=sps, beta=beta, tau=tau,
                cfo=cfo*baud, phase=phase, M=M, gain=gain,
                scores=scores, n=len(y), esn0=q.esn0_db, ebn0=q.ebn0_db, line=line,
                best=top, how=how, given=esn0, null=null_score(y)
            ))
    det = best[1]
    if min_evidence is not None and (det.over_noise < NULL_MARGIN or det.evidence < min_evidence):
        det.mod = None
    return det

class Lock:
    """A signal's symbols, as a snapshot: found once by a full search (`Lock.find`), then followed block by block
    (`update`) without searching again. Nothing is held between blocks, and they need not be consecutive, so a display
    that seeks, pauses or changes channel gets an answer for every frame:

        lk = rp.sync.Lock.find(x, Fs)               # raises rp.errors.NoLock(reason) when there is nothing to lock to
        lk = lk.update(x_next)                      # same modulation and roll-off; timing, rate and cfo followed
        lk.symbols, lk.quality.esn0_db, lk.stale    # stale: the signal is not the one found, search again
        lk = lk.refine(x_longer)                    # the rate searched again on more signal (Acquire, as history grows)
        lk = lk.follow(x_after, at=pos)             # the block right after this one: its symbols join on with no slip

    det: the Detection for this block (rate, timing on its lattice from x[0], cfo, phase, gain, modulation)
    found: the Detection of the full search the lock started from
    symbols: matched-filtered, frequency and phase corrected, unit rms, at `centres` (sample positions in the block)
    points: the constellation (unit rms). quality: rpps.sync.quality.Quality of the symbols
    stale: why the lock no longer fits (Es/N0 fell `drop_db` below the search's, the rate moved more than `drift` from
        it, or another of the searched modulations now fits better), None while it does
    x: the block the lock is of (after `follow`, with the end of the one before that the filter needs)
    end: the stream position just past x, when `at` gave where x starts (None otherwise)

    What an analyser shows of the block: `decisions()`, `errors()` and `summary()` (rpps.sync.quality: EVM, MER, IQ
    offset, gain imbalance, quadrature error), `derotated()` (x as the symbols see it), `matched(up)` (the matched filter
    between the symbols: eye diagrams) and `reference()` (the ideal waveform of the decided symbols)
    """
    __slots__ = ("det", "found", "symbols", "centres", "points", "quality", "stale", "Fs", "span", "drop_db", "drift", "x",
                 "end")
    MIN_SYMBOLS = 16 # Fewer symbols in a block are no measurement
    RECHECK_MARGIN = 0.05 # nats/symbol another modulation must lead by to call the lock stale (a QPSK lock on 8PSK at
    # 6 dB Es/N0: 8PSK leads by about 0.1; the right one leads the next by 0.13 there, more above)

    @classmethod
    def find(cls, x, Fs, names=None, betas=(0.2, 0.25, 0.35), span=8, baud=None, lock=5.0, min_samples=1024,
             drop_db=6.0, drift=2e-3, at=None, esn0=None, min_evidence=None):
        """Search x for a signal: its symbol rate, modulation, roll-off, timing, cfo and phase (`detect`)

        names / betas: candidate modulations and roll-offs (default: the PSK family, 0.2 0.25 0.35)
        baud: the symbol rate when known: no rate search and no line needed
        lock: symbol-rate line strength needed over its local floor (noise peaks at about 3-4); None takes any line
        min_samples: fewer samples are not searched (the rate search goes down to Fs/16: at least 64 symbols)
        drop_db / drift: when `update` calls the lock stale (an Es/N0 drop [dB], a relative rate change)
        esn0 / min_evidence: as `detect` (a known Es/N0; how sure the modulation must be: NoLock when it isn't, or
            when the samples hold no symbol-rate correlation)
        at: the stream position of x[0], when known: `follow` then checks the next block is the one right after
        Raises NoLock (reason, line) when x is too short or has no clear symbol rate"""
        x = cls._samples(x, min_samples, f"the search needs {min_samples}")
        with cls._numbers():
            line = None
            if not baud:
                baud, line = find_baud(x, Fs)
                if lock is not None and not line >= lock:
                    raise NoLock(f"no symbol rate (line {line:.1f}, needs {lock:g}): a carrier, or noise?", line)
                if not 0 < baud < Fs/2:
                    raise NoLock(f"no symbol rate (the strongest line is at {baud:.6g} Bd): a carrier, or noise?", line)
            det = detect(x, Fs, names, betas, span, baud=baud, esn0=esn0, min_evidence=min_evidence)
            det.line = line
            if det.mod is None:
                raise NoLock(f"modulation unclear: {det.best} leads by {det.evidence:.0f} nats (needs {min_evidence:g})"
                             if det.over_noise >= NULL_MARGIN else
                             f"nothing like a constellation: {det.over_noise:.0f} nats over Gaussian noise", line)
            lk = cls._make(x, Fs, det, candidates([det.mod])[det.mod], span, det, drop_db, drift)
        return cls._made(lk, x, det, at)

    def update(self, x, rounds=1, recheck=True, at=None):
        """The same signal in a new block x (anywhere in the stream: consecutive or not): timing and rate followed from
        this lock's (`track_timing`, `rounds` times), cfo and phase measured afresh, no search. Returns a new Lock;
        raises NoLock when x holds too few symbols

        recheck: score the other modulations the search tried on these symbols (the same matched filter output), and
        call the lock stale when one fits better by RECHECK_MARGIN: at low SNR a change of modulation can cost less Es/N0
        than `drop_db`. at: the stream position of x[0], when known (for `follow`)"""
        x = self._samples(x)
        d = self.det
        with self._numbers():
            sps, tau = track_timing(x, d.sps, d.beta, self.span, rounds)
            det = Detection(mod=d.mod, baud=self.Fs/sps, sps=sps, beta=d.beta, tau=tau, M=d.M, scores=self.found.scores,
                            line=self.found.line)
            lk = self._make(x, self.Fs, det, self.points, self.span, self.found, self.drop_db, self.drift, recheck)
        return self._made(lk, x, det, at)

    MAX_STEP = 0.3 # symbols the timing may move between consecutive blocks before `follow` calls the lock stale

    def follow(self, x, at=None, rounds=1, recheck=True):
        """The block right after this lock's (contiguous: its first sample follows the last of `x`), its symbols joining
        on to these with no symbol dropped or repeated: the first is the one after the last here, the timing, rate,
        carrier offset and phase carried on (the phase on the branch of its M-fold symmetry that continues this one's, so
        the decisions keep their rotation). Concatenating the symbols of a chain of follows gives the sequence one
        `update` over the joined record would (each block's estimates its own). Returns a new Lock, whose `x` starts with
        the end of this one's that the matched filter needs

        at: the stream position of x[0]. When both this lock's `end` and `at` are known and differ, the blocks are not
            contiguous: NoLock (start again with `update`). Stale (`stale`) also when the timing moved more than
            MAX_STEP of a symbol between the blocks: a slip is likely, so the joined symbols can't be trusted
        Raises NoLock when x holds too few symbols"""
        x = self._samples(x)
        if at is not None and self.end is not None and int(at) != self.end:
            raise NoLock(f"not the next block: it starts at {int(at)}, this one ended at {self.end} "
                         f"({int(at) - self.end:+d} samples)")
        d = self.det
        sps = d.sps
        nxt = self.centres[-1] + sps # the next symbol instant, in this x's positions
        keep = max(0, int(np.floor(nxt - self.span*sps - 6)))
        j = np.concatenate((self.x[keep:], x))
        c0 = nxt - keep
        with self._numbers():
            sps2, tau = track_timing(j, sps, d.beta, self.span, rounds)
            step = (tau - c0 + sps2/2) % sps2 - sps2/2 # the measured lattice nearest the predicted instant
            det = Detection(mod=d.mod, baud=self.Fs/sps2, sps=sps2, beta=d.beta, tau=c0 + step, M=d.M,
                            scores=self.found.scores, line=self.found.line)
            # The carrier phase predicted at the first new symbol, len(centres) symbols after this lock's first
            phase = d.phase + 2*np.pi*d.cfo/d.baud*len(self.centres)
            lk = self._make(j, self.Fs, det, self.points, self.span, self.found, self.drop_db, self.drift, recheck,
                            first=c0 + step, phase=phase)
        lk = self._made(lk, j, det)
        lk.end = None if self.end is None else self.end + len(x)
        if at is not None:
            lk.end = int(at) + len(x)
        if lk.stale is None and abs(step) > self.MAX_STEP*sps2:
            lk.stale = f"the timing moved {step/sps2:+.2f} symbols between the blocks: a slip is likely"
        return lk

    def refine(self, x, baud=None):
        """The search again on more of the signal (x: a longer stretch), its modulation and roll-off kept: the rate is
        searched afresh (or `baud`, when known), so an estimate made on few samples is corrected however far off it was,
        which `update`, following, can't. The result is the lock's new reference (`found`). Raises NoLock as `find`"""
        x = self._samples(x)
        with self._numbers():
            det = detect(x, self.Fs, [self.det.mod], (self.det.beta,), self.span, baud=baud)
            det.scores = self.found.scores # (only the kept modulation was scored)
            lk = self._make(x, self.Fs, det, self.points, self.span, det, self.drop_db, self.drift)
        return self._made(lk, x, det)

    # ---- NoLock, not numpy's errors, for what holds no symbols

    @staticmethod
    def _samples(x, need=1, why="no samples"):
        """x as an array, or NoLock: too few samples, not all finite, all zero"""
        x = np.asarray(x)
        if len(x) < need:
            raise NoLock(f"{len(x)} samples, {why}")
        if not np.all(np.isfinite(x)):
            raise NoLock("the samples are not all finite (NaN or inf)")
        if not np.any(x):
            raise NoLock("no signal: the samples are all zero")
        return x

    @staticmethod
    def _numbers():
        """The search's arithmetic under its own numpy error state (whatever np.seterr the caller set): what holds no
        symbols ends in a NoLock, from the checks around it, rather than a FloatingPointError"""
        return np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore")

    @classmethod
    def _made(cls, lk, x, det, at=None):
        if lk is None:
            raise NoLock(f"{len(x)} samples hold fewer than {cls.MIN_SYMBOLS} symbols at {det.baud:.6g} Bd")
        d = lk.det
        if not (np.isfinite([d.sps, d.tau, d.cfo, d.phase, d.gain, lk.quality.esn0_db]).all()
                and np.all(np.isfinite(lk.symbols))):
            raise NoLock("nothing like symbols here (the estimates are not finite)")
        lk.end = None if at is None else int(at) + len(x)
        return lk

    @classmethod
    def _make(cls, x, Fs, det, points, span, found, drop_db, drift, recheck=False, first=None, phase=None):
        """The lock for block x at det's rate, roll-off and timing: symbols, cfo, phase, gain and quality filled in.
        first: the first symbol's instant (default: the first the filter fits); phase: the carrier phase predicted there,
        which picks the branch of the M-fold ambiguity"""
        sps, tau = det.sps, det.tau
        c = centres_in(len(x), tau + np.ceil((span*sps + 5 - tau)/sps)*sps if first is None else first, sps, span)
        if len(c) < cls.MIN_SYMBOLS:
            return None
        y = matrix_mf(x, c, sps, det.beta, span)
        gain = float(np.sqrt(np.mean(np.abs(y)**2))) or 1.0
        syms, cfo, ph = correct(y/gain, points, det.M)
        if phase is not None and det.M > 1: # the branch nearest the prediction
            turn = 2*np.pi/det.M*np.round((phase - ph)/(2*np.pi/det.M))
            syms, ph = syms*np.exp(-1j*turn), ph + turn
        phase = ph
        q = measure(syms, points)
        det.cfo, det.phase, det.gain, det.n = cfo*det.baud, phase, gain, len(syms)
        det.esn0, det.ebn0 = q.esn0_db, q.ebn0_db
        lk = cls.__new__(cls)
        lk.det, lk.found, lk.symbols, lk.centres, lk.points, lk.quality = det, found, syms, c, points, q
        lk.Fs, lk.span, lk.drop_db, lk.drift, lk.x = Fs, span, drop_db, drift, x
        lk.stale = None
        if found is not det:
            if found.esn0 is not None and q.esn0_db < found.esn0 - drop_db:
                lk.stale = f"Es/N0 {q.esn0_db:.1f} dB, {found.esn0 - q.esn0_db:.1f} dB under the search's"
            elif abs(det.sps/found.sps - 1) > drift:
                lk.stale = f"rate {det.baud:.6g} Bd, {abs(det.sps/found.sps - 1)*1e6:.0f} ppm from the search's"
            elif recheck and found.scores and len(found.scores) > 1:
                # scored as the search scores (score_all), on this block's symbols
                ycs = {det.mod: (syms, points)}
                for name, pts in candidates([n for n in found.scores if n != det.mod]).items():
                    ycs[name] = (correct(y/gain, pts, symmetry(pts))[0], pts)
                others, _ = score_all(ycs, found.given)
                own = others.pop(det.mod)
                best = max(others, key=others.get)
                if others[best] > own + cls.RECHECK_MARGIN:
                    lk.stale = f"looks like {best} now ({others[best] - own:.2f} nats/symbol over {det.mod})"
        return lk

    # ---- what an analyser shows of the block

    def decisions(self):
        """The nearest constellation point to each symbol (unit rms, as `points`)"""
        return decide(self.symbols, self.points)

    def errors(self):
        """Each symbol's error against its decision: rpps.sync.quality.Errors (vector; evm_pct and mag_pct as a percentage
        of the constellation's rms; phase_deg)"""
        return errors(self.symbols, self.points)

    def summary(self):
        """An analyser's error summary of the symbols: rpps.sync.quality.Summary (EVM rms and peak, MER, magnitude and
        phase error, IQ offset, gain imbalance, quadrature error)"""
        return summary(self.symbols, self.points)

    def _turn(self, pos):
        """The carrier offset, phase and gain the symbols were corrected by, at sample positions `pos` of x"""
        d = self.det
        return np.exp(-2j*np.pi*d.cfo/self.Fs*(pos - self.centres[0]) - 1j*d.phase)/d.gain

    def derotated(self):
        """x with the carrier offset and phase taken out and scaled as the symbols are (their matched filter at `centres`
        gives about `symbols`; exactly for no carrier offset, the offset turning inside the filter otherwise)"""
        return self.x*self._turn(np.arange(len(self.x)))

    def matched(self, up=1, exact=False):
        """The matched filter's output between the symbols, `up` points a symbol: (t, z). t: time in symbols from the
        first symbol instant (centres[0]); z: the output corrected as the symbols are, z[::up] being `symbols`. From the
        first symbol to the last; the eye diagram of the block is z against t modulo a symbol or two.

        The filter runs once over the block, at its sample rate, and its output (band-limited) is read between the samples
        by a 12-tap windowed sinc (rpps.math.interp.fractional): about 1e-3 rms of the symbols' scale from the filter's own
        rows (a few 1e-3 at most, at 2 samples a symbol), the instants exact. exact: the filter itself at every point (its matrix, one row a point: slow for a dense eye)"""
        up = int(up)
        if up < 1:
            raise ValueError("matched: up is a whole number of points a symbol, 1 or more")
        d, c = self.det, self.centres
        pos = c[0] + np.arange((len(c) - 1)*up + 1)*(d.sps/up)
        pos[::up] = c # (the instants themselves, to the last bit)
        if exact:
            z = np.empty(len(pos), np.complex128)
            step = max(1, 2**20//int(2*self.span*d.sps + 2)) # Rows of the filter matrix a time
            for a in range(0, len(pos), step):
                p = pos[a:a + step]
                z[a:a + step] = matrix_mf(self.x, p, d.sps, d.beta, self.span)*self._turn(p)
            return (pos - c[0])/d.sps, z
        from ..signal.filters import rrc_pulse
        from ..math.interp import fractional_edge
        L = int(np.ceil(self.span*d.sps))
        h = rrc_pulse(np.arange(-L, L + 1)/d.sps, d.beta)
        h /= np.sqrt(np.sum(h**2))
        x = np.asarray(self.x, dtype=np.complex128)
        n = len(x) + len(h) - 1
        if len(h) > 64: # by FFT
            N = 1 << int(np.ceil(np.log2(n)))
            y = np.fft.ifft(np.fft.fft(x, N)*np.fft.fft(h, N))[:n]
        else:
            y = np.convolve(x, h)
        # y[i]: the filter's output at sample i - L of x (zeros beyond x); read at pos
        z = fractional_edge(y.astype(np.complex64), pos + L, 12).astype(np.complex128)*self._turn(pos)
        z[::up] = self.symbols
        return (pos - c[0])/d.sps, z

    def reference(self):
        """The ideal waveform of the block: the decided symbols through the root raised cosine pulse at `centres`, on x's
        samples, scaled and turned as `derotated()` (compare the two). Zero where no symbol of the lock reaches (the
        ends of x hold symbols the lock has not)"""
        from .time.mat import _rrc_rows
        d, c = self.det, self.centres
        dec = self.decisions()
        S = self.span*d.sps
        lags = np.arange(int(np.ceil(2*S)) + 2)
        r = np.zeros(len(self.x), np.complex128)
        step = max(1, 2**20//len(lags))
        for a in range(0, len(c), step):
            cc = c[a:a + step]
            j0 = np.ceil(cc - S).astype(np.int64)
            w = _rrc_rows(j0 - cc, lags, d.sps, d.beta)
            w /= np.sqrt(np.sum(w**2, axis=1, keepdims=True))
            idx = j0[:, None] + lags
            ok = (idx >= 0) & (idx < len(r))
            np.add.at(r, idx[ok], (dec[a:a + step, None]*w)[ok])
        return r

    def __repr__(self):
        return f"<Lock {self.det}{', stale: ' + self.stale if self.stale else ''}>"

class Acquire(Stage):
    """Detect as soon as the signal is there, then keep refining as the stream grows

    The matrix matched filter and the likelihood only need a few hundred symbols, so there is no
    fixed wait: blocks are held only until the symbol-rate line is significant (`lock` times the
    local floor, noise peaks at about 3-4). Then everything held is released at once, and the
    estimates are re-run on the growing history so baud and cfo tighten, up to `history` samples.

    Every decision is made on the stream's first P samples at fixed P (min_samples, then doubling), never on
    whatever has arrived, so how the stream is cut into blocks changes nothing: a block that passes one or more of
    those positions has them run in order. The modulation and roll-off are searched again in full at each position
    until two in a row agree (a first decision on a few thousand samples at low SNR can be wrong, and refining would
    keep it); after that, refinements retest only them, which is much cheaper. A block carries the detection made by
    its end (meta.det). At the stream's end (flush) what has not settled is decided on the whole stream.

    Calibrates meta.sps (to the detected one), so following stages get the fractional sps.

    lock: symbol-rate line strength required to detect
    max_wait: samples after which to detect regardless
    min_samples: samples held before the first try, whatever the line: the rate search goes down to Fs/16, so this is at
        least 64 symbols (a narrow signal decimated far, a carrier, gives a few samples a block, and a line in a few hundred
        samples is no rate)
    baud: the symbol rate when known (a channel plan, rpps.analysis.spectrum.cyclic): no waiting for a line, no rate search
    """
    __slots__ = ("lock", "max_wait", "history", "names", "betas", "min_samples", "baud")
    def __init__(self, lock=5.0, max_wait=2**17, history=2**16, names=None, betas=(0.2, 0.25, 0.35), min_samples=1024,
                 baud=None):
        self.baud = baud
        self.min_samples = min_samples
        self.lock = lock
        self.max_wait = max_wait
        self.history = history
        self.names = names
        self.betas = betas

    def new_state(self, meta):
        return SimpleNamespace(chunks=[], count=0, next=self.min_samples, done=False, held=True, lock=None, settled=False)

    def step(self, samples, st, meta):
        # Every block that passes carries the detection it passed with (meta.det): stages running behind in other
        # threads use that, not a refinement made for a later block
        if st.done or len(samples) == 0:
            if st.lock is not None:
                meta.det = st.lock.det
            return samples
        st.chunks.append(samples)
        st.count += len(samples)
        x = np.concatenate(st.chunks) if len(st.chunks) > 1 else st.chunks[0]
        st.chunks = [x]
        while st.count >= st.next and not st.done: # every decision position this block reached, in order
            self._decide(x[:st.next], st, meta)
            st.next *= 2
        if meta.flush and not st.done and st.count >= self.min_samples and (st.lock is None or not st.settled):
            self._decide(x, st, meta) # the stream's end is a position too: decide on all of it
        if st.lock is None:
            return samples[:0] # Nothing there (yet)
        det = st.lock.det
        meta.det = det
        meta.calibrate_sps(det.sps)
        if st.done:
            st.chunks = []
        if st.held: # First detection: release what was held
            st.held = False
            return x
        return samples

    def _decide(self, x, st, meta):
        """The decision at one position: x is the stream's first P samples"""
        P = len(x)
        try:
            if st.lock is None: # Not there yet: a line `lock` strong, or any line after max_wait
                st.lock = Lock.find(x, meta.Fs, self.names, self.betas, baud=self.baud, min_samples=self.min_samples,
                                    lock=self.lock if P < self.max_wait else None)
            elif not st.settled: # The full search again on twice the samples: settled when it agrees
                lk = Lock.find(x, meta.Fs, self.names, self.betas, baud=self.baud, min_samples=self.min_samples,
                               lock=None)
                st.settled = (lk.det.mod, lk.det.beta) == (st.lock.det.mod, st.lock.det.beta)
                st.lock = lk
            else: # Same modulation and roll-off, the rate searched again
                st.lock = st.lock.refine(x, baud=self.baud and st.lock.det.baud)
        except NoLock:
            return
        if P >= self.history:
            st.done = True
