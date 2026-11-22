"""Acquisition: symbol rate, timing, modulation, frequency offset and phase from baseband samples: once (`detect`), as
snapshots of separate blocks (`Lock`: a display that seeks, pauses and retunes) and as the stream arrives (`Acquire`)

Everything here is feed-forward: symbol rate and timing from the matrix matched filter,
frequency/phase from the M-th power of the symbols, modulation from the likelihood of the
corrected symbols under each candidate constellation.
"""
from types import SimpleNamespace

import numpy as np

from ..base import Stage
from ..errors import NoLock
from ..math.interp import peak
from ..math.prob import mixture_variance
from .rate import find_baud
from .time.mat import matrix_mf, centres_in, best_tau
from .quality import measure

class Detection:
    """Result of `detect`. Acquire stamps it on its blocks: meta.det

    line: strength of the symbol-rate line the rate came from, over its local floor (noise peaks at about 3-4; a rate
        taken from a weaker line is a guess)"""
    __slots__ = (
        "mod", "baud", "sps", "beta", "tau",
        "cfo", "phase", "M", "gain",
        "scores", "n", "esn0", "ebn0", "line"
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
                n for n in symbol.ls().get("psk", []) if "extends" not in _config.read_raw(_config.find("constellation", f"psk/{n}")))
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

def score(y, pts, iters=8):
    """Mean log-likelihood per symbol under equiprobable points with Gaussian noise (sigma estimated)"""
    d2 = np.abs(y[:, None] - pts[None, :])**2
    var = mixture_variance(d2, iters)
    logp = -d2/(2*var)
    m = logp.max(axis=1)
    ll = m + np.log(np.sum(np.exp(logp - m[:, None]), axis=1)) - np.log(len(pts)) - np.log(2*np.pi*var)
    return float(np.mean(ll))

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

def detect(x, Fs, names=None, betas=(0.2, 0.25, 0.35), span=8, baud=None):
    """Find symbol rate, timing, modulation, frequency offset and phase from baseband samples

    x: complex baseband at Fs, with the signal of interest centred and filtered
    names: candidate modulations, default everything in config/constellation
    betas: RRC roll-offs to try
    baud: the symbol rate when already known (a channel plan, rpps.analysis.spectrum.cyclic): timing refines it, no search
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
                scores=scores, n=len(y), esn0=q.esn0_db, ebn0=q.ebn0_db, line=line
            ))
    return best[1]

class Lock:
    """A signal's symbols, as a snapshot: found once by a full search (`Lock.find`), then followed block by block
    (`update`) without searching again. Nothing is held between blocks, and they need not be consecutive, so a display
    that seeks, pauses or changes channel gets an answer for every frame:

        lk = rp.sync.Lock.find(x, Fs)               # raises rp.errors.NoLock(reason) when there is nothing to lock to
        lk = lk.update(x_next)                      # same modulation and roll-off; timing, rate and cfo followed
        lk.symbols, lk.quality.esn0_db, lk.stale    # stale: the signal is not the one found, search again
        lk = lk.refine(x_longer)                    # the rate searched again on more signal (Acquire, as history grows)

    det: the Detection for this block (rate, timing on its lattice from x[0], cfo, phase, gain, modulation)
    found: the Detection of the full search the lock started from
    symbols: matched-filtered, frequency and phase corrected, unit rms, at `centres` (sample positions in the block)
    points: the constellation (unit rms). quality: rpps.sync.quality.Quality of the symbols
    stale: why the lock no longer fits (Es/N0 fell `drop_db` below the search's, the rate moved more than `drift` from
        it, or another of the searched modulations now fits better), None while it does
    """
    __slots__ = ("det", "found", "symbols", "centres", "points", "quality", "stale", "Fs", "span", "drop_db", "drift")
    MIN_SYMBOLS = 16 # Fewer symbols in a block are no measurement
    RECHECK_MARGIN = 0.05 # nats/symbol another modulation must lead by to call the lock stale (a QPSK lock on 8PSK at
    # 6 dB Es/N0: 8PSK leads by about 0.1; the right one leads the next by 0.13 there, more above)

    @classmethod
    def find(cls, x, Fs, names=None, betas=(0.2, 0.25, 0.35), span=8, baud=None, lock=5.0, min_samples=1024,
             drop_db=6.0, drift=2e-3):
        """Search x for a signal: its symbol rate, modulation, roll-off, timing, cfo and phase (`detect`)

        names / betas: candidate modulations and roll-offs (default: the PSK family, 0.2 0.25 0.35)
        baud: the symbol rate when known: no rate search and no line needed
        lock: symbol-rate line strength needed over its local floor (noise peaks at about 3-4); None takes any line
        min_samples: fewer samples are not searched (the rate search goes down to Fs/16: at least 64 symbols)
        drop_db / drift: when `update` calls the lock stale (an Es/N0 drop [dB], a relative rate change)
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
            det = detect(x, Fs, names, betas, span, baud=baud)
            det.line = line
            lk = cls._make(x, Fs, det, candidates([det.mod])[det.mod], span, det, drop_db, drift)
        return cls._made(lk, x, det)

    def update(self, x, rounds=1, recheck=True):
        """The same signal in a new block x (anywhere in the stream: consecutive or not): timing and rate followed from
        this lock's (`track_timing`, `rounds` times), cfo and phase measured afresh, no search. Returns a new Lock;
        raises NoLock when x holds too few symbols

        recheck: score the other modulations the search tried on these symbols (the same matched filter output), and
        call the lock stale when one fits better by RECHECK_MARGIN: at low SNR a change of modulation can cost less Es/N0
        than `drop_db`"""
        x = self._samples(x)
        d = self.det
        with self._numbers():
            sps, tau = track_timing(x, d.sps, d.beta, self.span, rounds)
            det = Detection(mod=d.mod, baud=self.Fs/sps, sps=sps, beta=d.beta, tau=tau, M=d.M, scores=self.found.scores,
                            line=self.found.line)
            lk = self._make(x, self.Fs, det, self.points, self.span, self.found, self.drop_db, self.drift, recheck)
        return self._made(lk, x, det)

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
    def _made(cls, lk, x, det):
        if lk is None:
            raise NoLock(f"{len(x)} samples hold fewer than {cls.MIN_SYMBOLS} symbols at {det.baud:.6g} Bd")
        d = lk.det
        if not (np.isfinite([d.sps, d.tau, d.cfo, d.phase, d.gain, lk.quality.esn0_db]).all()
                and np.all(np.isfinite(lk.symbols))):
            raise NoLock("nothing like symbols here (the estimates are not finite)")
        return lk

    @classmethod
    def _make(cls, x, Fs, det, points, span, found, drop_db, drift, recheck=False):
        """The lock for block x at det's rate, roll-off and timing: symbols, cfo, phase, gain and quality filled in"""
        sps, tau = det.sps, det.tau
        c = centres_in(len(x), tau + np.ceil((span*sps + 5 - tau)/sps)*sps, sps, span)
        if len(c) < cls.MIN_SYMBOLS:
            return None
        y = matrix_mf(x, c, sps, det.beta, span)
        gain = float(np.sqrt(np.mean(np.abs(y)**2))) or 1.0
        syms, cfo, phase = correct(y/gain, points, det.M)
        q = measure(syms, points)
        det.cfo, det.phase, det.gain, det.n = cfo*det.baud, phase, gain, len(syms)
        det.esn0, det.ebn0 = q.esn0_db, q.ebn0_db
        lk = cls.__new__(cls)
        lk.det, lk.found, lk.symbols, lk.centres, lk.points, lk.quality = det, found, syms, c, points, q
        lk.Fs, lk.span, lk.drop_db, lk.drift = Fs, span, drop_db, drift
        lk.stale = None
        if found is not det:
            if found.esn0 is not None and q.esn0_db < found.esn0 - drop_db:
                lk.stale = f"Es/N0 {q.esn0_db:.1f} dB, {found.esn0 - q.esn0_db:.1f} dB under the search's"
            elif abs(det.sps/found.sps - 1) > drift:
                lk.stale = f"rate {det.baud:.6g} Bd, {abs(det.sps/found.sps - 1)*1e6:.0f} ppm from the search's"
            elif recheck and found.scores and len(found.scores) > 1:
                own = score(syms, points)
                others = {}
                for name, pts in candidates([n for n in found.scores if n != det.mod]).items():
                    others[name] = score(correct(y/gain, pts, symmetry(pts))[0], pts)
                best = max(others, key=others.get)
                if others[best] > own + cls.RECHECK_MARGIN:
                    lk.stale = f"looks like {best} now ({others[best] - own:.2f} nats/symbol over {det.mod})"
        return lk

    def __repr__(self):
        return f"<Lock {self.det}{', stale: ' + self.stale if self.stale else ''}>"

class Acquire(Stage):
    """Detect as soon as the signal is there, then keep refining as the stream grows

    The matrix matched filter and the likelihood only need a few hundred symbols, so there is no
    fixed wait: blocks are held only until the symbol-rate line is significant (`lock` times the
    local floor, noise peaks at about 3-4), which can be the first block. Then everything held
    is released at once, and the estimates are re-run on the growing history (doubling each time, up
    to `history` samples) so baud and cfo tighten. Refinements retest only the detected
    modulation and roll-off, which is much cheaper.

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
        return SimpleNamespace(chunks=[], count=0, next=0, done=False, held=True, lock=None)

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
        if st.lock is None:
            if st.count < self.min_samples:
                return samples[:0] # Nothing there (yet)
            try: # After max_wait samples, whatever the line
                st.lock = Lock.find(x, meta.Fs, self.names, self.betas, baud=self.baud, min_samples=self.min_samples,
                                    lock=self.lock if st.count < self.max_wait else None)
            except NoLock:
                return samples[:0]
            st.next = 2*st.count
        elif st.count >= st.next: # Refine on the grown history: same modulation and roll-off, the rate searched again
            st.lock = st.lock.refine(x, baud=self.baud and st.lock.det.baud)
            st.next = 2*st.count
        else:
            meta.det = st.lock.det
            return samples
        det = st.lock.det
        meta.det = det
        meta.calibrate_sps(det.sps)
        if st.count >= self.history:
            st.done = True
            st.chunks = []
        if st.held: # First detection: release what was held
            st.held = False
            return x
        return samples
