"""Frequency-shift keying and continuous-phase modulation from the instantaneous frequency: tones, rate, index, pulse

    r = rp.analysis.fsk.analyze(x, Fs)     # x at baseband (a channel), constant envelope
    print(r)                               # e.g. "2-FSK: 9600.0 Bd, tones 4800.0 Hz apart (h 0.50: MSK), deviation
                                           #  +-2400.0 Hz, centre +312.0 Hz, Gaussian-like transitions (0.6 symbols)"
    sym = rp.analysis.fsk.demodulate(x, Fs, r)   # tone index per symbol, 0 = lowest

A constant-envelope signal carries its data in its frequency: the discriminator output f(t) = d/dt arg x / (2 pi) is a
pulse-amplitude signal whose levels are the tones. So:
  rate    the symbol rate is a line of f^2 / (df)^2 (rpps.analysis.cyclic.lines: significant or not, with a P_fa)
  timing  transitions (df)^2 pulse once a symbol at the symbol boundaries; the centres are half a symbol later
  tones   f at the centres clusters at M equally spaced levels: M (2, 4, 8) is the largest whose clusters are well apart
          (spacing over 4 rms widths); spacing / rate is the modulation index h (0.5 MSK, 1 Sunde's FSK)
  pulse   a frequency pulse longer than a symbol (Gaussian: GMSK, GFSK; raised cosine: partial-response CPM) pulls a
          symbol's frequency toward its neighbours'; rectangular pulses (CPFSK, MSK) do not
"""
import numpy as np

from .cyclic import noise_floor

from . import cyclic
from .cyclic import comb_rate

ENVELOPE = 1.25 # E|x|^4/(E|x|^2)^2 in band over this: not constant envelope (8PSK RRC 0.35 is ~1.3-1.6)
REPEAT = 0.2 # A tone the same as the one before this much more often than 1/M: sampled at twice the rate
SPLIT = 2.5 # Width a cluster must shrink by when split for its halves to be two tones
ISI_GAUSSIAN = 0.06 # Neighbours pulling a symbol this far (tone spacings) off its tone: a pulse longer than a symbol

class FSK:
    """Result of `analyze`. M: tones (None: not FSK). baud (Hz), baud_line (cyclic.Line), spacing (Hz between adjacent
    tones), h (spacing / baud), deviation (Hz, of the outer tones from the centre), centre (Hz: the carrier offset),
    levels (Hz, the tones), separation (spacing over the clusters' rms width), isi (how far neighbours pull a symbol off its tone, in spacings:
    ~0 rectangular, 0.1 and more Gaussian / partial response),
    tau (first symbol centre, samples), envelope (E|x|^4 / (E|x|^2)^2), why (when M is None)"""
    __slots__ = ("M", "baud", "baud_line", "spacing", "h", "deviation", "centre", "levels", "separation", "isi",
                 "tau", "envelope", "why")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def name(self):
        if self.M is None:
            return None
        if self.M == 2 and abs(self.h - 0.5) < 0.06:
            return "GMSK" if self.gaussian else "MSK"
        return f"{self.M}-{'GFSK' if self.gaussian else 'FSK'}"

    @property
    def gaussian(self):
        """The frequency pulse is longer than a symbol (Gaussian, raised cosine): neighbours pull a symbol off its tone"""
        return self.isi is not None and self.isi > ISI_GAUSSIAN

    def __str__(self):
        if self.M is None:
            return f"not FSK: {self.why}"
        pulse = "Gaussian / partial-response" if self.gaussian else "rectangular"
        return (f"{self.name}: {self.baud:.1f} Bd, tones {self.spacing:.1f} Hz apart (h {self.h:.2f}), deviation "
                f"+-{self.deviation:.1f} Hz, centre {self.centre:+.1f} Hz, {pulse} pulse (ISI {self.isi:.2f} "
                f"spacings), clusters {self.separation:.1f} widths apart, envelope {self.envelope:.2f}")

    __repr__ = __str__

def discriminator(x, Fs):
    """Instantaneous frequency (Hz) between samples: arg(x[n] conj(x[n-1])) Fs / 2 pi, len(x) - 1 values"""
    x = np.asarray(x, dtype=np.complex128)
    return np.angle(x[1:]*np.conj(x[:-1]))*Fs/(2*np.pi)

def _kmeans1(v, M, iters=30):
    """1-D k-means from equally spaced quantiles: (sorted centres, labels)"""
    c = np.quantile(v, (np.arange(M) + 0.5)/M)
    for _ in range(iters):
        lab = np.argmin(np.abs(v[:, None] - c[None, :]), axis=1)
        new = np.array([v[lab == j].mean() if np.any(lab == j) else c[j] for j in range(M)])
        if np.allclose(new, c):
            break
        c = new
    o = np.argsort(c)
    rank = np.empty(M, int)
    rank[o] = np.arange(M)
    return c[o], rank[lab]

def _band(x, Fs, margin=0.5, nfft=1024):
    """x with what lies well outside its band (the bins over 0 dB SNR, widened by `margin` of its width each side) taken off"""
    n = len(x)
    nfft = int(min(nfft, 2**int(np.log2(max(n//8, 16)))))
    seg = n//nfft
    S = np.fft.fftshift(np.mean(np.abs(np.fft.fft(x[:seg*nfft].reshape(seg, nfft)*np.hanning(nfft), axis=1))**2, axis=0))
    N0 = noise_floor(S)
    on = np.flatnonzero(S > 2*N0)
    if len(on) == 0:
        return x
    f = (np.arange(nfft) - nfft//2)/nfft
    a, b = f[on[0]], f[on[-1]]
    w = b - a
    a, b = a - margin*w, b + margin*w
    if a <= -0.5 and b >= 0.5:
        return x
    fx = np.fft.fftfreq(n)
    G = ((fx >= a) & (fx <= b)).astype(float)
    k = max(3, n//nfft) | 1 # soften the edges over a PSD bin
    G = np.convolve(np.concatenate((G[-(k//2):], G, G[:k//2])), np.ones(k)/k, mode="valid")
    return np.fft.ifft(np.fft.fft(x)*G)

def _repeats(v, lev):
    """How often a symbol's tone is the one before's (1/M for independent data)"""
    lab = np.argmin(np.abs(v[:, None] - lev[None, :]), axis=1)
    return float(np.mean(lab[1:] == lab[:-1]))

def _timing(f, sps):
    """First symbol centre (samples) from the discriminator f: boundaries are where (df)^2 pulses (after smoothing over
    an eighth of a symbol: f's noise is white), the centres half a symbol later"""
    g = np.diff(_smooth(f, sps/8))**2
    n = np.arange(len(g))
    c = np.sum(g*np.exp(-2j*np.pi*n/sps))
    boundary = (-np.angle(c)/(2*np.pi)*sps) % sps + 1.0 # +1: g[n] is between f[n] and f[n+1], f[n] between x[n], x[n+1]
    return (boundary + sps/2) % sps

def _centre_freqs(x, Fs, sps, tau):
    """(centres, frequency of each symbol, Hz): the phase advance across the middle half of the symbol (for CPFSK the
    phase is a line there, so this is the least noisy estimate), relative to the median discriminator output"""
    off = np.median(discriminator(x, Fs))
    fmax = np.percentile(np.abs(discriminator(x, Fs) - off), 99)
    # Half a symbol, but short enough that the outer tones' phase advance stays inside +-pi (0.4 of a cycle)
    L = max(1, min(int(round(sps/2)), int(0.4*Fs/max(fmax, 1e-9))))
    centres = tau + sps*np.arange(int((len(x) - 1 - L - tau)//sps) + 1)
    c = np.round(centres).astype(int)
    ok = (c - L//2 >= 0) & (c - L//2 + L < len(x))
    c, centres = c[ok], centres[ok]
    a = c - L//2
    v = np.angle(x[a + L]*np.conj(x[a])*np.exp(-2j*np.pi*off/Fs*L))*Fs/(2*np.pi*L)
    return centres, v

def _levels(v, Ms=(2, 4, 8), min_separation=3.0):
    """The tones: (M, centres, separation) or (None, ., best separation). M clusters are equally spaced, apart, about
    equally used, and (over the M/2 before) each half the size of one of those: splitting a cluster of one noisy tone in two
    only takes its width to ~0.6, splitting two tones takes it to the noise. Levels and the separation's width come from
    symbols whose neighbours are the same tone where there are enough: a Gaussian or partial-response pulse moves a
    symbol's frequency toward its neighbours' (ISI), which a cluster's width should not count"""
    best = (None, None, 0.0)
    prev = None # all-symbol width of the M taken
    for M in Ms:
        if len(v) < 8*M:
            break
        c, lab = _kmeans1(v, M)
        same = np.zeros(len(v), bool)
        same[1:-1] = (lab[1:-1] == lab[:-2]) & (lab[1:-1] == lab[2:])
        if all(np.count_nonzero(same & (lab == j)) >= 4 for j in range(M)):
            c = np.array([np.median(v[same & (lab == j)]) for j in range(M)])
            v_, lab_ = v[same], lab[same]
        else:
            v_, lab_ = v, lab
        d = np.diff(c)
        if np.any(d <= 0):
            continue
        spacing = float(np.median(d))
        width = np.sqrt(np.mean((v_ - c[lab_])**2))
        sep = spacing/max(width, 1e-12)
        even = np.max(np.abs(d - spacing)) < 0.35*spacing
        # Every tone is used about as often (scrambled data): clusters a timing error makes between the tones are thin
        full = np.min(np.bincount(lab, minlength=M)) >= 0.3*len(v)/M
        wall = np.sqrt(np.mean((v - c[lab])**2))
        split = prev is None or wall < prev/SPLIT
        if even and full and split and sep >= min_separation:
            best, prev = (M, c, sep), wall
        elif best[0] is None:
            best = (None, c, max(best[2], sep))
    return best

def _smooth(f, n):
    n = max(1, int(n))
    return np.convolve(f, np.ones(n)/n, mode="same") if n > 1 else f

def analyze(x, Fs, lo=None, hi=None, pfa=1e-3, mask=True, force=False, max_samples=2**21):
    """FSK / CPM parameters of baseband samples x (see the module); lo/hi: symbol rates searched (Hz). A signal whose
    envelope (in its band) is not nearly constant is not analysed unless `force`"""
    x = np.asarray(x, dtype=np.complex128)[:max_samples]
    p2 = np.mean(np.abs(x)**2) if len(x) else 0.0
    env = float(np.mean(np.abs(x)**4)/max(p2**2, 1e-300)) if len(x) else 0.0
    out = dict(envelope=env)
    if len(x) < 512:
        return FSK(why="too few samples", **out)
    if len(x) >= 1024 and mask:
        # Out-of-band noise only, with a wide margin: an FSK spectrum's skirts carry its transitions and its deviation
        x = _band(x, Fs)
    p2b = np.mean(np.abs(x)**2)
    out["envelope"] = float(np.mean(np.abs(x)**4)/max(p2b**2, 1e-300)) # in band: less of the noise
    if out["envelope"] > ENVELOPE and not force:
        return FSK(why=f"the envelope is not constant ({out['envelope']:.2f}): a linear modulation, OFDM or noise", **out)
    lo = Fs/256 if lo is None else lo
    hi = Fs/2 if hi is None else hi
    f = discriminator(x, Fs)
    f = f - np.median(f)
    lines = []
    # f is white noise plus a symbol-rate signal: smoothing it (over 1, 4, 16 samples) before squaring keeps the line of
    # a slow or Gaussian-filtered signal (GMSK's f^2 is nearly flat) out of the noise
    for k in (1, 4, 16):
        if hi*k > 2*Fs and k > 1:
            hi_k = min(hi, Fs/k)
        else:
            hi_k = hi
        fk = _smooth(f, k)
        for z, kind in ((fk**2, f"f^2/{k}"), (np.diff(fk)**2, f"(df)^2/{k}")):
            for l in cyclic.lines(z, Fs, lo, hi_k, pfa, top=4):
                if l.significant:
                    l.kind = kind
                    lines.append(l)
    if not lines:
        return FSK(why="no symbol-rate line in the instantaneous frequency", **out)
    baud, line = comb_rate(lines, Fs, lo)
    tau = _timing(f, Fs/baud)
    centres, v = _centre_freqs(x, Fs, Fs/baud, tau)
    M, lev, sep = _levels(v)
    if M is not None and _repeats(v, lev) > 1/M + REPEAT and baud/2 >= lo:
        # Every tone comes twice: a Gaussian pulse can leave the rate's own line weaker than its second harmonic's
        t2 = _timing(f, 2*Fs/baud)
        c2, v2 = _centre_freqs(x, Fs, 2*Fs/baud, t2)
        M2, lev2, sep2 = _levels(v2)
        if M2 is not None and _repeats(v2, lev2) <= 1/M2 + REPEAT:
            baud, tau, centres, v, M, lev, sep = baud/2, t2, c2, v2, M2, lev2, sep2
    best = (None, baud, line, tau, centres, v, M, lev, sep)
    _, baud, line, tau, centres, v, M, lev, sep = best
    off = float(np.median(discriminator(x, Fs)))
    out.update(baud=baud, baud_line=line, tau=tau, separation=sep)
    if M is None:
        return FSK(why=f"the frequency at the symbol centres does not fall in equally spaced clusters (best {sep:.1f} widths)",
                   **out)
    spacing = float(np.median(np.diff(lev)))
    centre = float(np.mean(lev)) + off
    # ISI: how far a symbol's frequency is pulled off its tone (in tone spacings) when both neighbours are other tones,
    # less the same when both are its own (noise alone): about 0 for rectangular pulses (CPFSK, MSK), 0.1-0.3 for a
    # Gaussian pulse (GMSK BT 0.3-0.5) or a raised-cosine one (partial-response CPM)
    lab = np.argmin(np.abs(v[:, None] - lev[None, :]), axis=1)
    dev = np.abs(v - lev[lab])/spacing
    other = np.zeros(len(v), bool)
    own = np.zeros(len(v), bool)
    other[1:-1] = (lab[:-2] != lab[1:-1]) & (lab[2:] != lab[1:-1])
    own[1:-1] = (lab[:-2] == lab[1:-1]) & (lab[2:] == lab[1:-1])
    isi = float(np.mean(dev[other]) - np.mean(dev[own])) if other.any() and own.any() else 0.0
    out.update(M=M, spacing=spacing, h=spacing/baud, deviation=spacing*(M - 1)/2, centre=centre,
               levels=lev + off, isi=max(isi, 0.0))
    return FSK(**out)

def demodulate(x, Fs, r, mask=True):
    """Tone index (0 = lowest) of every symbol, at the centres `analyze` found (r: its result; the same x, filtered the
    same way: mask)"""
    if r.M is None:
        raise ValueError(f"Not FSK: {r.why}")
    x = np.asarray(x, dtype=np.complex128)
    if len(x) >= 1024 and mask:
        x = _band(x, Fs)
    _, v = _centre_freqs(x, Fs, Fs/r.baud, r.tau)
    v = v + np.median(discriminator(x, Fs)) # _centre_freqs is relative to the median
    return np.argmin(np.abs(v[:, None] - np.asarray(r.levels)[None, :]), axis=1)

def cpm_index(x, Fs, baud, tau=None, h_range=(0.05, 2.0), steps=1951, lags=(1, 2, 3, 4)):
    """The modulation index h of a full-response CPM (CPFSK, MSK, M-ary), from the phase lattice: between symbol
    boundaries N apart the phase moves pi h (a sum of N odd levels), whose parity is N's, so exp(j dphi / h) is (-1)^N for
    every pair at the right h, and averages away at a wrong one. A submultiple of h fits too: the largest h scoring near
    the best is taken. tau: a symbol centre (samples, as `analyze` gives; default from the discriminator).
    Returns (h, score 0..1). Partial-response CPM (L > 1, GMSK) leaves pulses straddling the boundaries: lower scores"""
    x = np.asarray(x, dtype=np.complex128)
    sps = Fs/baud
    f = discriminator(x, Fs)
    if tau is None:
        tau = _timing(f - np.median(f), sps)
    ph = np.concatenate(([0.0], np.cumsum(np.angle(x[1:]*np.conj(x[:-1]))))) # unwrapped phase, rad
    b = (tau - sps/2) % sps + sps*np.arange(int((len(x) - 1 - (tau - sps/2) % sps)//sps))
    pb = np.interp(b, np.arange(len(ph)), ph)
    # The carrier offset adds 2 pi f N sps/Fs to every N-symbol step: take it off (the steps' mean, lattice symmetric)
    hs = np.linspace(h_range[0], h_range[1], steps)
    score = np.zeros(steps)
    for N in lags:
        d = pb[N:] - pb[:-N]
        d = d - np.mean(d)
        score += np.abs(np.mean(np.exp(1j*d[None, :]/hs[:, None]), axis=1))
    score /= len(lags)
    best = score.max()
    peaks = np.flatnonzero((score[1:-1] >= score[:-2]) & (score[1:-1] >= score[2:]) & (score[1:-1] >= 0.9*best)) + 1
    k = peaks[-1] if len(peaks) else int(np.argmax(score)) # the largest h whose peak is about the best
    return float(hs[k]), float(score[k])
