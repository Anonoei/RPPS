"""What else than noise is in locked symbols: amplifier compression (AM/AM, AM/PM), IQ imbalance, phase noise, neighbours

    a = rp.analysis.impairments.amplifier(y, points)   # per ring: gain (dB) and phase (deg) against the inner ring
    q = rp.analysis.impairments.iq(y)                  # image rejection (dB), the imbalance, and y corrected
    p = rp.analysis.impairments.phase_noise(y, points) # rms phase error against what the noise explains, and how slow
    n = rp.analysis.impairments.neighbours(x, Fs)      # other bands over the floor, and whether they touch this one

When Es/N0 and the SNR the floor allows disagree (rpps.analysis.cancel's excess, or a gap between SNR and Es/N0 over 1 dB)
these say why:
  amplifier   a saturating amplifier gives outer points less gain (AM/AM) and turns them (AM/PM): each ring's mean symbol
              against its point, relative to the innermost ring
  iq          a receiver whose I and Q arms differ in gain or phase adds a little of the conjugate: y = mu s + nu s*. For a
              constellation symmetric under 90 degree turns (QPSK, QAM, APSK, 8PSK) E[s^2] = 0, so E[y^2] / E[|y|^2] =
              2 mu nu / (|mu|^2 + |nu|^2): blind, no decisions needed; the image rejection is |mu/nu|^2
  phase noise the phase of each symbol against its decision: white noise turns each symbol on its own (its part is half
              the variance of neighbours' differences), slow phase noise turns neighbours alike; the excess and its
              spectrum say whether it is slow (a loop too narrow, an oscillator's close-in noise) or fast
  neighbours  every band over the floor (rpps.analysis.cyclic.occupied's runs) besides the strongest, and whether the
              spectrum drops to the floor between them: one that does not overlaps the signal
"""
import numpy as np


def _decide(y, pts):
    return pts[np.argmin(np.abs(y[:, None] - pts[None, :]), axis=1)]

def _normalize(y, pts):
    """y scaled so its decisions' points fit it (the gain measure fits)"""
    y = np.asarray(y, dtype=np.complex128)
    pts = np.asarray(pts, dtype=np.complex128)
    pts = pts/np.sqrt(np.mean(np.abs(pts)**2))
    y = y/np.sqrt(np.mean(np.abs(y)**2))
    for _ in range(3):
        d = _decide(y, pts)
        y = y/(np.sum((y*np.conj(d)).real)/np.sum(np.abs(d)**2))
    return y, pts

class Amplifier:
    """Result of `amplifier`: rings (radius of each ring's points), gain_db and phase_deg (each ring's mean symbol against
    its points, relative to the innermost ring), compression_db (the outermost ring's gain), am_pm (the outermost ring's
    turn, degrees)"""
    __slots__ = ("rings", "gain_db", "phase_deg", "compression_db", "am_pm")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        return (f"outer ring {self.compression_db:+.2f} dB and {self.am_pm:+.1f} deg against the inner "
                f"({len(self.rings)} rings)")

    __repr__ = __str__

def amplifier(y, points, tol=0.05):
    """AM/AM and AM/PM from locked symbols y and the constellation's points (rings: points within tol of a radius)"""
    y, pts = _normalize(y, points)
    r = np.abs(pts)
    rings = []
    for v in np.sort(r):
        if not rings or v - rings[-1] > tol:
            rings.append(v)
    d = _decide(y, pts)
    g, ph = [], []
    for R in rings:
        sel = np.abs(np.abs(d) - R) <= tol
        c = np.mean(y[sel]*np.conj(d[sel]))/np.mean(np.abs(d[sel])**2) # the ring's complex gain
        g.append(20*np.log10(abs(c)))
        ph.append(np.degrees(np.angle(c)))
    g = np.array(g) - g[0]
    ph = np.array(ph) - ph[0]
    return Amplifier(rings=np.array(rings), gain_db=g, phase_deg=ph, compression_db=float(g[-1]), am_pm=float(ph[-1]))

class IQ:
    """Result of `iq`: ratio (nu/mu: the conjugate's part), irr_db (image rejection, |mu/nu|^2 in dB), gain_db (I arm over Q),
    phase_deg (their angle off 90 degrees), dc (mean / rms), corrected (y with nu s* taken off and the DC)"""
    __slots__ = ("ratio", "irr_db", "gain_db", "phase_deg", "dc", "corrected")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        return (f"image rejection {self.irr_db:.1f} dB (I/Q gain {self.gain_db:+.2f} dB, phase {self.phase_deg:+.2f} deg), "
                f"DC {20*np.log10(max(abs(self.dc), 1e-12)):.1f} dB")

    __repr__ = __str__

def iq(y):
    """IQ imbalance of symbols (or samples) of a constellation with E[s^2] = 0, blind (see the module)"""
    y = np.asarray(y, dtype=np.complex128)
    dc = np.mean(y)
    z = y - dc
    rho = np.mean(z**2)/np.mean(np.abs(z)**2) # 2 mu nu / (|mu|^2 + |nu|^2) = 2 k / (1 + |k|^2), k = nu/mu*
    # solve 2k/(1 + |k|^2) = rho for the root with |k| < 1
    a = abs(rho)
    m = (1 - np.sqrt(max(1 - a**2, 0.0)))/a if a > 1e-12 else 0.0
    k = m*np.exp(1j*np.angle(rho)) if a > 1e-12 else 0j
    corrected = (z - k*np.conj(z))/(1 - abs(k)**2)
    # y = s + k s* is I scaled by (1 + Re k) - ..., Q by (1 - Re k), and Q turned by -Im k: small-imbalance reading
    gain = 20*np.log10((1 + k.real)/(1 - k.real)) if abs(k.real) < 1 else np.inf
    phase = np.degrees(2*k.imag)
    irr = -20*np.log10(abs(k)) if abs(k) > 0 else np.inf
    return IQ(ratio=complex(k), irr_db=float(irr), gain_db=float(gain), phase_deg=float(phase),
              dc=complex(dc/np.sqrt(np.mean(np.abs(y)**2))), corrected=corrected)

class PhaseNoise:
    """Result of `phase_noise`: rms_deg (of the phase error), white_deg (what the measured Es/N0 explains), excess_deg
    (the rest, rms), slow (the share of the phase error's power below a tenth of the symbol rate: much over 0.1 is phase
    noise slower than the symbols), spectrum (the phase error's, against frequency in symbol rates)"""
    __slots__ = ("rms_deg", "white_deg", "excess_deg", "slow", "spectrum", "freq")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        return (f"phase error {self.rms_deg:.2f} deg rms, noise explains {self.white_deg:.2f}: excess {self.excess_deg:.2f} "
                f"deg rms, {100*self.slow:.0f}% of it below a tenth of the symbol rate")

    __repr__ = __str__

def phase_noise(y, points, nfft=1024):
    """The phase of each symbol against its decision, against what white noise at the measured Es/N0 gives"""
    y, pts = _normalize(y, points)
    d = _decide(y, pts)
    th = np.angle(y*np.conj(d))
    # White noise turns each symbol independently; slow phase noise turns neighbours alike, so a difference of
    # neighbours has twice the white variance and little of the rest
    white = np.sqrt(np.var(np.diff(th))/2)
    rms = float(np.sqrt(np.mean(th**2)))
    nfft = int(min(nfft, 2**int(np.log2(max(len(th)//8, 16)))))
    seg = len(th)//nfft
    S = np.mean(np.abs(np.fft.rfft(th[:seg*nfft].reshape(seg, nfft)*np.hanning(nfft), axis=1))**2, axis=0)
    f = np.fft.rfftfreq(nfft)
    flat = np.median(S[f > 0.25]) # white part's level, from the fast half
    excess = np.maximum(S - flat, 0)
    slow = float(excess[f < 0.1].sum()/max(S.sum(), 1e-300))
    return PhaseNoise(rms_deg=float(np.degrees(rms)), white_deg=float(np.degrees(white)),
                      excess_deg=float(np.degrees(np.sqrt(max(rms**2 - white**2, 0.0)))), slow=slow, spectrum=S, freq=f)

class Neighbour:
    """A band over the floor besides the signal's: lo, hi (Hz), power (relative to the signal's), touching (the spectrum
    does not reach the floor between it and the signal)"""
    __slots__ = ("lo", "hi", "power", "touching")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        t = "overlapping or touching" if self.touching else "apart"
        return f"band {self.lo:+.0f} .. {self.hi:+.0f} Hz at {10*np.log10(max(self.power, 1e-30)):+.1f} dB, {t}"

    __repr__ = __str__

def neighbours(x, Fs, nfft=1024, margin_db=3.0):
    """Bands over the floor besides the strongest (the signal), from a Welch PSD: whether each reaches the floor between
    it and the signal (touching: a gap less than `margin_db` over the floor never opens)"""
    x = np.asarray(x, dtype=np.complex128)
    nfft = int(min(nfft, 2**int(np.log2(max(len(x)//8, 16)))))
    seg = len(x)//nfft
    S = np.fft.fftshift(np.mean(np.abs(np.fft.fft(x[:seg*nfft].reshape(seg, nfft)*np.hanning(nfft), axis=1))**2, axis=0))
    S = np.convolve(S, np.ones(5)/5, mode="same")
    f = (np.arange(nfft) - nfft//2)*Fs/nfft
    N0 = np.percentile(S, 10)
    on = S > 4*N0 # bands: 6 dB over the floor (a valley between two that touch stays over the floor)
    runs = []
    i = 0
    while i < nfft:
        if on[i]:
            j = i
            while j + 1 < nfft and on[j + 1]:
                j += 1
            if j - i + 1 >= 3:
                runs.append((i, j, float(np.sum(S[i:j + 1] - N0))))
            i = j + 1
        else:
            i += 1
    # Two bands that overlap make one run: split a run where it dips 6 dB under the peaks on both sides
    split = []
    for a, b, p in runs:
        while True:
            seg = S[a:b + 1]
            best = None
            for v in range(1, len(seg) - 1):
                left, right = seg[:v].max(), seg[v + 1:].max()
                if seg[v] < 0.25*min(left, right) and (best is None or seg[v]/min(left, right) < best[1]):
                    best = (v, seg[v]/min(left, right))
            if best is None:
                split.append((a, b, float(np.sum(S[a:b + 1] - N0))))
                break
            v = a + best[0]
            split.append((a, v, float(np.sum(S[a:v + 1] - N0))))
            a = v + 1
    # ... or where its level steps by 6 dB between two plateaus (overlapping bands of different power: no valley). Bands
    # of the same power overlapping cannot be told apart here (rpps.analysis.cyclic separates them by rate and carrier)
    runs = []
    wide = max(16, nfft//32) # a band is wider than this; so is each side of a split
    L = 10*np.log10(np.maximum(S, 1e-300))
    for a, b, p in split:
        pieces = [(a, b)]
        flat = lambda seg: np.percentile(seg, 75) - np.percentile(seg, 25) < 4.0 # a plateau, not a skirt
        while pieces:
            a_, b_ = pieces.pop()
            best = None
            for v in range(a_ + wide, b_ - wide + 1):
                lft, rgt = L[a_:v], L[v:b_ + 1]
                step = abs(np.median(lft) - np.median(rgt))
                if step >= 6 and (best is None or step > best[1]) and flat(lft[len(lft)//4:]) and flat(rgt[:3*len(rgt)//4]):
                    best = (v, step)
            if best is None:
                mid = L[a_ + (b_ - a_)//4:b_ - (b_ - a_)//4 + 1]
                if (a_, b_) == (a, b) or flat(mid): # a piece split off that is no plateau is a skirt between two
                    runs.append((a_, b_, float(np.sum(S[a_:b_ + 1] - N0))))
            else:
                pieces += [(a_, best[0] - 1), (best[0], b_)]
    runs = [r for r in runs if r[1] - r[0] + 1 >= wide] # pieces of a skirt the splits leave are no band
    if not runs:
        return []
    main = max(runs, key=lambda r: r[2])
    out = []
    for a, b, p in runs:
        if (a, b) == main[:2]:
            continue
        lo, hi = (b, main[0]) if b < main[0] else (main[1], a)
        gap = S[lo:hi + 1].min() if hi >= lo else S[min(a, main[0])]
        out.append(Neighbour(lo=float(f[a]), hi=float(f[b]), power=p/main[2],
                             touching=bool(gap > N0*10**(margin_db/10))))
    return out
