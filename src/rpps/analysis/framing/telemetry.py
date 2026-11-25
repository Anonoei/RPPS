"""Residual-carrier PM telemetry (PCM/PSK/PM): the carrier, each subcarrier's data, its line code and its frames

    t = rp.analysis.framing.telemetry.analyze(x, Fs)      # None when x has no residual carrier
    print(t)                                      # the carrier, then a line per subcarrier
    t.subcarriers[0].bits                         # data bits (line code removed), t.subcarriers[0].frame

Spacecraft telemetry and command links phase-modulate a carrier by less than 90 degrees, so part of the power stays in a
carrier line (rpps.analysis.modulation.pm finds it, the index, and the phase phi(t) around it). The data rides in phi, on the carrier
itself or on subcarriers: a subcarrier is BPSK-modulated (a square or sine wave whose sign is the data), at a rate set
apart from the others. Per subcarrier:
  1. its rate: the cyclic line at baseband (rpps.analysis.spectrum.cyclic), twice the spacing of its line pair (idle 1010 shows
     as lines at +-rate/2), and twice those (Manchester chips): the one whose integrate-and-dump symbols split cleanest,
     of those whose symbols don't come in equal pairs (twice the real rate)
  2. BPSK symbols: the subcarrier's phase from its square (a squared-carrier reference smoothed over 20 ms), integrated over
     each symbol from the boundary that splits them best
  3. the line code: Manchester (biphase-L: every pair of chips opposite) halves the rate; differential (NRZ-M/S) shows as
     frames that only line up after XOR with the previous bit
  4. idle (alternating or constant bits), else frames: the period of the bits' autocorrelation, the fixed bits within
     it, and counters (a field that steps by one every frame)
"""
import numpy as np

from ..modulation import pm as _pm

SMOOTH = 0.02    # s: the squared reference's smoothing
SPLIT = 2.5      # |mean|/std of |soft symbol| for symbols that split into two levels (noise alone gives about 1.3)
MANCHESTER = 0.9 # Fraction of chip pairs opposite for Manchester
FRAME_R = 0.1    # Autocorrelation peak for a frame period
MIN_FIXED = 0.05 # Fraction of a frame's bits fixed (beyond chance) for a frame
MIN_FRAME = 64
ABOVE = 4.0      # symmetry(): bins this many times the median count as signal
SYMMETRIC = 0.4  # symmetry(): sidebands that mirror about the carrier, as phase modulation's do
SIDEBANDS = 0.1  # symmetry(): their power to the carrier's: data in the phase, not a tone in noise

class TelemetryFrame:
    """period (bits), fixed (bits constant over the frames), frames (how many folded), counters [(bit, width, step)]"""
    __slots__ = ("period", "r", "fixed", "frames", "counters")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        c = "".join(f", a {w}-bit counter at bit {b} ({'+' if s > 0 else '-'}1 a frame)" for b, w, s in self.counters)
        return (f"{self.period}-bit frames (autocorrelation {self.r:.2f}), {self.fixed} bits fixed over {self.frames} "
                f"frames{c}")

class Subcarrier:
    """One subcarrier (or the data on the carrier itself: freq 0): freq, baud (symbol rate), quality (|mean|/std of the
    soft symbols' size), esn0 (dB), manchester, differential, rate (data bits/s), idle (None, or what pattern),
    bits (data), frame (TelemetryFrame or None), why (when no symbols split)"""
    __slots__ = ("freq", "baud", "quality", "esn0", "manchester", "differential", "rate", "idle", "bits", "frame", "why")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def ok(self):
        return self.bits is not None

    def head(self):
        """Where, the modulation and line code, without the bits' content"""
        where = f"subcarrier {self.freq/1e3:.3f} kHz" if self.freq else "on the carrier"
        if not self.ok:
            return where
        code = ", ".join(c for c, on in (("Manchester", self.manchester), ("differential", self.differential)) if on) or "NRZ-L"
        return f"{where}: BPSK {self.baud:.6g} Bd, Es/N0 {self.esn0:.1f} dB; {code}: {self.rate:.6g} bit/s"

    def __str__(self):
        if not self.ok:
            return f"{self.head()}: {self.why}"
        if self.idle:
            return self.head() + f"; idle ({self.idle})"
        return self.head() + (f"; {self.frame}" if self.frame else f"; {len(self.bits)} bits, no frame period in them")

class Telemetry:
    """pm: the rpps.analysis.modulation.pm result; subcarriers: [Subcarrier]"""
    __slots__ = ("pm", "subcarriers")
    def __init__(self, pm, subcarriers):
        self.pm, self.subcarriers = pm, subcarriers

    def __str__(self):
        return "\n".join([str(self.pm)] + ["  " + str(s) for s in self.subcarriers])

    __repr__ = __str__

def bpsk(z, fs, baud):
    """(soft symbols, quality, Es/N0 dB) of BPSK at complex baseband z: phase from z^2, integrate and dump at the boundary
    whose symbols split cleanest"""
    L = max(3, int(SMOOTH*fs)) | 1
    ref = np.convolve(z**2, np.hanning(L), mode="same")
    y = (z*np.exp(-0.5j*np.unwrap(np.angle(ref)))).real # Unwrapped: half a wrapped angle jumps by pi
    c = np.concatenate(([0.0], np.cumsum(y)))
    sps = fs/baud
    if sps < 2 or len(y) < 16*sps:
        return None, 0.0, None
    best = (None, -1.0)
    for tau in np.arange(0, sps, sps/32):
        b = tau + sps*np.arange(int((len(y) - tau)//sps) - 1)
        v = c[np.round(b + sps).astype(np.int64)] - c[np.round(b).astype(np.int64)]
        a = np.abs(v)
        q = float(a.mean()/(a.std() + 1e-12))
        if q > best[1]:
            best = (v, q)
    v, q = best
    a = np.abs(v)
    esn0 = float(10*np.log10(a.mean()**2/(2*a.var() + 1e-30)))
    return v, q, esn0

def doubled(chips, hold=0.9):
    """Whether chips come in equal pairs (at one alignment): symbols at twice the real rate"""
    c = np.asarray(chips)
    for off in (0, 1):
        n = (len(c) - off)//2
        if n >= 16 and np.mean(c[off:off + 2*n:2] == c[off + 1:off + 2*n:2]) >= hold:
            return True
    return False

def manchester(bits):
    """(fraction of chip pairs opposite, alignment, data bits (the first chip of each pair)) at the better alignment"""
    out = []
    for off in (0, 1):
        n = (len(bits) - off)//2
        p = bits[off:off + 2*n].reshape(n, 2)
        out.append((float(np.mean(p[:, 0] != p[:, 1])), off, p[:, 0].copy()))
    return max(out, key=lambda t: t[0])

def idle(bits):
    """What idle pattern bits are (alternating, constant), or None"""
    if len(bits) < 32:
        return None
    t = float(np.mean(bits[1:] != bits[:-1]))
    if t > 0.95:
        return "alternating 1010"
    if t < 0.05:
        return "constant"
    return None

def _fixed(bits, L, frames=None):
    """Fraction of the bits of L-bit frames that hold one value over every frame (the first `frames` of them: compare
    periods over as many frames, as fewer leave more fixed by chance); 0 with fewer than 3 frames"""
    F = np.asarray(bits[:len(bits)//L*L]).reshape(-1, L)[:frames]
    if len(F) < 3:
        return 0.0
    return float(np.mean(np.abs(F.mean(axis=0) - 0.5) >= 0.49))

def period(bits, min_lag=MIN_FRAME, top=12):
    """(lag, |r|, fixed fraction) of the frame period: of the autocorrelation peaks past min_lag standing out of the noise
    (and above FRAME_R) and their divisors, the smallest whose fold leaves nearly the most bits fixed beyond chance (byte
    or word structure correlates at short lags but fixes nothing; a frame's sync and header do). (None, 0, 0) when none
    fixes MIN_FIXED of its bits"""
    s = np.asarray(bits, dtype=np.float64)*2 - 1
    s = s - s.mean()
    n = len(s)
    if n < 3*min_lag:
        return None, 0.0, 0.0
    S = np.fft.fft(s, 2*n)
    r = np.fft.ifft(np.abs(S)**2).real[:n]
    r = r/(n - np.arange(n))/max(r[0]/n, 1e-12)
    lags = np.arange(min_lag, n//3 + 1)
    if not len(lags):
        return None, 0.0, 0.0
    cands = []
    for L in lags[np.argsort(np.abs(r[lags]))[::-1][:top]]:
        if abs(r[L]) < max(FRAME_R, 6/np.sqrt(n - L)):
            break
        cands.append(int(L))
    # Divisors of the peaks too: the frame itself may correlate less than its multiples
    cands = sorted({L//d for L in cands for d in range(1, L//min_lag + 1) if L % d == 0 and L//d >= min_lag})
    if not cands:
        return None, 0.0, 0.0
    # Bits fixed beyond chance: over F frames a bit with ones-density p holds one value by chance p^F + (1-p)^F of the
    # time, so fewer frames (a multiple of the frame) fix more by chance; a multiple of the frame fixes no more beyond it.
    # The smallest period fixing nearly the most beyond chance is the frame
    p1 = float(np.mean(bits))
    cands = [L for L in cands if n//L >= 3]
    ex = {}
    for L in cands:
        F = n//L
        c = p1**F + (1 - p1)**F
        ex[L] = (_fixed(bits, L) - c)/(1 - c) if c < 1 else 0.0
    top_x = max(ex.values(), default=0.0)
    if top_x < MIN_FIXED:
        return None, 0.0, 0.0
    L = min(L for L in cands if ex[L] >= 0.9*top_x)
    return L, float(abs(r[L])), _fixed(bits, L)

def counters(F, widths=(16, 8), hold=0.9):
    """[(bit, width, step)] of fields in the folded frames F that step by +-1 a frame (in `hold` of the frames: a bit
    error breaks a step or two), the widest first"""
    found = []
    if len(F) < 4:
        return found
    P = F.shape[1]
    for W in widths:
        pw = 1 << np.arange(W - 1, -1, -1)
        for o in range(P - W + 1):
            if any(o < b + w and b < o + W for b, w, _ in found):
                continue
            v = F[:, o:o + W] @ pw
            d = np.diff(v) % (1 << W)
            if np.mean(d == 1) >= hold:
                found.append((o, W, 1))
            elif np.mean(d == (1 << W) - 1) >= hold:
                found.append((o, W, -1))
    return found

def frame(bits):
    """TelemetryFrame of the bits, or None when no period shows"""
    L, r, f = period(bits)
    if L is None:
        return None
    F = np.asarray(bits[:len(bits)//L*L], dtype=np.int64).reshape(-1, L)
    return TelemetryFrame(period=L, r=r, fixed=int(round(f*L)), frames=len(F), counters=counters(F))

def line_code(chips):
    """(manchester, differential, data bits) of hard chips: Manchester when nearly every pair is opposite; differential
    when the XOR of neighbours frames with more bits fixed than the bits themselves"""
    frac, _, data = manchester(chips)
    man = frac >= MANCHESTER
    bits = data if man else chips
    diff = bits[1:] ^ bits[:-1]
    _, _, f0 = period(bits)
    _, _, f1 = period(diff)
    if f1 > f0:
        return man, True, diff
    return man, False, bits

def subcarrier(phi, Fs, fsc, half=None, rates=None):
    """Subcarrier at fsc (Hz) of the phase phi: its rate, symbols, line code, and frames (see the module)"""
    from ..spectrum.cyclic import features
    bw0 = max(4*half, 12e3) if half else 16e3
    z0 = _pm.subcarrier(phi, Fs, fsc, bw0)
    D = max(1, int(Fs/(4*bw0)))
    cands = set(rates or [])
    if half:
        cands.add(2*half)
    try:
        f = features(z0[::D][:2**19], Fs/D)
        if f.baud:
            cands.update((f.baud, 2*f.baud))
    except Exception:
        pass
    best = None
    for baud in sorted(c for c in cands if c and c < fsc):
        bw = 4*baud
        D = max(1, int(Fs/(4*bw)))
        z = _pm.subcarrier(phi, Fs, fsc, bw)[::D]
        v, q, esn0 = bpsk(z, Fs/D, baud)
        if v is None or doubled(v > 0):
            continue
        if best is None or q > best[1]:
            best = (baud, q, esn0, v)
    if best is None or best[1] < SPLIT:
        return Subcarrier(freq=fsc, baud=best[0] if best else None, quality=best[1] if best else 0.0,
                          why="no rate whose BPSK symbols split into two levels" +
                              (f" (best {best[0]:.6g} Bd, {best[1]:.2f})" if best else ""))
    baud, q, esn0, v = best
    chips = (v > 0).astype(np.uint8)
    idl = idle(chips)
    if idl:
        return Subcarrier(freq=fsc, baud=baud, quality=q, esn0=esn0, manchester=False, differential=False, rate=baud,
                          idle=idl, bits=chips, frame=None)
    man, diff, bits = line_code(chips)
    rate = baud/2 if man else baud
    idl = idle(bits)
    return Subcarrier(freq=fsc, baud=baud, quality=q, esn0=esn0, manchester=man, differential=diff, rate=rate, idle=idl,
                      bits=bits, frame=None if idl else frame(bits))

def symmetry(x, Fs, carrier, nfft=4096, band=None, snr_db=None):
    """(symmetry, sidebands): how well the spectrum mirrors about the carrier, 0..1 (sum(min)/sum(max) of the power
    standing clear of the noise, ABOVE times its level per bin, at carrier +-f), and that power's ratio to the carrier's.
    Phase modulation by a real phase whose sign flips as often as not (data) has a real autocorrelation, so a spectrum
    symmetric about its carrier, with sidebands that hold power; a tone beside another signal mirrors badly, a tone in
    noise has no sidebands. A Hann window keeps a strong carrier's own leakage out of them.
    band (Hz, centred) and snr_db (the signal's SNR in it): x was filtered to that band, so the noise is band-limited, not
    the median bin (which sits in the filter's stopband: in-band noise alone would stand clear of it, and mirror). Its
    level is then the band's mean over 1 + SNR"""
    x = np.asarray(x)
    k = len(x)//nfft
    if k < 1:
        return 0.0, 0.0
    w = np.hanning(nfft).astype(np.float32)
    P = np.fft.fftshift(np.mean(np.abs(np.fft.fft(x[:k*nfft].reshape(k, nfft)*w, axis=1))**2, axis=0))
    if band and snr_db is not None:
        f = (np.arange(nfft) - nfft//2)*Fs/nfft
        nf = float(np.mean(P[np.abs(f) < band/2]))/(1 + 10**(snr_db/10))
    else:
        nf = np.median(P)
    P = np.where(P > ABOVE*nf, P - nf, 0.0) # Only what stands clear of the noise: noise bins alone mirror at random
    c = nfft//2 + int(round(carrier*nfft/Fs))
    m = min(c, nfft - 1 - c)
    if m < 8:
        return 0.0, 0.0
    lo, hi = P[c - m:c - 3][::-1], P[c + 4:c + m + 1] # Three bins either side left out: the carrier's main lobe
    n = min(len(lo), len(hi))
    lo, hi = lo[:n], hi[:n]
    side = float(np.sum(lo) + np.sum(hi))
    sym = float(np.sum(np.minimum(lo, hi))/max(np.sum(np.maximum(lo, hi)), 1e-30))
    return sym, side/max(float(np.sum(P[c - 3:c + 4])), 1e-30)

def analyze(x, Fs, **kw):
    """Telemetry of x (complex baseband, Fs), or None when rpps.analysis.modulation.pm finds no residual carrier"""
    p = _pm.analyze(x, Fs, **kw)
    if p.carrier is None:
        return None
    phi = p.phase if p.phase is not None and len(p.phase) == len(x) else _pm.carrier_phase(x, Fs, p.carrier)[0]
    subs = []
    if p.rate:
        tau = p.tau or 0.0
        v = _pm.nrz(phi, Fs, p.rate, tau)
        a = np.abs(v)
        q = float(a.mean()/(a.std() + 1e-12))
        if q >= SPLIT:
            chips = (v > 0).astype(np.uint8)
            man, diff, bits = line_code(chips)
            idl = idle(bits)
            subs.append(Subcarrier(freq=0.0, baud=p.rate, quality=q, esn0=float(10*np.log10(a.mean()**2/(2*a.var()))),
                                   manchester=man, differential=diff, rate=p.rate/2 if man else p.rate, idle=idl,
                                   bits=bits, frame=None if idl else frame(bits)))
    for fsc, half in p.subcarriers:
        if fsc < Fs/2:
            subs.append(subcarrier(phi, Fs, fsc, half))
    return Telemetry(p, subs)
