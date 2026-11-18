"""Phase modulation with a residual carrier (PCM/PM, telemetry and ranging): carrier, index, the data on the carrier, tones
and subcarriers

    p = rp.analysis.pm.analyze(x, Fs)        # x at baseband (a channel)
    print(p)          # e.g. "PM, residual carrier at -1206.8 Hz with 45% of the power (index 0.84 rad); data on the
                      #  carrier at 131072.0 Bd; subcarrier 72000.0 Hz (lines +-4800.0 Hz); tones 19000.0 Hz"
    s = rp.analysis.pm.subcarrier(p.phase, Fs, 72000.0, 20e3)   # the subcarrier at complex baseband: cyclic / detect it
    y = rp.analysis.pm.nrz(p.phase, Fs, p.rate, p.tau)          # soft symbols of the data on the carrier

A transmitter that phase modulates its carrier by less than 90 degrees leaves part of the power in the carrier
(cos^2 of the index, for data of +-index), so the receiver can lock to it with a narrow loop (CCSDS 401: PCM/PM/NRZ,
PCM/PM/biphase, PCM/PSK/PM on a subcarrier, ranging tones). Looking for a PSK it seems an unbalanced constellation with a
spectral line in x itself, near-constant envelope, rates no PSK fits. The phase around the carrier, phi(t), is a real
baseband signal holding everything else:
  data on the carrier   NRZ: phi^2 has a line at the rate (phi itself, none). biphase (Manchester): a line in phi at the rate
  a subcarrier          a band around f_sc in phi, and (BPSK on it, squared) a line of phi^2 at 2 f_sc: a rate search on
                        phi^2 or |x|^2 finds 2 f_sc and calls it the symbol rate. Sending idle (alternating) bits it is
                        two lines of phi at f_sc +- rate/2. `subcarrier` takes it to complex baseband for its own rate
  tones                 single lines in phi (ranging, a subcarrier with nothing on it)
"""
import numpy as np

from . import cyclic
from .cyclic import comb_rate

class PM:
    """Result of `analyze`. carrier: offset (Hz) of the residual carrier, None if there is none; share: the carrier's part
    of the power; index: the modulation index (rad) that leaves that share for +-index data, arccos(sqrt(share));
    phase_rms (rad); rate: of the data on the carrier (Hz), None without a line; biphase: the rate's line is in phi itself;
    tau: the first symbol boundary (samples) of the data on the carrier; tones: lines of phi that are no pair (Hz);
    subcarriers: [(f_sc, half spacing of its line pair)]; phase: phi(t) (rad, a sample each input sample); lines: phi's"""
    __slots__ = ("carrier", "share", "index", "phase_rms", "rate", "biphase", "tau", "tones", "subcarriers", "phase",
                 "lines", "why")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        if self.carrier is None:
            return f"not PM with a residual carrier: {self.why}"
        out = [f"PM, residual carrier at {self.carrier:+.1f} Hz with {100*self.share:.0f}% of the power (index "
               f"{self.index:.2f} rad, phase rms {self.phase_rms:.2f} rad)"]
        if self.rate:
            out.append(f"data on the carrier at {self.rate:.1f} Bd ({'biphase' if self.biphase else 'NRZ'})")
        for fsc, half in self.subcarriers:
            out.append(f"subcarrier {fsc:.1f} Hz" + (f" (lines +-{half:.1f} Hz: idle at {2*half:.1f} Bd?)" if half else ""))
        if self.tones:
            out.append("tones " + ", ".join(f"{t:.1f} Hz" for t in self.tones))
        return "; ".join(out)

    __repr__ = __str__

BIMODAL = 2.5 # mean |soft symbol| over its scatter for data on the carrier (Es/N0 ~8 dB and up)
PATTERN = 0.8 # sign agreement at some lag up to 64 symbols over this: a pattern of tones, not data
MIN_SHARE = 0.05 # A carrier line holding less of the power than this is a pilot or a leak, not a PM carrier

def _smooth(z, L):
    """z through a triangular window of about L samples (two running means: O(n), where a convolution is O(n L))"""
    h = max(1, L//2)
    for _ in range(2):
        c = np.concatenate(([0], np.cumsum(z)))
        a = np.clip(np.arange(len(z)) - h//2, 0, len(z))
        b = np.clip(a + h, 0, len(z))
        z = (c[b] - c[a])/np.maximum(b - a, 1)
    return z

def carrier_phase(x, Fs, fc, loop=200.0):
    """(phi, z, ref): phi(t) the phase of x around a carrier at fc (Hz); z, x with fc taken off; ref, the carrier as the
    loop sees it, z smoothed over about 1/loop seconds (a narrow loop: it follows drift and phase noise slower than that,
    never the data)"""
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(len(x))
    z = x*np.exp(-2j*np.pi*fc/Fs*n)
    ref = _smooth(z, max(3, int(Fs/loop)))
    return np.angle(z*np.conj(ref)), z, ref

def share_of(z, ref):
    """The carrier's part of the power: the tracked carrier's power over all of it (a carrier that drifts or wanders over
    the capture keeps its share; |mean z|^2 would average it away)"""
    return float(np.mean(np.abs(ref)**2)/max(np.mean(np.abs(z)**2), 1e-300))

def analyze(x, Fs, pfa=1e-6, loop=200.0, lo=None, candidates=4, max_samples=2**22):
    """Residual carrier, index, data rate on the carrier, tones and subcarriers of baseband samples x (see the module)"""
    x = np.asarray(x, dtype=np.complex128)[:max_samples]
    if len(x) < 4096:
        return PM(why="too few samples")
    G, _, _ = cyclic.occupied(x, Fs)
    x = np.fft.ifft(np.fft.fft(x)*G)
    # Carrier candidates from a short span (a carrier drifting some Hz over the capture stays in a bin of it; over all of
    # it, it would spread over many fine bins under a steady spur), each judged by the share it holds when tracked
    span = int(min(len(x), max(4096, 2**int(np.log2(4*Fs/loop)))))
    xs = x[:span]
    ls = [l for l in cyclic.lines(xs/np.sqrt(np.mean(np.abs(xs)**2)), Fs, -Fs/2, Fs/2, pfa, top=candidates) if l.significant]
    if not ls:
        return PM(why="no carrier line")
    best = None
    for l in ls:
        phi_, z_, ref_ = carrier_phase(x, Fs, l.freq, loop)
        sh = share_of(z_, ref_)
        if best is None or sh > best[0]:
            best = (sh, l.freq, phi_, z_)
    share, fc, phi, z = best
    if share < MIN_SHARE:
        return PM(why=f"the line at {fc:.1f} Hz holds {100*share:.1f}% of the power: a pilot or a leak")
    lo = 2*loop if lo is None else lo
    lp = [l for l in cyclic.lines(phi, Fs, lo, Fs/2, pfa, top=12) if l.significant]
    l2 = [l for l in cyclic.lines(phi**2, Fs, lo, Fs/2, pfa, top=12) if l.significant]
    tol = lambda f: 2e-3*f + 2*Fs/len(phi)
    # phi's spectrum in coarse bins, against its floor: where a band (a subcarrier with data on it) is
    nb = 256
    seg = len(phi)//nb
    S = np.mean(np.abs(np.fft.rfft(phi[:seg*nb].reshape(seg, nb)*np.hanning(nb), axis=1))**2, axis=0)
    fS = np.fft.rfftfreq(nb, 1/Fs)
    floor = np.median(S) # phi's own floor: most of its bins are noise (below it the band mask's -20 dB, no floor at all)
    S = np.convolve(S, np.ones(3)/3, mode="same")
    # NRZ data on the carrier puts phi's power at DC (a sinc^2), and its rate's harmonics in phi^2 (and, the band edges
    # ringing alike at every transition, in phi): first, so they are not taken for subcarriers or tones
    rate, biphase, tau = None, False, None
    if l2 and S[1] > 0.1*S[1:].max(): # DC within 10 dB of phi's strongest part (a subcarrier's band is far over it)
        rate, _ = comb_rate(l2, Fs, lo)
        biphase = S[1] < S[int(np.argmin(np.abs(fS - rate)))] # biphase moves the power from DC to the rate
        n = np.arange(len(phi))
        g = np.diff(phi)**2 # transitions pulse at the boundaries
        c = np.sum(g*np.exp(-2j*np.pi*rate/Fs*n[1:]))
        tau = float((-np.angle(c)/(2*np.pi)*Fs/rate) % (Fs/rate) + 1.0)
        # Data really on the carrier demodulates to two clusters (a mean |y| several times its scatter); lines of a rate
        # alone (tones, products of subcarriers near DC) do not
        y = biphase_symbols(phi, Fs, rate, tau) if biphase else nrz(phi, Fs, rate, tau)
        a = np.abs(y)
        sg = np.sign(y)
        # ... and data (scrambled or not) changes from symbol to symbol unpredictably, where tones repeat a pattern
        periodic = len(sg) >= 256 and max(abs(np.mean(sg[k:]*sg[:-k])) for k in range(1, 65)) > PATTERN
        if len(a) < 16 or np.mean(a) < BIMODAL*np.std(a) or periodic:
            rate, biphase, tau = None, False, None
    harmonic = lambda f: rate is not None and abs(f/rate - round(f/rate)) < tol(f)/rate and round(f/rate) >= 1
    l2 = [l for l in l2 if not harmonic(l.freq)]
    lp = [l for l in lp if not harmonic(l.freq)]
    def peak(f):
        """The coarse bin of the band f is in, if f is at a peak of phi's spectrum well over its floor (else None)"""
        k = int(np.argmin(np.abs(fS - f)))
        if k < 1 or k >= len(S) - 1 or S[k] < 8*floor:
            return None
        return k if S[k] >= S[k - 1] and S[k] >= S[k + 1] else None
    # Subcarriers: a BPSK subcarrier at f squared is a line at 2 f, at the peak of a band of phi at f; a band, one
    subs, bands = [], set()
    for l in sorted(l2, key=lambda l: -l.strength):
        f = l.freq/2
        k = peak(f) if f > lo else None
        if k is not None and not ({k - 1, k, k + 1} & bands):
            subs.append((f, 0.0))
            bands.add(k)
    # ... or, sending idle (alternating) bits, two lines of phi at f +- rate/2
    pool = sorted(lp, key=lambda l: -l.strength)
    used = set()
    for i, a_ in enumerate(pool):
        for j, b_ in enumerate(pool):
            if j <= i or i in used or j in used or abs(a_.db - b_.db) > 6:
                continue
            fsc, half = (a_.freq + b_.freq)/2, abs(a_.freq - b_.freq)/2
            if half < lo or half > fsc/2 or (peak(fsc) is None and not any(abs(fsc - x) < tol(fsc) for x, _ in subs)):
                continue
            k = next((m for m, (x, _) in enumerate(subs) if abs(fsc - x) < tol(fsc)), None)
            if k is None:
                subs.append((fsc, half))
            elif subs[k][1] == 0.0:
                subs[k] = (subs[k][0], half)
            used.update((i, j))
    # Tones: phi's other lines (not a subcarrier's sideband or centre)
    tones = []
    for i, l in enumerate(pool):
        if i in used or any(abs(l.freq - fsc) < max(half, 0.0) + tol(fsc) for fsc, half in subs):
            continue
        tones.append(l.freq)
    subs = sorted(subs)
    return PM(carrier=fc, share=share, index=float(np.arccos(np.sqrt(min(share, 1.0)))), phase_rms=float(np.std(phi)),
              rate=rate, biphase=biphase, tau=tau, tones=sorted(tones), subcarriers=sorted(subs), phase=phi, lines=lp)

def subcarrier(phi, Fs, fsc, bw):
    """A subcarrier at fsc (Hz) of the phase phi at complex baseband, bw (Hz) wide: rpps.analysis.cyclic / detect it"""
    n = np.arange(len(phi))
    z = 2*np.asarray(phi, dtype=np.float64)*np.exp(-2j*np.pi*fsc/Fs*n)
    Z = np.fft.fft(z)
    f = np.abs(np.fft.fftfreq(len(z), 1/Fs))
    # A raised-cosine edge over a fifth of the band: a brick wall would ring, and its edges make lines of their own
    edge = np.clip((f - 0.4*bw)/(0.2*bw), 0, 1)
    return np.fft.ifft(Z*0.5*(1 + np.cos(np.pi*edge)))

def nrz(phi, Fs, rate, tau):
    """Soft symbols of NRZ data on the carrier: phi integrated over each bit (the matched filter of a rectangular pulse),
    from the boundary tau (samples)"""
    sps = Fs/rate
    c = np.concatenate(([0.0], np.cumsum(phi)))
    b = tau + sps*np.arange(int((len(phi) - tau)//sps))
    i = np.round(b).astype(np.int64)
    j = np.round(b + sps).astype(np.int64)
    ok = j <= len(phi)
    return (c[j[ok]] - c[i[ok]])/sps

def biphase_symbols(phi, Fs, rate, tau):
    """Soft symbols of biphase (Manchester) data on the carrier: the first half-bit's integral less the second's"""
    sps = Fs/rate
    c = np.concatenate(([0.0], np.cumsum(phi)))
    b = tau + sps*np.arange(int((len(phi) - tau)//sps))
    i, m, j = (np.round(b + f*sps).astype(np.int64) for f in (0.0, 0.5, 1.0))
    ok = j <= len(phi)
    return ((c[m[ok]] - c[i[ok]]) - (c[j[ok]] - c[m[ok]]))/sps
