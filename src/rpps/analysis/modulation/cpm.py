"""GMSK and SOQPSK: binary continuous-phase modulation demodulated by a sequence detector, blind to timing and carrier

    r = rp.analysis.modulation.cpm.demodulate(x, Fs, "gmsk_0.3")          # or "soqpsk_tg", "soqpsk_mil", "msk", "gmsk_0.5" ...
    r.bits, r.baud, r.cfo, r.match                              # bits, symbol rate, carrier offset (Hz), fit 0..1
    best = rp.analysis.modulation.cpm.identify(x, Fs)                      # [(match, scheme)], best first: which pulse it is
    y = rp.analysis.modulation.cpm.modulate(bits, "soqpsk_tg", sps=8)     # the reference transmitter (tests, regeneration)

A CPM signal is exp(j theta(t)), theta(t) = 2 pi h sum_k alpha_k q(t - kT), with q the integral of the frequency pulse g
(q(LT) = 1/2). Here h = 1/2 throughout (MSK, GMSK, SOQPSK), and the symbols alpha_k come from the bits:
  msk / gmsk   alpha_k = 2 b_k - 1; g rectangular over 1 symbol (MSK) or Gaussian (BT 0.25 / 0.3 / 0.5, L 4 / 4 / 3)
  gmsk_diff    GSM's differential precoding first: alpha_k = 1 - 2 (b_k xor b_{k-1}) (3GPP TS 45.004)
  soqpsk_mil   ternary alpha_k = (-1)^(k+1) a_{k-1} (a_k - a_{k-2}) / 2, a = 2b - 1 (MIL-STD-188-181), g rectangular
  soqpsk_tg    the same precoder; g the windowed spectral raised cosine of IRIG 106 (rho 0.7, B 1.25, T1 1.5, T2 0.5:
               8 symbols long)
SOQPSK's precoder gives the same symbols for the bits, their complement, and (with the other parity) the bits with every
other one inverted: the signal can't tell these four apart, so SOQPSK bits come out up to that (a frame word settles it).

The detector is a Viterbi search with per-survivor processing (Raheli, Polydoros & Tzou 1995): a trellis state is the
last S bits; each survivor carries the older symbols of the pulse's tail, its settled phase, and its own second-order
carrier loop (the error is the angle of its branch correlation), so a carrier offset left by the coarse estimate, a
pulse longer than S symbols (SOQPSK-TG's 8) and the phase all need no states of their own. Branch metric: Re sum
r(t) conj(s(t)) over the symbol at `ns` samples a symbol. `match` is the best path's metric over sqrt(energy of r x
energy of s): 1 for a clean signal of exactly this scheme; a wrong pulse, rate or timing scores lower, which is how
`identify` tells the schemes apart.

Front end (what `demodulate` does before the detector):
  lines    for h = 1/2, x^2 has lines at 2 fc +- baud/2 (MSK's two tones squared; weaker under a Gaussian or TG pulse):
           the strongest pair gives the symbol rate (their spacing) and the carrier (their middle, halved)
  rate     without the lines, rpps.analysis.modulation.fsk.analyze's (or `baud` given: then the pair is looked for baud apart)
  carrier  without the lines, the median instantaneous frequency
  timing   the detector run at every 1/ns of a symbol on a stretch; the best metric wins. On a
           long record, stretches across it are timed too and a line through them corrects the symbol clock (ppm)
Limits: h = 1/2 binary CPM only; the coarse carrier must leave under about baud/50 for the loop (`loop`) to pull in;
the bits of a spectrally inverted signal come out inverted.
"""
import numpy as np

from ... import _accel

# name: (pulse, parameter, L, precoder)
SCHEMES = {
    "msk": ("rect", None, 1, "none"),
    "gmsk_0.25": ("gaussian", 0.25, 4, "none"),
    "gmsk_0.3": ("gaussian", 0.3, 4, "none"),
    "gmsk_0.5": ("gaussian", 0.5, 3, "none"),
    "gmsk_diff_0.3": ("gaussian", 0.3, 4, "diff"),
    "soqpsk_mil": ("rect", None, 1, "soqpsk"),
    "soqpsk_tg": ("tg", None, 8, "soqpsk"),
}
H = 0.5

def frequency_pulse(name, sps):
    """The frequency pulse g of a scheme at sps samples a symbol, L*sps samples, sum(g)/sps = 1/2 (q(LT) = 1/2)"""
    kind, p, L, _ = SCHEMES[name]
    n = L*sps
    t = (np.arange(n) + 0.5)/sps - L/2 # symbol times, centred on the pulse
    if kind == "rect":
        g = np.ones(n)
    elif kind == "gaussian":
        # A rectangle of one symbol through a Gaussian filter of bandwidth-time BT
        s = np.sqrt(np.log(2))/(2*np.pi*p)
        from math import erf
        e = np.vectorize(erf)
        g = 0.5*(e((t + 0.5)/(np.sqrt(2)*s)) - e((t - 0.5)/(np.sqrt(2)*s)))
    elif kind == "tg":
        rho, B, T1, T2 = 0.7, 1.25, 1.5, 0.5
        u = t/2 # t / 2Ts
        x = rho*B*u
        den = 1 - 4*x**2
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(np.abs(den) < 1e-9, np.pi/4, np.cos(np.pi*x)/np.where(den == 0, 1, den))
            b = np.where(u == 0, 1.0, np.sin(np.pi*B*u)/(np.pi*B*np.where(u == 0, 1, u)))
        au = np.abs(u)
        w = np.where(au < T1, 1.0, np.where(au <= T1 + T2, 0.5 + 0.5*np.cos(np.pi/T2*(au - T1)), 0.0))
        g = a*b*w
    else:
        raise ValueError(kind)
    return g/g.sum()*sps/2

def phase_table(name, ns):
    """Q (L, ns): 2 pi h q((j + n/ns) T) at sample n of a symbol for the symbol j symbols back (q from g at ns samples:
    q at the sample's end)"""
    g = frequency_pulse(name, ns)
    q = np.cumsum(g)/ns # q at the end of each sample, q[-1] = 1/2
    L = SCHEMES[name][2]
    return (2*np.pi*H*q).reshape(L, ns)

def alpha_table(name):
    """atab (2, 8) int8: alpha_k for (k's parity, b_k + 2 b_{k-1} + 4 b_{k-2})"""
    pre = SCHEMES[name][3]
    t = np.zeros((2, 8), np.int8)
    for par in range(2):
        for i in range(8):
            b0, b1, b2 = i & 1, i >> 1 & 1, i >> 2 & 1
            if pre == "none":
                a = 2*b0 - 1
            elif pre == "diff":
                a = 1 - 2*(b0 ^ b1)
            else: # soqpsk: (-1)^(k+1) a_{k-1} (a_k - a_{k-2}) / 2
                a0, a1, a2 = 2*b0 - 1, 2*b1 - 1, 2*b2 - 1
                a = (-1)**(par + 1)*a1*(a0 - a2)//2
            t[par, i] = a
    return t

def alphas(bits, name, parity=0):
    """The CPM symbols of bits (the bits before the first are 0)"""
    b = np.concatenate(([0, 0], np.asarray(bits, dtype=np.int64) & 1))
    idx = b[2:] + 2*b[1:-1] + 4*b[:-2]
    par = (np.arange(len(idx)) + parity) & 1
    return alpha_table(name)[par, idx].astype(np.float64)

def modulate(bits, name, sps=8, parity=0):
    """Complex baseband of bits at sps samples a symbol (unit amplitude, phase 0 at the start)"""
    a = alphas(bits, name, parity)
    g = frequency_pulse(name, sps)
    f = np.zeros(len(a)*sps + len(g))
    f[:len(a)*sps:sps] = a
    f = np.convolve(f, g)[:len(a)*sps] # frequency, in symbols' units (each symbol integrates to alpha/2)
    theta = 2*np.pi*H*np.cumsum(f)/sps
    return np.exp(1j*theta)

def _mlse_python(r, Q, atab, S, parity, loop):
    """Reference per-survivor Viterbi (see the module); the C++ kernel `cpm_psp` is the same loop"""
    L, ns = Q.shape
    n = len(r)//ns
    N = 1 << S
    mask = N - 1
    k1, k2 = loop
    hpi = np.pi*H
    M = np.zeros(N)
    phi = np.zeros(N)
    psi = 2*np.pi*np.arange(N)/N # survivors start at phases around the circle: the loop pulls the right one in
    nu = np.zeros(N)
    Lh = max(L - 1, 1)
    ah = np.zeros((N, Lh))
    prev = np.zeros((n, N), np.int32)
    st = np.arange(N)
    b1 = (st & 1)
    b2 = (st >> 1) & 1
    for k in range(n):
        rk = r[k*ns:(k + 1)*ns]
        par = (k + parity) & 1
        # windows W (N, 2, L): alpha_k for b = 0, 1, then the survivor's alpha_{k-1} ... alpha_{k-L+1}
        an = atab[par][np.arange(2)[None, :] + 2*b1[:, None] + 4*b2[:, None]].astype(np.float64) # (N, 2)
        W = np.empty((N, 2, L))
        W[:, :, 0] = an
        if L > 1:
            W[:, :, 1:] = ah[:, None, :L - 1]
        theta = phi[:, None, None] + psi[:, None, None] + np.einsum("sbj,jn->sbn", W, Q)
        c = np.sum(rk[None, None, :]*np.exp(-1j*theta), axis=2) # (N, 2)
        m = M[:, None] + c.real
        t = np.arange(N)
        p0 = t >> 1
        p1 = p0 | (N >> 1)
        bb = t & 1
        m0, m1 = m[p0, bb], m[p1, bb]
        win = np.where(m0 >= m1, p0, p1)
        prev[k] = win
        cw = c[win, bb]
        e = np.angle(cw)
        nphi = phi[win] + hpi*W[win, bb, L - 1]
        npsi = psi[win] + k1*e + nu[win]
        nnu = nu[win] + k2*e
        nah = np.empty_like(ah)
        nah[:, 0] = an[win, bb]
        if Lh > 1:
            nah[:, 1:] = ah[win, :Lh - 1]
        M, phi, psi, nu, ah = m[win, bb], np.mod(nphi, 2*np.pi), np.mod(npsi, 2*np.pi), nnu, nah
        b2 = b1[win]
        b1 = bb.copy()
    s = int(np.argmax(M))
    best = float(M[s])
    bits = np.zeros(n, np.uint8)
    for k in range(n - 1, -1, -1):
        bits[k] = s & 1
        s = int(prev[k, s])
    return bits, best

def mlse(r, name, ns, parity=0, S=None, loop=(0.08, 0.002)):
    """Bits of r: samples at ns a symbol, sample 0 at a symbol's start, carrier near 0. S: trellis state bits (default
    3; at least 2, the SOQPSK precoder's memory). loop: (k1, k2) of each survivor's carrier loop, per symbol.
    Returns (bits, match)"""
    r = np.ascontiguousarray(r, dtype=np.complex128)
    Q = np.ascontiguousarray(phase_table(name, ns))
    atab = np.ascontiguousarray(alpha_table(name))
    S = (3 if SCHEMES[name][2] <= 4 else 5) if S is None else max(2, int(S))
    n = len(r)//ns
    if n < 1:
        raise ValueError("fewer samples than a symbol")
    r = r[:n*ns]
    nat = _accel.native
    if nat is not None and hasattr(nat, "cpm_psp"):
        bits, best = nat.cpm_psp(r, Q, atab, S, int(parity), float(loop[0]), float(loop[1]))
        bits = np.asarray(bits)
    else:
        bits, best = _mlse_python(r, Q, atab, S, int(parity), loop)
    match = best/np.sqrt(np.sum(np.abs(r)**2)*n*ns)
    return bits, float(match)

def interpolate(x, pos, taps=16, chunk=1 << 16):
    """x at fractional sample positions pos (Kaiser-windowed sinc, `taps` long; x band-limited below Fs/2; outside x
    is 0), `chunk` positions at a time"""
    x = np.asarray(x, dtype=np.complex128)
    pos = np.asarray(pos, dtype=np.float64)
    k = np.arange(-taps//2 + 1, taps//2 + 1)
    half = taps/2
    out = np.empty(len(pos), np.complex128)
    for a in range(0, len(pos), chunk):
        p = pos[a:a + chunk]
        i0 = np.floor(p).astype(np.int64)
        d = k[None, :] - (p - i0)[:, None]
        w = np.sinc(d)*np.i0(6.0*np.sqrt(np.clip(1 - (d/half)**2, 0, None)))/np.i0(6.0)
        idx = i0[:, None] + k[None, :]
        valid = (idx >= 0) & (idx < len(x))
        out[a:a + chunk] = np.sum(np.where(valid, x[np.clip(idx, 0, max(len(x) - 1, 0))], 0)*w, axis=1)
    return out

def _peak(P, f, i):
    """f at bin i refined by a parabola through its neighbours"""
    n = len(P)
    a, b, c = P[(i - 1) % n], P[i], P[(i + 1) % n]
    den = a - 2*b + c
    return f[i] + (0.5*(a - c)/den if den else 0.0)*(f[1] - f[0])

def square_lines(x, Fs, baud=None, min_ratio=10.0, top=12):
    """The x^2 line pair of an h = 1/2 CPM signal (see the module): (baud, cfo, ratio), ratio the weaker line over the
    spectrum's median (power); None when there is no pair over min_ratio. baud given: the pair is looked for that far
    apart"""
    x = np.asarray(x, dtype=np.complex128)
    nfft = min(1 << int(np.ceil(np.log2(max(len(x), 1024)))), 1 << 22)
    y = x[:nfft]**2
    P = np.abs(np.fft.fft(y*np.hanning(len(y)), nfft))**2
    f = np.fft.fftfreq(nfft, 1/Fs)
    med = float(np.median(P))
    if baud is not None:
        d = int(round(baud/(Fs/nfft)))
        Pm = np.minimum(P, np.roll(P, -d)) # the weaker of the lines at f and f + baud
        k = int(np.argmax(Pm))
        if Pm[k] < min_ratio*med:
            return None
        f1, f2 = _peak(P, f, k), _peak(P, f, (k + d) % nfft)
        return float(baud), float((f1 + f2)/4), float(Pm[k]/med)
    # every local maximum over min_ratio, strongest first; the pair whose weaker line is strongest
    loc = np.flatnonzero((P > np.roll(P, 1)) & (P >= np.roll(P, -1)) & (P > min_ratio*med))
    loc = loc[np.argsort(-P[loc])][:top]
    best = None
    for i in range(len(loc)):
        for j in range(i + 1, len(loc)):
            a, b = loc[i], loc[j]
            w = min(P[a], P[b])
            if abs(f[a] - f[b]) > 8*Fs/nfft and (best is None or w > best[0]):
                best = (w, a, b)
    if best is None:
        return None
    w, a, b = best
    f1, f2 = _peak(P, f, a), _peak(P, f, b)
    return float(abs(f1 - f2)), float((f1 + f2)/4), float(w/med)

class CPM:
    """Result of `demodulate`: bits, scheme, baud (Hz, clock-corrected), cfo (Hz), tau (s: the first symbol's start),
    match (0..1), lines (the weaker x^2 line over the spectrum's median; 0 when rate and carrier came from the
    instantaneous frequency instead)"""
    __slots__ = ("bits", "scheme", "baud", "cfo", "tau", "match", "lines")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        return (f"{self.scheme}: {len(self.bits)} bits at {self.baud:.2f} Bd, carrier {self.cfo:+.1f} Hz, match "
                f"{self.match:.3f}")

    __repr__ = __str__

def _symbols(x, Fs, baud, ns, tau, n, rate=1.0):
    """x at ns samples a symbol from time tau (s), n symbols, the symbol clock scaled by `rate`"""
    pos = (tau + np.arange(n*ns)/(ns*baud*rate))*Fs
    return interpolate(x, pos)

def _time(x, Fs, baud, name, ns, t0, n, S, loop):
    """(match, tau) best over tau in [t0, t0 + T) at 1/ns of a symbol, on n symbols"""
    best = (-np.inf, t0)
    for j in range(ns):
        tau = t0 + j/(ns*baud)
        _, m = mlse(_symbols(x, Fs, baud, ns, tau, n), name, ns, 0, S, loop)
        if m > best[0]:
            best = (m, tau)
    return best

def front(x, Fs, baud=None):
    """(baud, cfo, lines) of an h = 1/2 CPM signal: the x^2 pair, else fsk.analyze's rate and the median frequency"""
    sq = square_lines(x, Fs, baud)
    if sq is not None:
        return sq
    if baud is None:
        from . import fsk
        r = fsk.analyze(x, Fs, force=True)
        if r.baud is None:
            raise ValueError(f"no symbol rate: no x^2 lines, and {r.why}")
        baud = r.baud
    ph = np.angle(x[1:]*np.conj(x[:-1]))
    return float(baud), float(np.median(ph)*Fs/(2*np.pi)), 0.0

def demodulate(x, Fs, scheme="gmsk_0.3", baud=None, ns=8, S=None, loop=(0.08, 0.002), train=1500, clock=True):
    """Bits of a GMSK / SOQPSK signal x (complex baseband at Fs, a channel around it; see the module). baud: the symbol
    rate when known. train: symbols the timing search runs on. clock: correct the symbol clock from stretches across a
    long record. Returns CPM"""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme: one of {list(SCHEMES)}")
    x = np.asarray(x, dtype=np.complex128)
    baud, cfo, lines = front(x, Fs, baud)
    x = x*np.exp(-2j*np.pi*cfo*np.arange(len(x))/Fs)
    sps = Fs/baud
    nsym = int((len(x) - 16)/sps) - 2
    if nsym < 8:
        raise ValueError("fewer than 8 symbols")
    T = 1/baud
    m, tau = _time(x, Fs, baud, scheme, ns, 8/Fs, min(train, nsym), S, loop)
    rate = 1.0
    if clock and nsym > 4*train:
        # time stretches across the record; a line through their starts (in symbols) is the clock error
        starts = np.linspace(0, nsym - train - 2, 5).astype(int)
        taus = []
        for s0 in starts:
            _, tt = _time(x, Fs, baud, scheme, ns, tau + s0*T - T/2, train, S, loop)
            taus.append(tt - tau - s0*T)
        taus = np.unwrap(np.array(taus)*2*np.pi/T)*T/(2*np.pi)
        slope = np.polyfit(starts*T, taus, 1)[0] # seconds of drift a second
        rate = 1/(1 + slope)
    n = int((len(x)/Fs - tau - 16/Fs)*baud*rate) - 1
    bits, match = mlse(_symbols(x, Fs, baud, ns, tau, n, rate), scheme, ns, 0, S, loop)
    return CPM(bits=bits, scheme=scheme, baud=baud*rate, cfo=cfo, tau=tau, match=match, lines=lines)

def identify(x, Fs, baud=None, schemes=None, n=1500, ns=8):
    """Which scheme fits x best: [(match, scheme)] best first, each timed on its own over n symbols"""
    x = np.asarray(x, dtype=np.complex128)
    baud, _, _ = front(x, Fs, baud)
    x = x[:int((n + 40)*Fs/baud)]
    out = []
    for name in schemes or SCHEMES:
        r = demodulate(x, Fs, name, baud, ns=ns, train=n, clock=False)
        out.append((r.match, name))
    return sorted(out, key=lambda t: -t[0])
