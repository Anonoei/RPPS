"""Two signals of equal power on one channel, separated: each one's carrier and timing, then both symbol streams

    p = rp.analysis.separate.estimate(x, Fs, baud, "qpsk")     # two Params: carrier (Hz), timing (s), complex gain
    r = rp.analysis.separate.separate(x, Fs, baud, "qpsk")     # (or params=p) r.symbols[0], r.symbols[1], r.evm

A paired carrier (two transmitters sharing a frequency, "carrier in carrier") at equal power is beyond cancellation:
each is the other's interference at 0 dB, so neither can be demodulated first. What still tells them apart is their
own carrier offset, timing and phase:

  carriers  x^M of an M-PSK signal has a line at M f_i, each signal its own (the cross terms of the two fall elsewhere):
            the two strongest lines near the middle of the x^M spectrum are M f_1 and M f_2
  timing    the M-th power of a shaped PSK signal also pulses once a symbol: lines at M f_i +- baud, whose phase over the
            centre line's is -2 pi baud tau_i (each signal's own, read at its own lines); the half-symbol ambiguity of a
            symmetric pulse is settled by which of tau and tau + T/2 gives the more PSK-like matched-filter output
  gains     each signal's phase (mod 2 pi / M) from its centre line (a^M times a positive sum), its amplitude half the
            matched filter's power, then both refitted from decisions, slowly
            varying (a running mean over `smooth` symbols, which takes up a carrier estimate a little off)

Then both streams are matched-filtered on their own grids, z_i[k], and what each picks up of the other is
c_ij(k, m) = A_j RC((m T + tau_j) - (k T + tau_i)) exp(j 2 pi (f_j - f_i) t_km) (the raised-cosine overlap of the
pulses, turned by the carrier difference). Detection is a sequence estimate over both streams at once (the
"paired carrier" receiver): the symbols of both, merged in time order, each output z[n] mostly the sum of its own
symbol and its two neighbours of the other signal, so a Viterbi search over states (u[n-1], u[n]) (M^2 of them) scores
every output exactly; the further neighbours (RC tails) are taken out with the previous pass's decisions, and the
gains refitted, `iterations` times.

Limits: equal or comparable rates, both of the same PSK. `estimate` needs the two carriers resolved in the x^M
spectrum: apart by more than about 4 / (M x the record's length) (measured: 20 Hz apart from 0.16 s of QPSK is
separated without a symbol error at 15 dB, 5 Hz is not). Closer carriers need `params` from elsewhere (a known
offset, a longer record); the detector then works on timing and phase differences alone, but a stretch where the two
coincide in timing and phase as well is genuinely ambiguous. Signal 0 is the lower carrier.
"""
import numpy as np

class Params:
    """One signal's: cfo (Hz), tau (s: a symbol centre), gain (complex), line (its x^M line over the spectrum's median)"""
    __slots__ = ("cfo", "tau", "gain", "line")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return (f"<signal at {self.cfo:+.2f} Hz, timing {self.tau*1e6:.3f} us, gain {abs(self.gain):.3f} "
                f"/ {np.degrees(np.angle(self.gain)):.1f} deg, line {self.line:.0f}x>")

class Separation:
    """`separate`: symbols [s0, s1] (decided, on each one's grid), soft [s0, s1] (corrected matched-filter outputs with
    the other signal taken out), params [Params, Params], evm [e0, e1], iterations"""
    __slots__ = ("symbols", "soft", "params", "evm", "iterations")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        return "two signals: " + "; ".join(f"{p.cfo:+.1f} Hz, EVM {100*e:.1f}%" for p, e in zip(self.params, self.evm))

    __repr__ = __str__

def _points(mod):
    from .. import symbol as rsym
    return np.asarray(rsym.load(mod).points, dtype=np.complex128)

def _rc(t, beta):
    """Raised cosine (the RRC pulse with itself) at t symbols, 1 at 0"""
    t = np.asarray(t, dtype=np.float64)
    out = np.sinc(t)*np.cos(np.pi*beta*t)
    den = 1 - (2*beta*t)**2
    sing = np.abs(den) < 1e-8
    out = np.where(sing, np.pi/4*np.sinc(1/(2*beta)) if beta > 0 else 0.0, out/np.where(sing, 1, den))
    return out

def _mf(x, Fs, baud, beta, cfo, tau, n0, n):
    """Matched-filter outputs of x at symbol centres tau + k T (k from n0, n of them), x turned down by cfo"""
    from ..sync.time.mat import matrix_mf
    sps = Fs/baud
    c = tau*Fs + (n0 + np.arange(n))*sps
    y = x*np.exp(-2j*np.pi*cfo*np.arange(len(x))/Fs)
    return matrix_mf(y, c, sps, beta, 8)

def _grid(x, Fs, baud, tau):
    """First and count of symbol centres tau + k T whose pulse fits in x"""
    sps = Fs/baud
    n0 = int(np.ceil((8*sps + 2 - tau*Fs)/sps))
    last = (len(x) - 8*sps - 6 - tau*Fs)/sps
    return n0, int(np.floor(last)) - n0 + 1

def estimate(x, Fs, baud, mod="qpsk", beta=0.35, search=None):
    """Both signals' parameters (see the module): [Params, Params], the lower carrier first. search: the largest
    carrier offset (Hz) looked at (default baud/4)"""
    x = np.asarray(x, dtype=np.complex128)
    pts = _points(mod)
    M = len(pts) if len(pts) <= 8 else 4
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(n)))
    y = x**M
    Y = np.fft.fft(y, nfft)
    P = np.abs(Y)**2
    f = np.fft.fftfreq(nfft, 1/Fs)
    search = baud/4 if search is None else search
    med = np.median(P)
    near = np.flatnonzero(np.abs(f) <= M*search)
    loc = near[(P[near] >= P[near - 1]) & (P[near] >= P[(near + 1) % nfft])]
    loc = loc[np.argsort(-P[loc])]
    peaks = []
    for i in loc:
        if all(abs(f[i] - f[j]) > 4*Fs/nfft for j in peaks):
            peaks.append(i)
        if len(peaks) == 2:
            break
    if len(peaks) < 2:
        raise ValueError("fewer than two carrier lines in x^M")
    t = np.arange(n)/Fs
    out = []
    for i in sorted(peaks, key=lambda j: f[j]):
        a, b, c = P[i - 1], P[i], P[(i + 1) % nfft]
        den = a - 2*b + c
        fm = f[i] + (0.5*(a - c)/den if den else 0.0)*Fs/nfft # M f_i
        L0 = np.sum(y*np.exp(-2j*np.pi*fm*t))
        Lp = np.sum(y*np.exp(-2j*np.pi*(fm + baud)*t))
        Lm = np.sum(y*np.exp(-2j*np.pi*(fm - baud)*t))
        # the periodic part's coefficient at +baud over 0 is real for a symmetric pulse: its phase is -2 pi baud tau
        ang = np.angle(Lp*np.conj(L0)) - np.angle(Lm*np.conj(L0)) # both sides, the pulse's own sign cancelling
        tau = (-ang/2/(2*np.pi*baud)) % (1/baud)
        cfo = fm/M
        best = None
        for tt in (tau, (tau + 0.5/baud) % (1/baud)):
            n0, cnt = _grid(x, Fs, baud, tt)
            z = _mf(x, Fs, baud, beta, cfo, tt, n0, min(cnt, 4000))
            q = np.abs(np.mean((z/np.sqrt(np.mean(np.abs(z)**2)))**M))
            if best is None or q > best[0]:
                best = (q, tt, z)
        _, tau, z = best
        # the line is a^M (the same for every point of M-PSK) times the pulse's M-th power summed (> 0): the phase
        # of this signal alone, mod 2 pi / M (the matched filter's own z^M is the two signals' cross terms as much)
        ph = (np.angle(L0) - np.angle(pts[0]**M))/M
        amp = np.sqrt(np.mean(np.abs(z)**2)/2) # equal power: half of the matched filter's output each (refitted later)
        out.append(Params(cfo=float(cfo), tau=float(tau), gain=amp*np.exp(1j*ph), line=float(b/med)))
    return out

def _viterbi(z, g, wl, wr, resid, pts):
    """MLSE over a merged sequence u: z[n] = g[n] u[n] + wl[n] u[n-1] + wr[n] u[n+1] + resid[n] + noise (wl, wr: what
    the neighbours add, gains included). State (u[n-1], u[n]); returns the indices of u"""
    M = len(pts)
    L = len(z)
    S = M*M
    a = np.repeat(np.arange(M), M) # state s = a*M + b: (u[n-1], u[n]) = (pts[a], pts[b])
    b = np.tile(np.arange(M), M)
    metric = np.zeros(S)
    back = np.zeros((L, S), np.int32)
    for n in range(L - 1):
        # next symbol c: from (a, b) to (b, c); the observation z[n] scores (a, b, c)
        pred = (g[n]*pts[b] + wl[n]*pts[a])[:, None] + wr[n]*pts[None, :] + resid[n] # (S, M)
        m = metric[:, None] - np.abs(z[n] - pred)**2
        # new state (b, c): from the states (a', b) for every a'
        m3 = m.reshape(M, M, M) # (a, b, c)
        best_a = np.argmax(m3, axis=0) # (b, c)
        metric = np.take_along_axis(m3, best_a[None], axis=0)[0].reshape(-1)
        back[n + 1] = (best_a*M + np.arange(M)[:, None]).reshape(-1) # previous state (a, b) for each (b, c)
    # the last observation, with no symbol after it
    m = metric - np.abs(z[L - 1] - (g[L - 1]*pts[b] + wl[L - 1]*pts[a]) - resid[L - 1])**2
    st = int(np.argmax(m))
    out = np.zeros(L, np.int64)
    for n in range(L - 1, -1, -1):
        out[n] = st % M
        if n:
            st = int(back[n, st])
    return out

def separate(x, Fs, baud, mod="qpsk", beta=0.35, params=None, iterations=4, smooth=256, taps=6):
    """Both symbol streams of two overlapping signals (see the module): Separation"""
    x = np.asarray(x, dtype=np.complex128)
    pts = _points(mod)
    ps = params or estimate(x, Fs, baud, mod, beta)
    T = 1/baud
    Z, tk = [], []
    for p in ps:
        n0, cnt = _grid(x, Fs, baud, p.tau)
        Z.append(_mf(x, Fs, baud, beta, p.cfo, p.tau, n0, cnt))
        tk.append(p.tau + (n0 + np.arange(cnt))*T)
    # the merged sequence, in time order
    sig = np.concatenate((np.zeros(len(tk[0]), int), np.ones(len(tk[1]), int)))
    idx = np.concatenate((np.arange(len(tk[0])), np.arange(len(tk[1]))))
    tt = np.concatenate(tk)
    order = np.argsort(tt, kind="stable")
    sig, idx, tt = sig[order], idx[order], tt[order]
    z = np.concatenate(Z)[order]
    L = len(z)
    cf = np.array([ps[0].cfo, ps[1].cfo])
    # coupling of merged symbol p into observation n (other signal only), for p = n - taps .. n + taps
    offs = np.arange(-taps, taps + 1)
    P = np.clip(np.arange(L)[:, None] + offs[None, :], 0, L - 1)
    valid = ((np.arange(L)[:, None] + offs[None, :]) >= 0) & ((np.arange(L)[:, None] + offs[None, :]) < L) & \
        (sig[P] != sig[:, None])
    W = _rc((tt[P] - tt[:, None])/T, beta)*np.exp(2j*np.pi*(cf[sig[P]] - cf[sig][:, None])*(tt[P] + tt[:, None])/2)*valid
    gains = [np.full(len(tk[i]), ps[i].gain, np.complex128) for i in range(2)]
    dec = np.zeros(L, np.complex128) # previous pass's decisions (none on the first)
    centre = taps
    for it in range(1, iterations + 1):
        g = np.where(sig == 0, gains[0][np.minimum(idx, len(tk[0]) - 1)], gains[1][np.minimum(idx, len(tk[1]) - 1)])
        GW = W*g[P] # with the coupled symbol's gain
        wl, wr = GW[:, centre - 1], GW[:, centre + 1]
        far = GW.copy()
        far[:, centre - 1:centre + 2] = 0
        resid = np.sum(far*dec[P], axis=1) if it > 1 else np.zeros(L, np.complex128)
        u = _viterbi(z, g, wl, wr, resid, pts)
        dec = pts[u]
        # gains, refitted from the decisions with everything else they hear taken out: a running mean
        clean = z - np.sum(GW*dec[P], axis=1)
        for i in range(2):
            sel = sig == i
            gi = clean[sel]*np.conj(dec[sel])/np.abs(dec[sel])**2
            ker = np.ones(smooth)/smooth
            gains[i] = np.convolve(gi, ker, mode="same")/np.convolve(np.ones(len(gi)), ker, mode="same")
    g = np.where(sig == 0, gains[0][np.minimum(idx, len(tk[0]) - 1)], gains[1][np.minimum(idx, len(tk[1]) - 1)])
    other = np.sum(W*g[P]*dec[P], axis=1)
    y = (z - other)/g
    syms, soft, evm = [], [], []
    for i in range(2):
        sel = sig == i
        yi = y[sel][np.argsort(idx[sel])]
        di = dec[sel][np.argsort(idx[sel])]
        soft.append(yi)
        syms.append(di)
        evm.append(float(np.sqrt(np.mean(np.abs(yi - di)**2)/np.mean(np.abs(pts)**2))))
    return Separation(symbols=syms, soft=soft, params=list(ps), evm=evm, iterations=iterations)
