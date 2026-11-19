"""How much of a signal we are locked to: regenerate it from its symbols, subtract it, and measure what is left

    c = rp.analysis.modulation.cancel.cancel(x, Fs, symbols, sps, tau)   # x: samples; symbols: decided (or re-encoded) symbols,
                                                              # symbol k at sample tau + k sps
    print(c)       # e.g. "excess 0.2 dB over the floor: SNR 18.3 dB after cancelling, 18.5 dB the floor allows"
    c.residual     # x without the signal: run rp.analysis.survey / cyclic.features on it for what was underneath

The paired-carrier (PCMA) receiver's step, used as a measurement. A linear modulation is x[n] = sum_k s_k h(n - tau -
k sps) + noise. With the symbols known (decisions, or better the decoded bits encoded again), the pulse h, gain, phase,
timing, a fractional rate and any linear distortion are one least-squares fit: h is a combination of band-limited
kernels half a symbol apart (`span` symbols each side), fitted again every `block` symbols so slow phase, gain and timing
drift are followed. What the fit cannot take off is noise plus whatever our model of the signal misses: phase noise
faster than a block, non-linearity, symbol decisions that are wrong, interference, a slip.

The measure is the residual's density in the signal's band against the noise floor outside it (the capture must be
wider than the signal): `excess_db` 0 means everything left is thermal noise, and the SNR after cancelling equals the
SNR the floor allows; every dB of excess is signal we are not locked to. `profile` is the excess block by block (where
lock is lost or a burst of interference arrives). The residual is also how a second, weaker signal under the first is
found: the strong one taken off, survey the rest.
"""
import numpy as np

from ..spectrum.cyclic import noise_floor

class Cancelled:
    """Result of `cancel`. residual: x less the regenerated signal; model: the regenerated signal; excess_db: residual
    density in band over the floor outside (None without an outside); snr_cancel_db: signal over residual in band;
    snr_floor_db: signal over the floor in band (what perfect lock leaves); profile: excess per block (dB);
    bandwidth: the band (Hz) measured; cfo: the frequency (Hz) taken off before fitting; explained: fraction of x's
    in-band power the model takes off"""
    __slots__ = ("residual", "model", "excess_db", "snr_cancel_db", "snr_floor_db", "profile", "bandwidth", "cfo",
                 "explained", "taps")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def loss_db(self):
        """SNR lost to what we are not locked to"""
        return None if self.snr_floor_db is None else self.snr_floor_db - self.snr_cancel_db

    def __str__(self):
        if self.excess_db is None:
            return (f"SNR {self.snr_cancel_db:.1f} dB after cancelling ({100*self.explained:.1f}% of the in-band power "
                    "taken off); no out-of-band floor to compare with")
        bad = int(np.sum(np.asarray(self.profile) > 3.0))
        return (f"excess {self.excess_db:.2f} dB over the floor: SNR {self.snr_cancel_db:.1f} dB after cancelling, "
                f"{self.snr_floor_db:.1f} dB the floor allows (loss {self.loss_db:.2f} dB); {bad} of "
                f"{len(self.profile)} blocks more than 3 dB over")

    __repr__ = __str__

def kernel(t, sps, span):
    """Band-limited kernel (|f| < 1/sps cycles a sample, the band of a pulse with roll-off up to 1) of half-symbol
    spacing, Hann-tapered to +-span symbols: t in samples"""
    u = t/(sps/2)
    w = np.where(np.abs(t) < span*sps, 0.5 + 0.5*np.cos(np.pi*t/(span*sps)), 0.0)
    return np.sinc(u)*w

def regressors(n, symbols, sps, tau, span=6):
    """The basis signals: column j (of 4 span + 1) is sum_k s_k kernel(n - tau - k sps - j' sps/2), j' = j - 2 span.
    Returns an n x (4 span + 1) complex array"""
    s = np.asarray(symbols, dtype=np.complex128)
    J = np.arange(-2*span, 2*span + 1)
    t = np.arange(n, dtype=np.float64) - tau
    k0 = np.floor(t/sps).astype(np.int64)
    u = t - k0*sps # where in its symbol each sample is
    A = np.zeros((n, len(J)), dtype=np.complex128)
    reach = int(np.ceil(1.5*span + 1)) # symbols each side a sample hears: kernel span + its offset
    off = J*sps/2
    # The kernels depend on a sample only through its place u in its symbol: tabulate over u (linear interpolation
    # between 1024 points a symbol: -80 dB) instead of evaluating them for every sample
    G = 1024
    grid = np.arange(G + 1)/G*sps
    g = u/sps*G
    i = np.minimum(g.astype(np.int64), G - 1)
    w = (g - i)[:, None]
    for d in range(-reach, reach + 1):
        k = k0 + d
        ok = (k >= 0) & (k < len(s))
        if not ok.any():
            continue
        T = kernel(grid[:, None] - d*sps - off[None, :], sps, span) # (G + 1) x columns
        io, wo = i[ok], w[ok]
        A[ok] += s[k[ok], None]*(T[io]*(1 - wo) + T[io + 1]*wo)
    return A

def _fit(x, A, block_n, widely):
    """Block-wise least squares: (model, coefficients per block)"""
    n = len(x)
    model = np.zeros(n, dtype=np.complex128)
    coefs = []
    for a in range(0, n, block_n):
        b = min(n, a + block_n)
        if b - a < block_n//2 and a > 0: # A short tail joins the block before
            a = max(0, n - block_n)
        M = A[a:b]
        if widely:
            M = np.hstack((M, np.conj(M)))
        H = M.conj().T
        R = H @ M
        R[np.diag_indices_from(R)] += 1e-9*np.real(np.trace(R))/len(R) # the kernels overlap: keep it invertible
        c = np.linalg.solve(R, H @ x[a:b])
        model[a:b] = M @ c
        coefs.append(c)
    return model, coefs

def _psd(x, nfft):
    seg = len(x)//nfft
    w = np.hanning(nfft)
    return np.fft.fftshift(np.mean(np.abs(np.fft.fft(x[:seg*nfft].reshape(seg, nfft)*w, axis=1))**2, axis=0))/np.sum(w**2)

def decide(y, points):
    """Nearest constellation point of each symbol"""
    pts = np.asarray(points, dtype=np.complex128)
    return pts[np.argmin(np.abs(np.asarray(y)[:, None] - pts[None, :]), axis=1)]

def cancel(x, Fs, symbols, sps, tau=0.0, span=6, block=256, widely=False, cfo=None, nfft=1024):
    """Regenerate the signal from its symbols and take it off x (see the module). symbols: complex symbol values, symbol k
    at sample tau + k sps (sps may be fractional; a slow clock drift is followed by the block fits). block: symbols per
    fit. widely: fit the conjugate too (IQ imbalance). cfo: Hz to take off first (None: estimated from the fits)"""
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    A = regressors(n, symbols, sps, tau, span)
    block_n = max(int(block*sps), 8*A.shape[1])
    t = np.arange(n)
    f0 = 0.0
    if cfo is None:
        # The fits' phase from block to block is the residual frequency: take it off and fit again
        _, coefs = _fit(x, A, block_n, widely)
        if len(coefs) > 2:
            ph = np.unwrap([np.angle(np.vdot(coefs[0], c)) for c in coefs])
            centres = (np.arange(len(coefs)) + 0.5)*block_n
            f0 = float(np.polyfit(centres, ph, 1)[0]/(2*np.pi)*Fs)
    else:
        f0 = float(cfo)
    rot = np.exp(-2j*np.pi*f0/Fs*t)
    model, coefs = _fit(x*rot, A, block_n, widely)
    model = model*np.conj(rot)
    r = x - model
    P = A.shape[1]*(2 if widely else 1)
    dof = block_n/max(block_n - P, 1) # each fit takes P of the block's noise dimensions with it
    # In-band against out-of-band densities
    nfft = int(min(nfft, 2**int(np.log2(max(n//8, 16)))))
    Px, Pr, Pm = _psd(x, nfft), _psd(r, nfft)*dof, _psd(model, nfft)
    floor = noise_floor(Px)
    band = Pm > 2*floor # where the signal stands over the floor
    out = ~band & (Px < 3*floor)
    bw = Fs*np.count_nonzero(band)/nfft
    sig = np.sum(Pm[band])
    res = np.sum(Pr[band])
    explained = float(1 - res/max(np.sum(Px[band]), 1e-300))
    snr_c = 10*np.log10(max(sig, 1e-300)/max(res, 1e-300))
    if np.count_nonzero(out) >= nfft//16:
        N0 = float(np.median(Px[out])) # the median: skirts of the signal or another one nearby stay out of it
        excess = 10*np.log10(max(np.mean(Pr[band]), 1e-300)/N0)
        snr_f = 10*np.log10(max(sig, 1e-300)/(N0*np.count_nonzero(band)))
        # Block by block: the residual's power over what the floor gives that band, every `block` symbols
        frac = np.count_nonzero(band)/nfft
        prof = []
        for a in range(0, n - block_n + 1, block_n):
            seg = r[a:a + block_n]
            # The residual is not only in band; filter it to the band through the PSD bins
            S = np.fft.fftshift(np.fft.fft(seg*np.hanning(len(seg))))
            k = np.linspace(0, nfft, len(S), endpoint=False).astype(int)
            pin = np.sum(np.abs(S[band[k]])**2)/np.sum(np.hanning(len(seg))**2)
            prof.append(10*np.log10(max(pin*dof, 1e-300)/(N0*frac*len(seg))))
    else:
        excess, snr_f, prof = None, None, []
    return Cancelled(residual=r.astype(np.complex64), model=model.astype(np.complex64), excess_db=excess,
                     snr_cancel_db=float(snr_c), snr_floor_db=snr_f, profile=np.asarray(prof), bandwidth=bw, cfo=f0,
                     explained=explained, taps=A.shape[1])

def align(x, symbols, sps, search=None, span=6):
    """Where the symbols are in x: tau (samples) of symbol 0, from the cross-correlation of x with the symbols shaped by
    the basis's centre kernel. Only the symbol offset needs to be right: the fitted pulse takes up any timing error
    within a few symbols. search: the samples tau may be in (default the first quarter of x)"""
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    s = np.asarray(symbols, dtype=np.complex128)
    m = min(len(s), 2000)
    probe = regressors(int(m*sps) + int(2*span*sps), s[:m], sps, 0.0, span)[:, 2*span]
    N = 1 << int(np.ceil(np.log2(n + len(probe))))
    c = np.fft.ifft(np.fft.fft(x, N)*np.conj(np.fft.fft(probe, N)))
    hi = n//4 if search is None else search
    cand = np.concatenate((c[:hi], c[N - int(2*sps):])) # tau a little negative too (symbol 0 before x starts)
    lags = np.concatenate((np.arange(hi), np.arange(-int(2*sps), 0)))
    return float(lags[int(np.argmax(np.abs(cand)))])
