"""Spectral coherence over all cycle frequencies (FFT accumulation), the significant ones, and blind FRESH filtering

    table = rp.analysis.spectrum.scf.cycle_frequencies(x, Fs)              # [(alpha Hz, coherence, conjugate), ...] significant
    f, a, c = rp.analysis.spectrum.scf.coherence(x, Fs)                     # every (f, alpha) point
    y, w = rp.analysis.spectrum.scf.fresh(x, Fs, ref, shifts=(-125e3, 0, 125e3), train=slice(0, len(ref)))

rpps.analysis.spectrum.cyclic tests chosen nonlinearities (|x|^2, x^M) for lines; that sees the strongest signal's rate and carrier.
The spectral correlation function S_x^a(f) = E[X(f + a/2) X*(f - a/2)] at every cycle frequency a shows every
cyclostationary signal in the band at once, each by its own pattern of cycle frequencies (rates, carriers, and their
sums): the way to see signals that share a band. One computes
it over all (f, a); here by the FFT accumulation method (Roberts, Brown & Loomis 1991), every pair of channels' product
transformed over time, as rpps.analysis.spectrum.cyclic.scf. Normalized by the two channels' power, the
spectral coherence is near 0 for noise and up to 1 for a feature, whatever the power: thresholded, it is a blind table of
cycle frequencies (the CSP Blog's "spectral coherence and blind estimation of significant cycle frequencies").
Conjugate: E[X(f + a/2) X(a/2 - f)], the cycle frequencies of x^2: twice a carrier, and that plus and minus the rate, for
BPSK / OQPSK / MSK-like signals.

fresh: a frequency-shift filter (Gardner 1993) sums filtered copies of x shifted by multiples of a signal's cycle
frequencies, the spectral redundancy of a band-limited cyclostationary signal; an interferer overlapping it in frequency
but with other cycle frequencies does not share that redundancy and can be taken out, which no filter of x alone can do.
The weights are trained by least squares on a stretch where the wanted signal is known: a preamble, or the signal
rpps.analysis.modulation.cancel regenerates from decisions (decision-directed). (Blind weights, by self-coherence restoral (Agee,
Schell & Gardner 1990), need reference branches that are not shifts of the same copies: with one sensor that is not
done here.)
"""
import numpy as np

def coherence(x, Fs, nfft=64, frames=None, hop=None, conjugate=False):
    """Spectral coherence by the FFT accumulation method (as rpps.analysis.spectrum.cyclic.scf): (f Hz, alpha Hz, |coherence|) over
    the points each channel pair is responsible for. |C| = |S^a(f)| / sqrt(S(f + a/2) S(f - a/2)), each channel's own power
    the denominator, so 0 <= |C| <= 1. conjugate: X(f + a/2) X(a/2 - f), the cycle frequencies of x^2"""
    x = np.asarray(x, dtype=np.complex128)
    hop = nfft//4 if hop is None else hop
    P = (len(x) - nfft)//hop + 1 if frames is None else frames
    P = int(2**int(np.log2(max(P, 2))))
    if (P - 1)*hop + nfft > len(x):
        raise ValueError("too few samples for the frames asked")
    idx = np.arange(nfft)[None, :] + hop*np.arange(P)[:, None]
    X = np.fft.fftshift(np.fft.fft(x[idx]*np.hamming(nfft), axis=1), axes=1)
    k = np.arange(-nfft//2, nfft//2)
    X = X*np.exp(-2j*np.pi*k[None, :]*hop*np.arange(P)[:, None]/nfft)
    w = np.hamming(P)[:, None]
    power = np.sum(w*np.abs(X)**2, axis=0)/np.sum(w) # each channel's, weighted as the products are: |C| <= 1
    power = power + 1e-3*power.max() # an empty channel (outside every signal) would make 0/0 coherences
    fk = k*Fs/nfft
    q = np.arange(-P//2, P//2)*Fs/(hop*P)
    keep = np.abs(q) <= Fs/(2*nfft)
    ff, aa, cc = [], [], []
    for a in range(nfft):
        prod = X[:, a:a + 1]*(X if conjugate else np.conj(X))*w # X(f1) X(f2): alpha = f1 + f2; X(f1) X*(f2): f1 - f2
        Z = np.fft.fftshift(np.fft.fft(prod, axis=0), axes=0)/np.sum(w)
        Z = Z[keep]
        alpha = fk[a] + (fk[None, :] if conjugate else -fk[None, :]) + q[keep][:, None]
        f = (fk[a] + (-fk[None, :] if conjugate else fk[None, :]))/2 + 0*q[keep][:, None]
        c = np.abs(Z)/np.sqrt(power[a]*power[None, :])
        ff.append(f.ravel()); aa.append(alpha.ravel()); cc.append(c.ravel())
    return np.concatenate(ff), np.concatenate(aa), np.concatenate(cc)

def cycle_frequencies(x, Fs, nfft=64, conjugate=None, pfa=1e-3, merge=None):
    """Significant cycle frequencies: [(alpha Hz, coherence, conjugate)] best first, non-conjugate a > 0 only (it is
    symmetric) and a = 0 left out. The threshold is where noise alone would exceed it with probability pfa over all the
    points (coherence of noise: Rayleigh, its scale from the median of all points). merge: Hz within which points are one
    cycle frequency (default Fs/nfft)"""
    out = []
    for conj in ((False, True) if conjugate is None else (conjugate,)):
        f, a, c = coherence(x, Fs, nfft, conjugate=conj)
        n = len(c)
        merge_ = Fs/nfft if merge is None else merge # a feature shows in neighbouring channel pairs that far off
        # noise's coherence is Rayleigh: its scale from the median of all points; P(c > t) = exp(-t^2 / (2 s^2)), n points
        s = np.median(c)/np.sqrt(2*np.log(2))
        # (15% over Rayleigh's: measured on noise, the window over frames makes the tail heavier, and the points correlate)
        thr = TAIL*s*np.sqrt(2*np.log(n/pfa))
        sel = c > thr
        if not conj:
            sel &= (a > merge_) # a = 0 is the spectrum itself; negative a mirrors positive
        order = np.argsort(-c[sel])
        taken = []
        for av, cv in zip(a[sel][order], c[sel][order]):
            if all(abs(av - t) > merge_ for t, _ in taken):
                taken.append((float(av), float(cv)))
        out += [(t, cv, conj) for t, cv in taken]
    return sorted(out, key=lambda t: -t[1])

TAIL = 1.15

def _branches(x, shifts, taps, Fs, conj_shifts=()):
    """Regressors: x shifted by each frequency (Hz), and conj(x) by each conj shift, each delayed 0 .. taps-1 samples"""
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(len(x))
    cols = []
    for s in shifts:
        u = x*np.exp(2j*np.pi*s/Fs*n)
        for d in range(taps):
            cols.append(np.concatenate((np.zeros(d, complex), u[:len(u) - d])))
    for s in conj_shifts:
        u = np.conj(x)*np.exp(2j*np.pi*s/Fs*n)
        for d in range(taps):
            cols.append(np.concatenate((np.zeros(d, complex), u[:len(u) - d])))
    return np.stack(cols, axis=1)

def fresh(x, Fs, reference, shifts, taps=16, conj_shifts=(), train=None, reg=1e-6):
    """A frequency-shift filter for the signal `reference` is a stretch of (a known preamble, training, or what
    rpps.analysis.modulation.cancel regenerates from decisions): the least-squares weights over x's copies shifted by `shifts` (Hz;
    multiples of the wanted signal's cycle frequencies, e.g. -Rs, 0, Rs) and, for signals with conjugate features, conj(x)
    shifted by `conj_shifts`. train: the samples (a slice or index array) the reference covers (default all). Returns
    (the filtered signal over all of x, the weights). With shifts (0,) it is an ordinary Wiener filter; the shifted copies
    are what let it take out an interferer overlapping the signal (Gardner 1993)"""
    X = _branches(x, tuple(shifts), taps, Fs, tuple(conj_shifts))
    d = np.asarray(reference, dtype=np.complex128)
    sel = np.arange(len(d)) if train is None else np.arange(len(x))[train]
    A = X[sel]
    R = A.conj().T @ A
    R += reg*np.trace(R).real/len(R)*np.eye(len(R))
    w = np.linalg.solve(R, A.conj().T @ d[:len(sel)])
    return X @ w, w
