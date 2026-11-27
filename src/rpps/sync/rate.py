"""Symbol rate estimates: the |x|^2 cyclostationary line on samples (`find_baud`), the rate of symbols (`find_rate`), the
pulse's roll-off from the spectrum (`find_rolloff`)"""
import numpy as np

from ..math.interp import parabolic_offset

CLEAR_LINE = 5.0 # A symbol-rate line this many times its floor is not noise (noise peaks reach 3-4)

def find_baud(x, Fs, lo=None, hi=None, pad=4):
    """Find the symbol rate from IQ samples using the |x|^2 cyclostationary line

    lo/hi: search range in Hz [Fs/16, Fs/2.5]
    returns (baud, strength). Strength is the line over the local floor, a few is noise.
    Under 2.5 samples a symbol the line is at Fs/2.5 or above, and at 2 |x|^2 aliases it onto Fs/2: with the default
    range and no line over CLEAR_LINE, x is interpolated to twice its rate (in frequency) and Fs/2.5 to Fs/1.5 searched
    there too, its line taken when clear, stronger and no wider than the signal (16QAM at exactly 2 samples a symbol read a noise peak as its rate)
    """
    n = len(x)
    if n < 256:
        return 0.0, 0.0
    got = _line(x, Fs, Fs/16 if lo is None else lo, Fs/2.5 if hi is None else hi, pad)
    if hi is None and got[1] < CLEAR_LINE:
        X = np.fft.fft(np.asarray(x, dtype=np.complex128))
        h = n//2
        Z = np.zeros(2*n, np.complex128)
        Z[:h], Z[2*n - (n - h):] = X[:h], X[h:]
        up = _line(np.fft.ifft(Z), 2*Fs, Fs/2.5, Fs/1.5, pad)
        # Clear on its own (a weak true line beats noise up there), and a signal as wide as such a rate: a narrow one
        # shows lines at Fs/2 that are no rate of its (a channel's own filtering)
        if up[1] >= CLEAR_LINE and up[1] > got[1] and _width(X, Fs) >= 0.7*up[0]:
            return up
    return got

def _width(X, Fs, share=0.9):
    """The width [Hz] of the fewest bins of spectrum X (an FFT at Fs) that hold `share` of its power"""
    P = np.sort(np.abs(X)**2)[::-1]
    c = np.cumsum(P)
    return (int(np.searchsorted(c, share*c[-1])) + 1)*Fs/len(X)

def _line(x, Fs, lo, hi, pad):
    """find_baud's search over [lo, hi] Hz: (baud, strength)"""
    n = len(x)
    z = np.abs(x)**2
    z = (z - z.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(z, pad*n))
    freq = np.fft.rfftfreq(pad*n, 1/Fs)
    k = min(pad*64+1, len(spec)//2*2 - 1)
    floor = np.convolve(spec, np.ones(k)/k, mode="same")[:len(spec)]
    ratio = spec/(floor + 1e-30)
    band = np.where((freq >= lo) & (freq <= hi), ratio, 0.0)
    k = int(np.argmax(band))
    d = parabolic_offset(*np.log(spec[k-1:k+2] + 1e-30)) if 0 < k < len(spec)-1 else 0.0
    return (k + d)*Fs/(pad*n), float(band[k])

def find_rate(syms):
    """Find symbol rate (as a fraction of the sample rate) from IQ samples"""
    return find_baud(syms, 1.0)[0]


def raised_cosine_psd(f, baud, beta):
    """The power spectrum of RRC-shaped symbols (|RRC|^2, a raised cosine) at frequencies f [Hz], 1 in the flat part"""
    a = np.abs(np.asarray(f, dtype=np.float64))/baud
    lo, hi = (1 - beta)/2, (1 + beta)/2
    out = np.where(a <= lo, 1.0, 0.0)
    if beta > 0:
        mid = (a > lo) & (a < hi)
        out = np.where(mid, 0.5*(1 + np.cos(np.pi/beta*(a - lo))), out)
    return out

ROLLOFF_GRID = np.round(np.arange(0.02, 0.61, 0.01), 2)
ROLLOFFS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35) # The roll-offs standards use (DVB-S2X's smallest three, DVB-S2's, 0.25)

def nearest_rolloff(beta):
    """The standard roll-off (ROLLOFFS) nearest a measured one"""
    return min(ROLLOFFS, key=lambda r: abs(r - beta))

def find_rolloff(x, Fs, baud, cfo=0.0, nfft=None, grid=ROLLOFF_GRID):
    """The roll-off of an RRC-shaped signal from its spectrum: (beta, residual)

    A Welch spectrum of x (the signal at `cfo` Hz from x's centre, `baud` symbols a second, some floor beside it) fitted
    by a raised cosine of each beta in `grid` plus a flat floor, each by least squares in its level and floor; the beta
    whose fit leaves the least (residual: its rms over the signal's level). It measures the composite edge: a transponder
    or receiver filter that steepens the edges reads as a smaller beta than the pulse's. Matched-filter likelihoods can't
    tell 0.05 from 0.15 (rpps.sync.detect names 0.15 for both); the edges can: 8192 symbols at 10 dB read 0.05, 0.10,
    0.20, 0.35 as 0.068, 0.109, 0.201, 0.39 (+-0.01 .. 0.05), each nearest its own of ROLLOFFS."""
    x = np.asarray(x)
    if nfft is None: # bins of a 64th of the rate or finer (a 0.05 edge is 3 of them; finer bins average fewer segments
        nfft = int(min(2**16, 2**np.ceil(np.log2(64*Fs/baud)), max(64, 2**np.floor(np.log2(max(len(x)//8, 64))))))
    n = len(x)//nfft*nfft
    if n == 0:
        return None, None
    w = np.hanning(nfft)
    P = np.mean(np.abs(np.fft.fft(x[:n].reshape(-1, nfft)*w, axis=1))**2, axis=0)
    f = np.fft.fftfreq(nfft, 1/Fs) - cfo
    f = (f + Fs/2) % Fs - Fs/2
    use = np.abs(f) < min(0.8*baud, 0.5*Fs) # the signal and its edges, with a little floor
    f, P = f[use], P[use]
    best = (np.inf, None)
    for beta in grid:
        h = raised_cosine_psd(f, baud, beta)
        A = np.stack((h, np.ones_like(h)), axis=1)
        coef, *_ = np.linalg.lstsq(A, P, rcond=None)
        r = float(np.sqrt(np.mean((A @ coef - P)**2))/max(coef[0], 1e-30))
        if r < best[0]:
            best = (r, float(beta))
    return best[1], best[0]
