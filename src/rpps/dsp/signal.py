import numpy as np

class Signal:
    """A signal found in a spectrum

    mid: centre [Hz], width: occupied bandwidth [Hz], baud: kept as an alias of width for older callers
    bottom: noise floor [dB], height: peak above the floor [dB], snr: in-band power over noise power in that band [dB]
    """
    def __init__(self, mid, width, bottom, height, snr=None):
        self.mid = mid
        self.width = width
        self.bottom = bottom
        self.height = height
        self.snr = snr

    @property
    def baud(self):
        return self.width

    def __repr__(self):
        snr = "" if self.snr is None else f", snr {self.snr:.1f} dB"
        return f"<Signal {self.mid/1e6:+.4f} MHz, {self.width/1e3:.1f} kHz wide{snr}>"

def welch(samps, Fs: float = 1, nfft: int = 4096, overlap: float = 0.5):
    """Averaged periodogram (Hann window), fftshifted. Returns (power per bin, number of averages)"""
    samps = np.asarray(samps)
    step = max(1, int(nfft*(1-overlap)))
    starts = np.arange(0, len(samps) - nfft + 1, step)
    if len(starts) == 0:
        raise ValueError(f"Need at least {nfft} samples")
    win = np.hanning(nfft)
    acc = np.zeros(nfft)
    for s in starts:
        acc += np.abs(np.fft.fft(samps[s:s+nfft]*win))**2
    acc /= len(starts) * np.sum(win**2) * Fs
    return np.fft.fftshift(acc), len(starts)

def noise_floor(db, iters=4, margin=3.0):
    """Robust noise floor [dB]: median, then median of what is within `margin` dB above it (signals drop out)"""
    floor = np.median(db)
    for _ in range(iters):
        sel = db < floor + margin
        floor = np.median(db[sel])
    return floor

def find_signals(samps, Fs: float = 1, nfft=None, min_snr_db=None, gap=None, occupied=0.99, min_bins=4, edge=0.02):
    """Find signals in a spectrum

    Averages periodograms (more samples means a finer, quieter spectrum), finds the noise floor, and takes every
    run of bins above floor + threshold as a signal. The threshold is a few standard deviations of the averaged
    estimate, so long captures can see weaker and narrower signals.

    nfft: FFT size [largest power of two giving at least ~16 averages, up to 65536]
    min_snr_db: detection threshold above the floor [4 sigma of the estimate, at least 1 dB]
    gap: bins of dip tolerated inside one signal [nfft/256]
    occupied: fraction of the signal's power that defines its width (edges are found from the cumulative power)
    min_bins: narrowest signal reported
    edge: fraction of the band at each end to ignore (receiver filter roll-off)
    Returns a list of Signal sorted by frequency.
    """
    samps = np.asarray(samps)
    N = len(samps)
    if nfft is None:
        nfft = 1 << int(np.clip(np.floor(np.log2(max(N//16, 64))), 6, 16))
    pw, K = welch(samps, Fs, nfft)
    db = 10*np.log10(pw + 1e-30)
    sigma = 4.34/np.sqrt(K) # Standard deviation of a K-average periodogram, in dB
    floor = noise_floor(db)
    thr = floor + (min_snr_db if min_snr_db is not None else max(4*sigma, 1.0))

    # Light smoothing so noise peaks inside a signal don't split it
    k = max(3, nfft//1024*2 + 1)
    sm = 10*np.log10(np.convolve(pw, np.ones(k)/k, mode="same") + 1e-30)
    above = sm > thr
    gap = max(2, nfft//256) if gap is None else gap
    idx = np.nonzero(above)[0]
    runs = []
    for i in idx:
        if runs and i - runs[-1][1] <= gap:
            runs[-1][1] = i
        else:
            runs.append([i, i])

    bin_hz = Fs/nfft
    floor_lin = 10**(floor/10)
    out = []
    for lo, hi in runs:
        if hi - lo + 1 < min_bins:
            continue
        # Widen a little: the smoothed edge sits inside the skirts
        lo, hi = max(0, lo - k//2), min(nfft-1, hi + k//2)
        excess = np.clip(pw[lo:hi+1] - floor_lin, 0, None)
        if excess.sum() <= 0:
            continue
        c = np.cumsum(excess)/excess.sum()
        e0 = lo + int(np.searchsorted(c, (1-occupied)/2))
        e1 = lo + int(np.searchsorted(c, 1-(1-occupied)/2))
        if e0 < edge*nfft or e1 > (1-edge)*nfft:
            continue
        width = max(e1 - e0 + 1, 1)*bin_hz
        mid = ((e0 + e1)/2 - nfft/2)*bin_hz
        band = pw[e0:e1+1]
        snr = 10*np.log10(max(np.sum(band - floor_lin), 1e-30)/(floor_lin*len(band)))
        out.append(Signal(mid, width, floor, float(db[lo:hi+1].max() - floor), float(snr)))
    return out
