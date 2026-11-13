from statistics import NormalDist
from types import SimpleNamespace

import numpy as np

from ..sequence.code import Code

def acquire_code(x, code: Code, sps=1, index=0, Fs=None, doppler=(0.0,), periods=None, threshold=None, pfa=1e-3):
    """Find where a spreading code starts in a signal, and its frequency offset: the parallel code-phase search

    For each trial frequency the signal is mixed down, every code period is correlated with the sequence at all
    delays at once (FFT), and the powers of the periods are added. Power is added, not amplitude, so data modulated on the
    code (a bit flip, a differential phase change) does not cancel it. A peak that stands out of the rest is the signal.

    x: complex samples at `sps` samples per chip. index: which sequence of a set. Fs: sample rate, needed for `doppler`.
    doppler: trial frequency offsets [Hz]; each can only be found to within about 1/(period) Hz, so space them by half that
    periods: how many whole code periods to use (default all there are)
    threshold: the peak's ratio to the mean power of the other delays that counts as locked. Default: what noise alone
        exceeds, in the best of all the (frequency, delay) cells and periods tried, with probability `pfa` (the
        sum of P periods' noise power is gamma distributed; its tail is taken from the Wilson-Hilferty approximation)

    Returns offset (samples into x where a code period starts: give it to Despread), doppler [Hz], metric (peak / mean
    of the other delays at the best frequency), threshold, locked, and `power` (frequencies, delays) for plotting"""
    x = np.asarray(x, dtype=np.complex128)
    sps = int(sps)
    L = code.length*sps
    P = len(x)//L if periods is None else int(periods)
    if P < 1 or P*L > len(x):
        raise ValueError(f"{len(x)} samples is not {max(P, 1)} period(s) of {L}")
    doppler = np.atleast_1d(np.asarray(doppler, dtype=float))
    if np.any(doppler != 0) and Fs is None:
        raise ValueError("Fs is needed to search frequency offsets")
    ref = np.repeat(code[index], sps)
    fref = np.conj(np.fft.fft(ref))
    blocks = x[:P*L]
    power = np.empty((len(doppler), L))
    t = np.arange(P*L)
    for k, f in enumerate(doppler):
        mixed = blocks*np.exp(-2j*np.pi*f*t/Fs) if f else blocks
        c = np.fft.ifft(np.fft.fft(mixed.reshape(P, L), axis=1)*fref, axis=1) # c[p, d]: period p against the code delayed by d
        power[k] = (np.abs(c)**2).sum(axis=0)
    k, d = np.unravel_index(np.argmax(power), power.shape)
    rest = np.delete(power[k], d)
    metric = float(power[k, d]/max(rest.mean(), 1e-30))
    if threshold is None:
        threshold = noise_threshold(P, power.size, pfa)
    return SimpleNamespace(offset=int(d), doppler=float(doppler[k]), metric=metric, threshold=float(threshold), locked=metric >= threshold, power=power)

def noise_threshold(periods, cells, pfa=1e-3):
    """The ratio to the mean that the largest of `cells` noise-only powers, each summed over `periods`, exceeds with probability pfa"""
    dof = 2*periods # the power of a complex Gaussian is chi-square with 2 degrees of freedom; a sum of P of them, 2P
    z = NormalDist().inv_cdf(1 - pfa/cells)
    return (1 - 2/(9*dof) + z*np.sqrt(2/(9*dof)))**3
