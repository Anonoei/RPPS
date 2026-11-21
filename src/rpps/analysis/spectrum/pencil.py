"""Spectral lines from short records: the matrix pencil (Hua & Sarkar 1990)

    lines = rp.analysis.spectrum.pencil.lines(x, Fs)         # [(freq Hz, amplitude, damping /s), ...] strongest first

An FFT resolves lines 1/T apart (T the record) and smears each over its window. A sum of K exponentials
x[n] = sum a_k z_k^n has a Hankel matrix of rank K, the Vandermonde structure of the z_k; the matrix pencil takes the z_k
from it directly (eigenvalues of Y1^+ Y0, Y0 and Y1 the matrix without its last and first columns, through the K principal
singular vectors), so two lines a fraction of 1/T apart are told apart and each frequency is found to far finer than a bin:
FSK tones in a short burst, a pilot beside a carrier, cyclic lines of a short capture. K comes from the singular values
(those over `floor` of the largest), or is given.
"""
import numpy as np

def lines(x, Fs, K=None, L=None, floor=0.05, max_samples=4096):
    """Exponentials in x: [(frequency Hz, amplitude (complex), damping 1/s (0 for a steady line))], strongest first.
    K: how many (default: the singular values over `floor` of the largest). L: the pencil parameter (default n/3)"""
    x = np.asarray(x, dtype=np.complex128)[:max_samples]
    n = len(x)
    L = n//3 if L is None else int(L)
    if n < 8 or not 1 <= L < n - 1:
        raise ValueError("too few samples")
    Y = np.lib.stride_tricks.sliding_window_view(x, L + 1) # (n - L) x (L + 1) Hankel
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    if K is None:
        K = int(np.sum(s > floor*s[0]))
    K = max(1, min(K, L))
    V = Vh[:K].T # (L + 1) x K: Y's rows, sums of the vectors (z^0 .. z^L), are in the span of these
    V0, V1 = V[:-1], V[1:]
    z = np.linalg.eigvals(np.linalg.pinv(V0) @ V1)
    # amplitudes by least squares on the Vandermonde of the z
    Z = z[None, :]**np.arange(n)[:, None]
    a, *_ = np.linalg.lstsq(Z, x, rcond=None)
    f = np.angle(z)/(2*np.pi)*Fs
    damp = -np.log(np.maximum(np.abs(z), 1e-300))*Fs
    order = np.argsort(-np.abs(a))
    return [(float(f[i]), complex(a[i]), float(damp[i])) for i in order]
