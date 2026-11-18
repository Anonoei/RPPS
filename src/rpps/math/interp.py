"""Interpolation"""
import numpy as np

def parabolic_offset(a, b, c):
    """Vertex offset (-0.5..0.5) of the parabola through three equally spaced samples a, b, c (b in the middle)"""
    den = a - 2*b + c
    return 0.0 if den == 0 else 0.5*(a - c)/den

def peak(spec, circular=False, log=True):
    """Index of the largest value of a spectrum, and the sub-bin offset from a parabola through its neighbours

    log: fit the parabola to log(spec), the right model for a spectral line
    circular: the spectrum wraps around (an FFT), so the neighbours of the end bins are the other end
    Returns (k, offset), the peak is at k + offset. A peak on the end bin of a non-circular spectrum gets offset 0.
    """
    spec = np.asarray(spec)
    k = int(np.argmax(spec))
    n = len(spec)
    if not circular and not 0 < k < n - 1:
        return k, 0.0
    a, b, c = spec[(k - 1) % n], spec[k], spec[(k + 1) % n]
    if log:
        a, b, c = np.log(a + 1e-30), np.log(b + 1e-30), np.log(c + 1e-30)
    return k, parabolic_offset(a, b, c)

# ---- Fractional delay: x at non-integer positions

REACH = 6 # Samples a fractional interpolator reads on each side of floor(t) at most (12 taps): what a stream keeps

_PHASES = 32 # Rows of a windowed-sinc table: fractional offsets k/32, blended linearly between neighbours
_TABLES = {}

def _sinc_table(taps):
    """Windowed-sinc fractional delay filters, (phases + 1, taps): row k delays by k/phases, taps n = 1-taps/2 .. taps/2"""
    if taps not in _TABLES:
        n = np.arange(1 - taps//2, taps//2 + 1)
        d = n[None, :] - (np.arange(_PHASES + 1)/_PHASES)[:, None]
        beta = taps/2 # Kaiser over |d| <= taps/2: 8 taps -40 dB or better to 0.34 cycles a sample (2 sps, roll-off 0.35), 12 taps -57 dB
        w = np.i0(beta*np.sqrt(np.clip(1 - (d/(taps/2))**2, 0, None)))/np.i0(beta)
        _TABLES[taps] = (np.sinc(d)*w, n)
    return _TABLES[taps]

def doubled(x, a, b, base=0):
    """x at a, a + 1/2, a + 1, ... up to b - REACH (its positions are absolute: x[0] is position `base`), band limited (the
    12-tap windowed sinc at a half sample: one fixed FIR for the half samples). Before x's start it reads zeros"""
    tab, n = _sinc_table(2*REACH)
    h = tab[_PHASES//2] # x(i + 1/2) = sum_n h[n] x[i + n], n = 1-REACH .. REACH
    lo = a + n[0]
    x = np.asarray(x)
    pre = max(0, base - lo)
    ext = x[max(lo - base, 0):b - base]
    if pre:
        ext = np.concatenate((np.zeros(pre, x.dtype), ext))
    half = np.convolve(ext, h[::-1], "valid") # len b - a - REACH
    K = len(half)
    out = np.empty(2*K, np.result_type(x.dtype, np.complex128) if x.dtype.kind == "c" else np.float64)
    out[0::2] = ext[-n[0]:-n[0] + K]
    out[1::2] = half
    return out

def fractional_edge(x, t, taps=4):
    """`fractional` where t may be near either end of x: the samples beyond them are zeros (a stream's start or its
    flushed end, as a filter's history is)"""
    t = np.asarray(t, dtype=np.float64)
    if not len(t) or (np.floor(t.min()) - REACH + 1 >= 0 and np.floor(t.max()) + REACH < len(x)):
        return fractional(x, t, taps) # (within x: nothing to pad)
    return fractional(np.pad(np.asarray(x), REACH), t + REACH, taps)

def fractional(x, t, taps=4):
    """x (complex or real) at the fractional positions t. taps 4: cubic Lagrange through x[i-1 .. i+2], i = floor(t);
    taps 8, 12...: a windowed sinc over x[i+1-taps/2 .. i+taps/2] (a band-limited interpolator: the error of the matched
    filter's output at 2 samples per symbol is -44 dB with 8 taps and -56 dB with 12, against -26 dB cubic)"""
    t = np.asarray(t, dtype=np.float64)
    i = np.floor(t).astype(np.int64)
    u = t - i
    if taps == 4:
        c = (-u*(u - 1)*(u - 2)/6, (u + 1)*(u - 1)*(u - 2)/2, -(u + 1)*u*(u - 2)/2, (u + 1)*u*(u - 1)/6)
        return c[0]*x[i - 1] + c[1]*x[i] + c[2]*x[i + 1] + c[3]*x[i + 2]
    if taps//2 > REACH or taps % 2:
        raise ValueError(f"taps is 4 or an even number up to {2*REACH}, not {taps}")
    tab, n = _sinc_table(taps)
    from .. import _accel
    if _accel.native is not None and np.asarray(x).dtype == np.complex64: # Two short dot products a position, in C++
        return _accel.native.fractional_read(x, t, tab, int(n[0]))
    p = u*_PHASES
    k = np.minimum(np.floor(p).astype(np.int64), _PHASES - 1)
    a = (p - k)[:, None]
    h = (1 - a)*tab[k] + a*tab[k + 1]
    return np.sum(h*np.asarray(x)[i[:, None] + n[None, :]], axis=1)
