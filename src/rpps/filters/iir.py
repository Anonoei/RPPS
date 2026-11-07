"""IIR filters, with numpy only

An IIR is a cascade of second-order sections, one row `[b0 b1 b2 a0 a1 a2]` each:
    H(z) = (b0 + b1 z^-1 + b2 z^-2) / (a0 + a1 z^-1 + a2 z^-2)
`IIR(sos)` filters a stream with them. The design helpers here make the sections from a specification:
`butter`, `cheby1`, `cheby2`, the RBJ `biquad`s, `single_pole`, and `zpk2sos` / `tf2sos` for coefficients from elsewhere.
Frequencies are fractions of the sample rate (0.5 is Nyquist).

Filtering is exact but not sample by sample: a block of up to BLOCK samples is one matrix product per section
(the section as a state space system, whose impulse-response matrix and state transition are precomputed), so it runs at
BLAS speed and any block length works. An IIR has no constant group delay, so a stream's `delay` is not changed by it.
"""
from types import SimpleNamespace
import warnings

import numpy as np

from .filter import Filter

BLOCK = 256 # Samples handled by one matrix product
MAX_ORDER_TF = 8 # Above this, a transfer function (b, a) converts badly: use sections or zpk

# ---- Sections

def as_sos(sos):
    """Sections as a float (n, 6) array normalised to a0 = 1"""
    sos = np.atleast_2d(np.asarray(sos, dtype=float))
    if sos.ndim != 2 or sos.shape[1] != 6 or len(sos) == 0:
        raise ValueError(f"sos is a list of rows [b0 b1 b2 a0 a1 a2], got shape {sos.shape}")
    if np.any(sos[:, 3] == 0):
        raise ValueError("a section has a0 = 0")
    return sos / sos[:, 3:4]

def poles(sos):
    """Every pole of the cascade (a first-order section has one: its unused second pole at the origin is not listed)"""
    sos = as_sos(sos)
    return np.concatenate([np.roots(np.trim_zeros([1.0, r[4], r[5]], "b")) for r in sos])

def sos2zpk(sos):
    """(zeros, poles, gain) of a cascade. Zeros at infinity (a section whose b0 is 0) are left out"""
    sos = as_sos(sos)
    z, p, k = [], [], 1.0
    for r in sos:
        b = np.trim_zeros(r[:3], "f")
        k *= b[0] if len(b) else 0.0
        z.append(np.roots(b) if len(b) > 1 else np.zeros(0))
        p.append(np.roots([1.0, r[4], r[5]]))
    return np.concatenate(z), np.concatenate(p), k

class _Section:
    """One biquad as a state space system (transposed direct form II), with the matrices for processing BLOCK samples at once"""
    def __init__(self, row, L=BLOCK):
        b0, b1, b2, _, a1, a2 = row
        A = np.array([[-a1, 1.0], [-a2, 0.0]])
        B = np.array([b1 - a1*b0, b2 - a2*b0])
        P = np.empty((L + 1, 2, 2))   # A^k
        P[0] = np.eye(2)
        for k in range(1, L + 1):
            P[k] = P[k-1] @ A
        PB = P @ B                    # A^k B
        h = np.empty(L)               # Impulse response
        h[0] = b0
        h[1:] = PB[:L-1, 0]
        i = np.subtract.outer(np.arange(L), np.arange(L))
        self.L, self.P, self.PB = L, P, PB
        self.T = np.where(i >= 0, h[np.clip(i, 0, None)], 0.0)  # y = T x for zero state
        self.O = P[:L, 0, :]                                    # y = O z for zero input

    def process(self, X, Z):
        """X: (n, c) input columns (1: real, 2: real and imaginary), Z: (2, c) state. Returns (Y, new Z)"""
        L, out = self.L, np.empty_like(X)
        for s in range(0, len(X), L):
            x = X[s:s+L]
            m = len(x)
            out[s:s+m] = self.T[:m, :m] @ x + self.O[:m] @ Z
            Z = self.P[m] @ Z + self.PB[:m][::-1].T @ x
        return out, Z

class IIR(Filter):
    """Cascade of second-order sections `sos` (rows [b0 b1 b2 a0 a1 a2]); `gain` scales the output

    The state carries across blocks, so any block length (including 0) works. Real or complex samples: the coefficients are real
    and act on both parts. Poles must be inside or on the unit circle (an integrator is allowed)."""
    def __init__(self, sos, gain=1.0):
        self.sos = as_sos(sos)
        self.sos[0, :3] *= gain
        if len(poles(self.sos)) and np.max(np.abs(poles(self.sos))) > 1 + 1e-9:
            raise ValueError(f"unstable IIR: a pole has radius {np.max(np.abs(poles(self.sos))):.6g}")
        self._sections = [_Section(r) for r in self.sos]

    def __str__(self):
        return f"IIR:{len(self.sos)}"

    def new_state(self, meta):
        return SimpleNamespace(z=[np.zeros((2, 2)) for _ in self._sections])

    def _run(self, samples, zs):
        x = np.asarray(samples)
        if len(x) == 0:
            return x
        cplx = x.dtype.kind == "c"
        X = np.stack([x.real, x.imag], axis=1).astype(float) if cplx else x.astype(float)[:, None]
        c = X.shape[1]
        for i, sec in enumerate(self._sections):
            X, z = sec.process(X, zs[i][:, :c])
            zs[i][:, :c] = z
        return X[:, 0] + 1j*X[:, 1] if cplx else X[:, 0]

    def step(self, samples, st, meta):
        return self._run(samples, st.z)

    def run(self, samples):
        """One-shot (stateless) filtering from zero state"""
        return self._run(samples, self.new_state(None).z)

    def response(self, n=512):
        """(f, H): frequency response at n points from 0 to Nyquist (f in cycles per sample)"""
        f = np.linspace(0, 0.5, n)
        zi = np.exp(-2j*np.pi*f)
        H = np.ones(n, dtype=complex)
        for b0, b1, b2, _, a1, a2 in self.sos:
            H *= (b0 + b1*zi + b2*zi**2) / (1 + a1*zi + a2*zi**2)
        return f, H

# ---- Poles and zeros to sections

def _split(roots, tol=1e-7):
    """Conjugate pairs (the one with positive imaginary part) and real roots of a root list"""
    r = np.asarray(roots, dtype=complex)
    real = np.abs(r.imag) <= tol*np.maximum(1.0, np.abs(r))
    pos, neg = r[~real & (r.imag > 0)], r[~real & (r.imag < 0)]
    if len(pos) != len(neg):
        raise ValueError("complex roots must come in conjugate pairs")
    return list(pos), sorted(r[real].real, key=abs)

def _pairs(roots):
    """Roots grouped for sections: [(r1, r2)] with r2 = conj(r1), two reals, or None for a last single real"""
    cplx, real = _split(roots)
    out = [(c, np.conj(c)) for c in cplx]
    out += [(real[i], real[i+1]) for i in range(0, len(real) - 1, 2)]
    if len(real) % 2:
        out.append((real[-1], None))
    return out

def _poly(pair):
    """[1, c1, c2] of (1 - r1 z^-1)(1 - r2 z^-1)"""
    r1, r2 = pair
    if r2 is None:
        return np.array([1.0, -np.real(r1), 0.0])
    return np.array([1.0, -np.real(r1 + r2), np.real(r1*r2)])

def _delay_rows(n):
    return [[0, 0, 1, 1, 0, 0]]*(n // 2) + ([[0, 1, 0, 1, 0, 0]] if n % 2 else [])

def zpk2sos(z, p, k=1.0):
    """Sections for a filter of zeros z, poles p and gain k (z-plane, conjugates in pairs). Fewer zeros than poles means delays"""
    z, p = np.asarray(z, dtype=complex), np.asarray(p, dtype=complex)
    if len(z) > len(p):
        raise ValueError(f"{len(z)} zeros and {len(p)} poles: the filter would not be causal")
    delay = len(p) - len(z)
    if len(p) == 0:
        return as_sos([[k, 0, 0, 1, 0, 0]])
    zp = _pairs(np.concatenate((z, np.zeros(delay))))
    pp = sorted(_pairs(p), key=lambda q: abs(q[0]))
    rows = []
    for pair in reversed(pp): # The poles nearest the unit circle pick their zeros first
        j = min(range(len(zp)), key=lambda i: min(abs(a - b) for a in pair if a is not None for b in zp[i] if b is not None))
        rows.append((pair, zp.pop(j)))
    rows.reverse()
    n = len(rows)
    g = abs(k)**(1.0/n) if k != 0 else 0.0
    sos = [np.concatenate((_poly(zero)*g*(np.sign(k) if i == 0 and k != 0 else 1.0), _poly(pole))) for i, (pole, zero) in enumerate(rows)]
    return as_sos(sos + _delay_rows(delay))

def tf2sos(b, a=(1.0,)):
    """Sections for the transfer function b(z^-1) / a(z^-1) (coefficients of rising powers of z^-1, as scipy.signal)"""
    b, a = np.atleast_1d(np.asarray(b, dtype=float)), np.atleast_1d(np.asarray(a, dtype=float))
    if len(a) == 0 or a[0] == 0:
        raise ValueError("a[0] must not be 0")
    if not np.any(b):
        raise ValueError("all numerator coefficients are 0")
    delay = len(b) - len(np.trim_zeros(b, "f")) # Leading zeros of b are a pure delay
    b = np.trim_zeros(b, "f")
    n = max(len(a), len(b)) - 1
    if n > MAX_ORDER_TF:
        warnings.warn(f"order {n} transfer function: converting to sections loses accuracy, give sos or zpk instead")
    b, a = np.pad(b, (0, n + 1 - len(b))), np.pad(a, (0, n + 1 - len(a)))
    if n == 0:
        sos = as_sos([[b[0]/a[0], 0, 0, 1, 0, 0]])
    else:
        sos = zpk2sos(np.roots(b), np.roots(a), b[0]/a[0])
    return as_sos(np.vstack((sos, _delay_rows(delay)))) if delay else sos

# ---- Designs

def _proto(kind, n, ripple, atten):
    """Analog lowpass prototype of cutoff 1 rad/s: (z, p, k)"""
    m = np.arange(-n + 1, n, 2)
    if kind == "butter":
        return np.zeros(0), -np.exp(1j*np.pi*m/(2*n)), 1.0
    if kind == "cheby1":
        eps = np.sqrt(10**(0.1*ripple) - 1)
        mu = np.arcsinh(1/eps)/n
        p = -np.sinh(mu + 1j*np.pi*m/(2*n))
        k = np.prod(-p).real
        return np.zeros(0), p, k/np.sqrt(1 + eps**2) if n % 2 == 0 else k
    de = 1/np.sqrt(10**(0.1*atten) - 1) # cheby2
    mu = np.arcsinh(1/de)/n
    zm = np.concatenate((np.arange(-n + 1, 0, 2), np.arange(2, n, 2))) if n % 2 else m # Odd order: no zero at infinity
    z = -np.conj(1j/np.sin(zm*np.pi/(2*n)))
    p = -np.exp(1j*np.pi*m/(2*n))
    p = 1/(np.sinh(mu)*p.real + 1j*np.cosh(mu)*p.imag)
    return z, p, (np.prod(-p)/np.prod(-z)).real

def _lp2lp(z, p, k, wo):
    return z*wo, p*wo, k*wo**(len(p) - len(z))

def _lp2hp(z, p, k, wo):
    d = len(p) - len(z)
    return np.concatenate((wo/z, np.zeros(d))), wo/p, k*(np.prod(-z)/np.prod(-p)).real

def _lp2bp(z, p, k, wo, bw):
    d = len(p) - len(z)
    zl, pl = z*bw/2, p*bw/2
    zb = np.concatenate((zl + np.sqrt(zl**2 - wo**2 + 0j), zl - np.sqrt(zl**2 - wo**2 + 0j), np.zeros(d)))
    pb = np.concatenate((pl + np.sqrt(pl**2 - wo**2 + 0j), pl - np.sqrt(pl**2 - wo**2 + 0j)))
    return zb, pb, k*bw**d

def _lp2bs(z, p, k, wo, bw):
    d = len(p) - len(z)
    zh, ph = (bw/2)/z, (bw/2)/p
    zb = np.concatenate((zh + np.sqrt(zh**2 - wo**2 + 0j), zh - np.sqrt(zh**2 - wo**2 + 0j), np.full(d, 1j*wo), np.full(d, -1j*wo)))
    pb = np.concatenate((ph + np.sqrt(ph**2 - wo**2 + 0j), ph - np.sqrt(ph**2 - wo**2 + 0j)))
    return zb, pb, k*(np.prod(-z)/np.prod(-p)).real

def _bilinear(z, p, k):
    """s = 2 (z-1)/(z+1)"""
    d = len(p) - len(z)
    return np.concatenate(((2 + z)/(2 - z), -np.ones(d))), (2 + p)/(2 - p), k*(np.prod(2 - z)/np.prod(2 - p)).real

PASSES = ("lowpass", "highpass", "bandpass", "bandstop")

def design(kind, order, pass_, cutoff, ripple_db=None, atten_db=None):
    """Sections of a Butterworth ("butter") or Chebyshev ("cheby1": ripple_db in the pass band, "cheby2": atten_db in the stop band) filter

    cutoff: the edge as a fraction of the sample rate (the -3 dB point of a butter; the ripple edge of cheby1; the stop edge of cheby2),
    or [low, high] for bandpass and bandstop."""
    if kind not in ("butter", "cheby1", "cheby2"):
        raise ValueError(f"unknown design {kind!r}")
    if pass_ not in PASSES:
        raise ValueError(f"pass must be one of {PASSES}, not {pass_!r}")
    if int(order) != order or not 1 <= order <= 40:
        raise ValueError(f"order must be a whole number from 1 to 40, not {order}")
    if kind == "cheby1" and not (ripple_db and ripple_db > 0):
        raise ValueError("cheby1 needs ripple_db > 0")
    if kind == "cheby2" and not (atten_db and atten_db > 0):
        raise ValueError("cheby2 needs atten_db > 0")
    band = pass_ in ("bandpass", "bandstop")
    edges = np.atleast_1d(np.asarray(cutoff, dtype=float))
    if len(edges) != (2 if band else 1):
        raise ValueError(f"a {pass_} needs {'[low, high]' if band else 'one cutoff'}")
    if np.any(edges <= 0) or np.any(edges >= 0.5) or (band and edges[0] >= edges[1]):
        raise ValueError(f"cutoff must lie in (0, 0.5) of the sample rate, and low < high: {cutoff}")
    z, p, k = _proto(kind, int(order), ripple_db, atten_db)
    w = 2*np.tan(np.pi*edges)
    if pass_ == "lowpass":
        z, p, k = _lp2lp(z, p, k, w[0])
    elif pass_ == "highpass":
        z, p, k = _lp2hp(z, p, k, w[0])
    else:
        wo, bw = np.sqrt(w[0]*w[1]), w[1] - w[0]
        z, p, k = (_lp2bp if pass_ == "bandpass" else _lp2bs)(z, p, k, wo, bw)
    return zpk2sos(*_bilinear(z, p, k))

def single_pole(alpha):
    """y[n] = alpha x[n] + (1 - alpha) y[n-1]: a leaky integrator / exponential average (0 < alpha <= 1)"""
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], not {alpha}")
    return as_sos([[alpha, 0, 0, 1, -(1 - alpha), 0]])

BIQUADS = ("lowpass", "highpass", "bandpass", "notch", "allpass", "peaking", "lowshelf", "highshelf")

def biquad(shape, f0, q=0.7071067811865476, gain_db=0.0):
    """One RBJ audio-EQ cookbook biquad. f0: centre/corner as a fraction of the sample rate; q: quality factor;
    gain_db: for peaking and the shelves"""
    if shape not in BIQUADS:
        raise ValueError(f"biquad shape must be one of {BIQUADS}, not {shape!r}")
    if not 0 < f0 < 0.5:
        raise ValueError(f"f0 must lie in (0, 0.5) of the sample rate, not {f0}")
    if q <= 0:
        raise ValueError("q must be positive")
    w0 = 2*np.pi*f0
    c, alpha, A = np.cos(w0), np.sin(w0)/(2*q), 10**(gain_db/40)
    sa = 2*np.sqrt(A)*alpha
    if shape == "lowpass":
        row = [(1 - c)/2, 1 - c, (1 - c)/2, 1 + alpha, -2*c, 1 - alpha]
    elif shape == "highpass":
        row = [(1 + c)/2, -(1 + c), (1 + c)/2, 1 + alpha, -2*c, 1 - alpha]
    elif shape == "bandpass":
        row = [alpha, 0, -alpha, 1 + alpha, -2*c, 1 - alpha]
    elif shape == "notch":
        row = [1, -2*c, 1, 1 + alpha, -2*c, 1 - alpha]
    elif shape == "allpass":
        row = [1 - alpha, -2*c, 1 + alpha, 1 + alpha, -2*c, 1 - alpha]
    elif shape == "peaking":
        row = [1 + alpha*A, -2*c, 1 - alpha*A, 1 + alpha/A, -2*c, 1 - alpha/A]
    elif shape == "lowshelf":
        row = [A*((A + 1) - (A - 1)*c + sa), 2*A*((A - 1) - (A + 1)*c), A*((A + 1) - (A - 1)*c - sa),
               (A + 1) + (A - 1)*c + sa, -2*((A - 1) + (A + 1)*c), (A + 1) + (A - 1)*c - sa]
    else:
        row = [A*((A + 1) + (A - 1)*c + sa), -2*A*((A - 1) + (A + 1)*c), A*((A + 1) + (A - 1)*c - sa),
               (A + 1) - (A - 1)*c + sa, 2*((A - 1) - (A + 1)*c), (A + 1) - (A - 1)*c - sa]
    return as_sos([row])
