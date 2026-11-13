"""Parks-McClellan equiripple FIR design (the Remez exchange), numpy only

The amplitude response of a linear-phase FIR is A(w) = Q(w) P(w), P a cosine series of r terms and Q fixed by the type:

    type  taps  symmetry   Q            h
    I     odd   even       1            h[c +- m] =  b[m] / 2
    II    even  even       cos(w/2)     h[c +- m] =  b[m] / 2
    III   odd   odd        sin(w)       h[c + m]  =  s[m] / 2, h[c - m] = -s[m] / 2   (Hilbert, differentiator)
    IV    even  odd        sin(w/2)     as III

with c = (taps - 1) / 2. Dividing the desired response by Q and multiplying the weight by it turns every type into the same
weighted approximation of D by P. Each pass of the exchange fits P through r points of the current extremal set with
the error alternating in sign, then moves the extremals to the extrema of the weighted error on a dense grid. The error equioscillates at the optimum.
"""
import numpy as np

_DENSITY = 16

def _check(taps, bands, desired, weight, kind):
    if kind not in ("bandpass", "hilbert", "differentiator"):
        raise ValueError(f"remez: type must be bandpass, hilbert or differentiator, not {kind!r}")
    taps = int(taps)
    if taps < 3:
        raise ValueError("remez: needs at least 3 taps")
    bands = np.asarray(bands, np.float64).ravel()
    desired = np.asarray(desired, np.float64).ravel()
    if bands.size % 2 or bands.size == 0:
        raise ValueError("remez: bands is a list of (low, high) edges, an even count")
    nb = bands.size // 2
    if desired.size != nb:
        raise ValueError(f"remez: desired needs one value per band ({nb}), not {desired.size}")
    weight = np.ones(nb) if weight is None else np.asarray(weight, np.float64).ravel()
    if weight.size != nb:
        raise ValueError(f"remez: weight needs one value per band ({nb}), not {weight.size}")
    if np.any(np.diff(bands) < 0) or np.any(bands < 0) or np.any(bands > 0.5):
        raise ValueError("remez: bands must rise within [0, fs/2]")
    if np.any(weight <= 0):
        raise ValueError("remez: weights must be positive")
    return taps, bands.reshape(nb, 2), desired, weight

def _q(w, sym, even):
    """Q(w) of the filter type"""
    if sym:
        return np.cos(w/2) if even else np.ones_like(w)
    return np.sin(w/2) if even else np.sin(w)

def _grid(bands, desired, weight, r, kind, sym, even, density):
    """Dense frequency grid over the bands (fractions of fs), with the desired response and weight of P (Q divided out)"""
    delf = 0.5/(density*r)
    zero_lo = not sym                            # Q = sin(w) or sin(w/2) is zero at 0
    zero_hi = (sym and even) or (not sym and not even) # cos(w/2) and sin(w) are zero at fs/2
    fs_, ds_, ws_ = [], [], []
    for (lo, hi), d, wt in zip(bands, desired, weight):
        n = max(int(np.ceil((hi - lo)/delf)), 2)
        f = np.linspace(lo, hi, n + 1)
        if zero_lo and f[0] == 0: # Where Q is zero its division is undefined: step in by a fraction of the spacing
            f[0] = (f[1] - f[0])/2
        if zero_hi and f[-1] == 0.5:
            f[-1] = 0.5 - (f[-1] - f[-2])/2
        fs_.append(f)
        if kind == "differentiator":
            ds_.append(d*f)
            ws_.append(wt/f if d != 0 else np.full_like(f, wt))
        else:
            ds_.append(np.full_like(f, d))
            ws_.append(np.full_like(f, wt))
    f = np.concatenate(fs_)
    seg = np.concatenate([[0], np.cumsum([len(x) for x in fs_])])
    q = _q(2*np.pi*f, sym, even)
    return f, np.concatenate(ds_)/q, np.concatenate(ws_)*q, seg

def _log_weights(x):
    """Barycentric weights 1/prod(x_k - x_j) as sign and log magnitude (the product over hundreds of nodes over/underflows)"""
    diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, 1.0)
    sign = np.prod(np.sign(diff), axis=1)
    return sign, -np.sum(np.log(np.abs(diff)), axis=1)

def _fit(x_ext, D_ext, W_ext):
    """Deviation and the values of P at the first r extremals, from r+1 extremals"""
    r1 = len(x_ext)
    sgn, lg = _log_weights(x_ext)
    a = sgn*np.exp(lg - lg.max())
    alt = (-1.0)**np.arange(r1)
    delta = np.sum(a*D_ext)/np.sum(a*alt/W_ext)
    y = D_ext - alt*delta/W_ext
    return delta, y[:-1]

def _eval(x, x_nodes, y):
    """P at x (cos w), barycentric through the nodes"""
    sgn, lg = _log_weights(x_nodes)
    beta = sgn*np.exp(lg - lg.max())
    dx = x[:, None] - x_nodes[None, :]
    hit = np.abs(dx) < 1e-14
    dx[hit] = 1.0
    t = beta[None, :]/dx
    out = (t @ y)/np.sum(t, axis=1)
    row = hit.any(axis=1)
    if row.any():
        out[row] = y[hit[row].argmax(axis=1)]
    return out

def _extrema(E, seg):
    """Index of the largest |E| in each run of one sign (inside each band, the band ends included). A run next to a sign change can
    peak at its end, where the error is steep, so a local maximum test would miss it"""
    out = []
    for a, b in zip(seg[:-1], seg[1:]):
        pos = E[a:b] > 0
        cuts = np.nonzero(pos[1:] != pos[:-1])[0] + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, b - a]):
            i = a + lo + int(np.argmax(np.abs(E[a + lo:a + hi])))
            if out and (E[out[-1]] > 0) == (E[i] > 0): # Same sign across a gap between bands: keep the larger
                if abs(E[i]) > abs(E[out[-1]]):
                    out[-1] = i
            else:
                out.append(i)
    return out

def _reduce(idx, E, count):
    """Drop the weaker end of the set until `count` extremals are left"""
    idx = list(idx)
    while len(idx) > count:
        if abs(E[idx[0]]) < abs(E[idx[-1]]):
            idx.pop(0)
        else:
            idx.pop()
    return idx

def remez(taps, bands, desired, weight=None, fs=1.0, type="bandpass", maxiter=100, density=_DENSITY):
    """Equiripple FIR of `taps` coefficients, optimal in the weighted Chebyshev sense: the maximum error is the smallest possible

    bands    edges of the bands that count, in Hz of sample rate `fs`, rising: [0, 1000, 1500, 5000] is two bands. The gaps are don't-care
    desired  one gain per band (`bandpass`, `hilbert`), or one slope per band (`differentiator`)
    weight   one weight per band, default all 1. A heavier band is held to a smaller error
    type     "bandpass" (a filter; the default), "hilbert" or "differentiator" (odd symmetry)

    Returns the float64 coefficients. Raises ValueError for an impossible specification and RuntimeError when the exchange does not converge
    """
    taps, bands, desired, weight = _check(taps, np.asarray(bands, np.float64)/fs, desired, weight, type)
    sym = type == "bandpass"
    even = taps % 2 == 0
    r = (taps + 1)//2 if sym and not even else taps//2 if even else (taps - 1)//2 # terms of P: (N+1)/2 (I), N/2 (II, IV), (N-1)/2 (III)
    f, D, W, seg = _grid(bands, desired, weight, r, type, sym, even, density)
    if len(f) < r + 1:
        raise ValueError("remez: too many taps for the width of the bands")
    x = np.cos(2*np.pi*f)
    ext = np.round(np.linspace(0, len(f) - 1, r + 1)).astype(int)
    for _ in range(maxiter):
        delta, y = _fit(x[ext], D[ext], W[ext])
        A = _eval(x, x[ext][:-1], y)
        E = W*(D - A)
        new = _extrema(E, seg)
        if len(new) < r + 1:
            raise RuntimeError("remez: failure to converge (lost extremals): the ripple is below what float64 resolves, try a wider transition band or fewer taps")
        new = _reduce(new, E, r + 1)
        e = np.abs(E[new])
        if np.array_equal(new, ext) or (e.max() - e.min()) <= 1e-9*e.max():
            ext = np.array(new)
            delta, y = _fit(x[ext], D[ext], W[ext])
            break
        ext = np.array(new)
    else:
        raise RuntimeError(f"remez: no convergence in {maxiter} iterations")
    # P is a cosine series of r terms: recover it from r samples at the Chebyshev points (exact for that degree)
    wj = np.pi*(np.arange(r) + 0.5)/r
    P = _eval(np.cos(wj), x[ext][:-1], y)
    c = (2.0/r)*np.cos(np.outer(np.arange(r), wj)) @ P
    c[0] /= 2
    h = _taps(c, taps, sym, even)
    return h if sym else -h # antisymmetric: A > 0 is a positive impulse response after the centre (Hilbert), as scipy

def _taps(c, n, sym, even):
    """Impulse response of A = Q P with P = sum c[k] cos(k w)"""
    h = np.zeros(n)
    mid = (n - 1)/2
    r = len(c)
    if sym and not even: # type I
        m = np.arange(r)
        h[int(mid) + m] = c/np.where(m == 0, 1.0, 2.0)
        h[int(mid) - m] = h[int(mid) + m]
        return h
    if sym: # type II: cos(w/2) cos(kw) = [cos((k+1/2)w) + cos((k-1/2)w)]/2
        b = np.zeros(r + 1) # b[j]: coefficient of cos((j - 1/2) w), j = 1..r
        b[1] += c[0]
        for k in range(1, r):
            b[k + 1] += c[k]/2
            b[k] += c[k]/2
        j = np.arange(1, r + 1)
        h[n//2 + j - 1] = b[1:]/2
        h[n//2 - j] = b[1:]/2
        return h
    if not even: # type III: sin(w) cos(kw) = [sin((k+1)w) - sin((k-1)w)]/2
        s = np.zeros(r + 1)
        s[1] += c[0]
        for k in range(1, r):
            s[k + 1] += c[k]/2
            s[k - 1] -= c[k]/2
        m = np.arange(1, r + 1)
        h[int(mid) + m] = s[1:]/2
        h[int(mid) - m] = -s[1:]/2
        return h
    # type IV: sin(w/2) cos(kw) = [sin((k+1/2)w) - sin((k-1/2)w)]/2
    s = np.zeros(r + 1)
    s[1] += c[0]
    for k in range(1, r):
        s[k + 1] += c[k]/2
        s[k] -= c[k]/2
    j = np.arange(1, r + 1)
    h[n//2 + j - 1] = s[1:]/2
    h[n//2 - j] = -s[1:]/2
    return h
