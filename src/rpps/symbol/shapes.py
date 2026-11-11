"""Constellation generators: points (unit mean power) and the labels that go with them

A label list is in the form `Mapping` uses: labels[point] = the value (bits, first bit most significant) it carries.
"""
import numpy as np

def gray(n):
    """Gray code of integers"""
    n = np.asarray(n)
    return n ^ (n >> 1)

def _unit_power(points):
    points = np.asarray(points, dtype=np.complex128)
    return points / np.sqrt(np.mean(np.abs(points)**2))

def psk(M, offset=None):
    """M points round the unit circle, counter-clockwise from `offset` (default: pi/M, so QPSK sits at 45 degrees)
    Labels: Gray, going round the circle"""
    offset = np.pi/M if offset is None else offset
    pts = np.exp(1j*(offset + 2*np.pi*np.arange(M)/M))
    return pts, gray(np.arange(M))

def ask(M, bipolar=True):
    """M amplitudes, evenly spaced, on the real axis (bipolar: -1..1 like PAM, otherwise 0..1 like OOK for M=2). Labels: Gray"""
    lev = np.arange(M)
    pts = (2*lev - (M-1)) if bipolar else lev
    return _unit_power(pts), gray(lev)

def qam(M):
    """Square (4, 16, 64, 256...) or cross (32, 128, 512...) QAM, labels Gray per axis"""
    k = int(np.log2(M))
    if 2**k != M or k < 2:
        raise ValueError(f"QAM needs M = 2**k with k >= 2, got {M}")
    if k % 2 == 0:
        L = 2**(k//2)
        i, q = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        pts = (2*i - (L-1)) + 1j*(2*q - (L-1))
        lab = (gray(i) << (k//2)) | gray(q)
        return _unit_power(pts.reshape(-1)), lab.reshape(-1)
    # Cross: a square of 6c*6c points without its four c*c corner blocks. No labelling is Gray everywhere, so start from
    # the plain order and swap labels while that cuts the bit differences between nearest neighbours (fixed seed)
    c = 2**((k-5)//2) # Corner block side
    L = 6*c # Side of the square
    pts = np.array([(2*a - (L-1)) + 1j*(2*b - (L-1)) for a in range(L) for b in range(L)
                    if not ((a < c or a >= L-c) and (b < c or b >= L-c))])
    return _unit_power(pts), _best_labels(pts, k)

def _best_labels(pts, k):
    M = len(pts)
    d = np.abs(pts[:, None] - pts[None, :])
    np.fill_diagonal(d, np.inf)
    near = np.argwhere(np.abs(d - d.min()) < 1e-9) # Nearest-neighbour pairs
    lab = np.arange(M)
    pop = np.array([bin(i).count("1") for i in range(2**k)])
    def cost(l):
        return pop[l[near[:, 0]] ^ l[near[:, 1]]].sum()
    best = cost(lab)
    rng = np.random.default_rng(0)
    cur, temp = best, 0.4
    keep = lab.copy()
    steps = 60*M if M <= 128 else 30*M
    for n in range(steps): # Annealing: worse swaps are accepted early on
        t = temp*(1 - n/steps)
        i, j = rng.integers(0, M, 2)
        lab[[i, j]] = lab[[j, i]]
        c = cost(lab)
        if c <= cur or rng.random() < np.exp((cur - c)/max(t, 1e-3)):
            cur = c
            if c < best:
                best, keep = c, lab.copy()
        else:
            lab[[i, j]] = lab[[j, i]]
    lab = keep
    return lab

# DVB-S2 radius ratios (outer/inner) by code rate would go here; these are a common mid-rate choice
_APSK_RATIO = {(4, 12): 3.15, (4, 12, 16): (2.84, 5.27), (8, 8): 2.0}

def apsk(rings=(4, 12), ratios=None, phases=None):
    """Amplitude-phase keying on concentric rings

    rings: points per ring, innermost first, e.g. (4, 12) is 16APSK and (4, 12, 16) is 32APSK
    ratios: radius of each outer ring over the innermost one (default DVB-S2-like)
    Labels: the ring picks the top bits, Gray round the ring for the rest (not the DVB-S2 labelling, which is a config definition)
    """
    rings = tuple(rings)
    M = sum(rings)
    if 2**int(np.log2(M)) != M:
        raise ValueError(f"APSK needs a power of two points, got {M}")
    if ratios is None:
        ratios = _APSK_RATIO.get(rings, tuple(2.0*(i+1) for i in range(len(rings)-1)))
    ratios = (1.0, *np.atleast_1d(ratios))
    pts, lab, base = [], [], 0
    for r, (n, rad) in enumerate(zip(rings, ratios)):
        ph = np.pi/n if phases is None else phases[r]
        pts += list(rad*np.exp(1j*(ph + 2*np.pi*np.arange(n)/n)))
        # Gray round a ring whose size is a power of two, plain order otherwise
        lab += list(base + (gray(np.arange(n)) if 2**int(np.log2(n)) == n else np.arange(n)))
        base += n
    return _unit_power(pts), np.array(lab)
