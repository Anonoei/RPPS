"""Whether hard decisions from locked symbols can be trusted for blind tests (`gate`); the Es/N0 it uses is
rpps.sync.quality.measure"""
import numpy as np

from ...sync.quality import measure

class Gate:
    """Result of `gate`: whether hard decisions from these symbols can be trusted for blind tests (rank, LFSR, code
    search), and why not. rotation: the symbols' mean angle off their decisions (degrees); drift: the largest change of
    that over the blocks (degrees); near: fraction of decisions within `margin` minimum distances of a boundary, and
    near_expected: the same for Gaussian noise at the measured Es/N0 (more than twice that: something other than noise
    moves the symbols toward the boundaries); ser: expected symbol error rate at the measured Es/N0; quality: `measure`'s;
    displaced: the largest distance of a cluster's mean from its point, in minimum distances (compression, AM/PM, IQ
    imbalance or the wrong constellation: a few hundredths is noise). ok: rotation and drift under `max_turn` of the
    half-angle between points, near under twice expected, displaced under a quarter"""
    __slots__ = ("ok", "why", "rotation", "drift", "near", "near_expected", "ser", "quality", "limit", "displaced")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        head = "trust the decisions" if self.ok else f"do not trust the decisions: {self.why}"
        return (f"{head} (rotation {self.rotation:+.1f} deg, drift {self.drift:.1f} deg, limit {self.limit:.1f}; "
                f"{100*self.near:.1f}% near a boundary vs {100*self.near_expected:.1f}% for noise; expected SER "
                f"{self.ser:.1e}; clusters off their points by up to {self.displaced:.2f} of the minimum distance; "
                f"{self.quality})")

    __repr__ = __str__

def _second(d2):
    """(nearest, second nearest) squared distances per row"""
    p = np.partition(d2, 1, axis=1)
    return p[:, 0], p[:, 1]

def gate(y, points, blocks=16, margin=0.1, max_turn=0.15, seed=0):
    """Check locked symbols before their hard decisions go into a blind test (see Gate). A frame turned by a phase
    reference error, a carrier that slips through a block, a gain or constellation mismatch: each shows as a rotation,
    a drift, or decisions crowding the boundaries, which noise at the measured Es/N0 does not explain"""
    y = np.asarray(y, dtype=np.complex128)
    pts = np.asarray(points, dtype=np.complex128)
    pts = pts/np.sqrt(np.mean(np.abs(pts)**2))
    q = measure(y, pts)
    y = y/np.sqrt(np.mean(np.abs(y)**2))
    for _ in range(3): # The same amplitude fit as measure
        dec = pts[np.argmin(np.abs(y[:, None] - pts[None, :]), axis=1)]
        y = y/(np.sum((y*np.conj(dec)).real)/np.sum(np.abs(dec)**2))
    d2 = np.abs(y[:, None] - pts[None, :])**2
    dec = pts[np.argmin(d2, axis=1)]
    rot = float(np.degrees(np.angle(np.sum(y*np.conj(dec)))))
    n = len(y)
    B = max(1, min(blocks, n//64))
    turns = [np.degrees(np.angle(np.sum(b*np.conj(c)))) for b, c in zip(np.array_split(y, B), np.array_split(dec, B))]
    drift = float(np.max(turns) - np.min(turns)) if B > 1 else 0.0
    dmin = np.min(np.abs(pts[:, None] - pts[None, :]) + np.eye(len(pts))*1e9)
    a, b = _second(d2)
    m = (np.sqrt(b) - np.sqrt(a))/dmin
    near = float(np.mean(m < margin))
    # The same statistics for these points under Gaussian noise at the measured sigma
    rng = np.random.default_rng(seed)
    k = 20000
    s = pts[rng.integers(0, len(pts), k)]
    z = s + q.sigma*(rng.standard_normal(k) + 1j*rng.standard_normal(k))
    dz = np.abs(z[:, None] - pts[None, :])**2
    za, zb = _second(dz)
    near_exp = float(np.mean((np.sqrt(zb) - np.sqrt(za))/dmin < margin))
    ser = float(np.mean(pts[np.argmin(dz, axis=1)] != s))
    # The angle between neighbouring points seen from the origin, for the outermost ring: what a turn may not approach
    r = np.abs(pts)
    outer = pts[r > 0.999*r.max()]
    half = 180.0/len(outer) if len(outer) > 1 else 90.0
    limit = max_turn*half
    why = []
    if abs(rot) > limit:
        why.append(f"turned {rot:+.1f} deg off the points")
    if drift > limit:
        why.append(f"the turn changes by {drift:.1f} deg across the blocks (a slipping or drifting carrier)")
    idx = np.argmin(d2, axis=1)
    off = [abs(np.mean(y[idx == j] - pts[j])) for j in range(len(pts)) if np.count_nonzero(idx == j) >= 8]
    displaced = float(max(off)/dmin) if off else 0.0
    if displaced > 0.25:
        why.append(f"clusters sit up to {displaced:.2f} minimum distances off their points (compression, IQ imbalance, or "
                   "not this constellation)")
    if near > 2*near_exp + 3/np.sqrt(n):
        why.append(f"{100*near:.1f}% of decisions near a boundary, noise explains {100*near_exp:.1f}%")
    return Gate(ok=not why, why="; ".join(why), rotation=rot, drift=drift, near=near, near_expected=near_exp, ser=ser,
                quality=q, limit=limit, displaced=displaced)
