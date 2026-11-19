"""TDMA bursts demodulated one by one: each acquired on the preamble the bursts share, as a TDMA receiver must

    r = rp.analysis.spectrum.burst.demodulate(x, Fs)              # or mod=, baud=, preamble= (symbols) when known
    print(r)       # "12 bursts, qpsk at 100.0 kBd, a 48-symbol preamble; frame 4.615 ms, 8 slots; EVM 6.1% median"
    b = r.bursts[0]
    b.symbols, b.bits, b.cfo, b.evm, b.slot, b.frame   # carrier-, timing- and phase-corrected symbols, decided bits

Every burst comes from another transmitter (or the same one after a gap), with its own carrier offset, phase and symbol
timing, and is too short to train loops on: so each is acquired on its own, data-aided, from the preamble.

  1. rpps.analysis.framing.frames.bursts gates the bursts and acquires each blind (rate, modulation, offset); bursts too short
     for that keep only their span. The modulation is the bursts' majority, the rate their median (or `mod` / `baud`).
  2. The preamble: the leading run of symbols the bursts agree on, as phase steps (rpps.analysis.framing.frames.preamble, or
     `shared_preamble` for a short one, whichever is longer), or `preamble` given. Its symbols are rebuilt from the steps and turned onto the constellation (one rotation for all
     bursts: the M-fold phase ambiguity is the same for every burst, so their bits agree with each other).
  3. In each burst the preamble is found by correlating its shaped waveform in pieces (the pieces' magnitudes added: a
     carrier offset costs little), then the fractional timing that best matches the preamble's phase steps (blind to
     the carrier), then from z_k = y_k conj(p_k) over the preamble: the carrier offset (the mean turn of z_k z*_{k-1})
     and the phase and gain (sum z_k with the turn taken out).
  4. The burst's symbols, from the preamble to the end of the gate, are taken at that timing and corrected; a second-
     order decision-directed loop follows what is left of the carrier through the data; decisions give bits, and the
     error vector magnitude against them.
  5. The TDMA structure: the period of the bursts' starts (rpps.analysis.spectrum.waveform.bursts, else `frame`) and the slots,
     clusters of the starts within a frame (to within `slot_tol` symbols): each burst's frame and slot.
Limits: timing is held over the burst (a burst over a few thousand symbols with a clock off by many ppm drifts); a
burst whose preamble is weaker than the data's noise floor allows is reported with `ok` False and no bits.
"""
import numpy as np

from ..framing import frames

class BurstResult:
    """One burst: start (sample of its preamble's first symbol, fractional), symbols, bits, cfo (Hz), phase (rad), gain,
    score (the preamble's correlation 0..1), evm (rms, of the data after the preamble), frame, slot, ok"""
    __slots__ = ("start", "symbols", "bits", "cfo", "phase", "gain", "score", "evm", "frame", "slot", "ok")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        if not self.ok:
            return f"<burst at {self.start:.0f}: not acquired (preamble score {self.score:.2f})>"
        sl = f", frame {self.frame} slot {self.slot}" if self.slot is not None else ""
        return (f"<burst at {self.start:.1f}: {len(self.symbols)} symbols, cfo {self.cfo:+.1f} Hz, EVM "
                f"{100*self.evm:.1f}%{sl}>")

class Bursts:
    """`demodulate`: bursts [BurstResult], mod, baud, beta, preamble (symbols), frame (s, None), slots (offsets within
    the frame, s)"""
    __slots__ = ("bursts", "mod", "baud", "beta", "preamble", "frame", "slots")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        ok = [b for b in self.bursts if b.ok]
        s = (f"{len(self.bursts)} bursts ({len(ok)} demodulated), {self.mod} at {self.baud/1e3:.3f} kBd, a "
             f"{len(self.preamble)}-symbol preamble")
        if self.frame:
            s += f"; frame {self.frame*1e3:.3f} ms, {len(self.slots)} slots"
        if ok:
            s += f"; EVM {100*np.median([b.evm for b in ok]):.1f}% median"
        return s

    __repr__ = __str__

def shared_preamble(burst_list, search=8, win=12, agree=0.8, max_len=512):
    """The leading run of symbols the bursts share, as phase steps: (length in steps, steps, where it starts in each
    burst's symbols), or (0, None, None). Like rpps.analysis.framing.frames.preamble, but aligned on a short window (`win` steps
    near the start, so a preamble shorter than framing's 64-step window aligns too), each burst against every other's
    start, the reference being the burst the others agree with most"""
    S = [b.symbols for b in burst_list if b.symbols is not None and len(b.symbols) > 2*search + win + 4]
    if len(S) < 3:
        return 0, None, None
    D = [v[1:]*np.conj(v[:-1])/(np.abs(v[1:]*v[:-1]) + 1e-12) for v in S]
    def align(ref):
        offs, score = [], 0.0
        for d in D:
            sc = [np.abs(np.vdot(ref, d[k:k + win])) for k in range(2*search + 1) if k + win <= len(d)]
            offs.append(int(np.argmax(sc)) - search)
            score += max(sc)
        return offs, score
    best = None
    for i in range(min(len(D), 8)):
        offs, score = align(D[i][search:search + win])
        if best is None or score > best[1]:
            best = (offs, score)
    offs = best[0]
    c = max(0, -min(offs))
    L = min(max_len, min(len(d) - o - c for d, o in zip(D, offs)))
    rows = np.array([d[o + c:o + c + L] for d, o in zip(D, offs)])
    m = np.abs(np.mean(rows, axis=0))
    s0 = next((i for i in range(min(len(m), 2*search + 2)) if m[i] > agree), None)
    if s0 is None:
        return 0, None, None
    n = 0
    while s0 + n < len(m) and m[s0 + n] > agree:
        n += 1
    return n, np.mean(rows, axis=0)[s0:s0 + n], [o + c + s0 for o in offs]

def preamble_symbols(steps, points):
    """Symbols of a preamble from its phase steps d[n] = s[n] conj(s[n-1]) (rpps.analysis.framing.frames.preamble), turned so
    they sit on the constellation `points` (one rotation, chosen from the points' own symmetry)"""
    d = np.asarray(steps, dtype=np.complex128)
    d = d/np.maximum(np.abs(d), 1e-12)
    s = np.concatenate(([1.0 + 0j], np.cumprod(d)))
    pts = np.asarray(points, dtype=np.complex128)
    r = np.sqrt(np.mean(np.abs(pts)**2))
    best = None
    for th in np.linspace(0, 2*np.pi, 720, endpoint=False):
        q = s*r*np.exp(1j*th)
        e = np.min(np.abs(q[:, None] - pts[None, :]), axis=1).mean()
        if best is None or e < best[0]:
            best = (e, q)
    q = best[1]
    return pts[np.argmin(np.abs(q[:, None] - pts[None, :]), axis=1)]

def _bits(idx, con):
    """Bits of point indices under the constellation's map (label of point i: its map's entry i, first bit most
    significant)"""
    lab = np.asarray(con.mapping.arr)[idx]
    m = con.bits_per_symbol
    return ((lab[:, None] >> np.arange(m - 1, -1, -1)[None, :]) & 1).astype(np.uint8).reshape(-1)

def _mf(x, centres, sps, beta):
    from ...sync.time.mat import matrix_mf
    return matrix_mf(x, centres, sps, beta, 8)

def acquire(x, Fs, pre, baud, beta=0.35, search=None, steps=16):
    """The preamble `pre` (symbols) in samples x at Fs: (start sample of its first symbol (fractional), score 0..1,
    cfo (rad a symbol), phase, gain), or None when x is too short. search: (lo, hi) samples where the first symbol may be
    (default: anywhere)"""
    x = np.asarray(x, dtype=np.complex128)
    sps = Fs/baud
    n = len(pre)
    span = 8
    hits = frames.find_word(x, Fs, pre, [baud], beta=beta, threshold=0.0, top=4)
    if search is not None:
        hits = [h for h in hits if search[0] <= h.at <= search[1]] or hits
    if not hits:
        return None
    at = hits[0].at
    dp = pre[1:]*np.conj(pre[:-1])
    best = None
    for f in np.arange(-steps//2, steps//2 + 1)/steps:
        t0 = at + f*sps
        c = t0 + sps*np.arange(n)
        if c[0] - span*sps < 0 or c[-1] + span*sps + 2 >= len(x):
            continue
        y = _mf(x, c, sps, beta)
        m = np.abs(np.sum(y[1:]*np.conj(y[:-1])*np.conj(dp)))/(np.sum(np.abs(y)**2) + 1e-30)
        if best is None or m > best[0]:
            best = (m, t0, y)
    if best is None:
        return None
    _, t0, y = best
    z = y*np.conj(pre)
    w = float(np.angle(np.sum(z[1:]*np.conj(z[:-1])))) # rad a symbol
    zz = z*np.exp(-1j*w*np.arange(n))
    c = np.sum(zz)
    phase = float(np.angle(c))
    gain = float(np.abs(c)/np.sum(np.abs(pre)**2))
    score = float(np.abs(c)/np.sqrt(np.sum(np.abs(y)**2)*np.sum(np.abs(pre)**2)))
    return t0, score, w, phase, gain

def track(y, points, k1=0.05, k2=0.002):
    """A second-order decision-directed phase loop over corrected symbols y (one a symbol): the symbols with what was
    left of the carrier taken off"""
    out = np.empty_like(y)
    ph = nu = 0.0
    for i, v in enumerate(y):
        r = v*np.exp(-1j*ph)
        d = points[np.argmin(np.abs(points - r))]
        e = np.angle(r*np.conj(d))
        out[i] = r
        nu += k2*e
        ph += k1*e + nu
    return out

def demodulate(x, Fs, mod=None, baud=None, beta=None, preamble=None, frame=None, slot_tol=4, min_score=0.5, **kw):
    """Every burst of x, demodulated on its own from the shared preamble (see the module). kw: frames.gate's"""
    from ... import symbol as rsym
    x = np.asarray(x, dtype=np.complex128)
    B = frames.bursts(x, Fs, baud=baud, **kw)
    acq = [b for b in B if b.det is not None]
    if mod is None:
        if not acq:
            raise ValueError("no burst long enough to tell the modulation: give mod")
        mods = [b.det.mod for b in acq]
        mod = max(set(mods), key=mods.count)
    if baud is None:
        if not acq:
            raise ValueError("no burst long enough to measure the rate: give baud")
        baud = float(np.median([b.det.baud for b in acq]))
    if beta is None:
        beta = float(np.median([b.det.beta for b in acq])) if acq else 0.35
    con = rsym.load(mod)
    pts = np.asarray(con.points, dtype=np.complex128)
    if preamble is None:
        n, d, _ = frames.preamble(B)
        n2, d2, _ = shared_preamble(B)
        if n2 > n:
            n, d = n2, d2
        if not n:
            raise ValueError("the bursts share no preamble: give preamble (symbols)")
        pre = preamble_symbols(d, pts)
    else:
        pre = np.asarray(preamble, dtype=np.complex128)
    sps = Fs/baud
    out = []
    for b in B:
        pad = int(np.ceil(16*sps)) + len(pre)
        a2, b2 = max(0, b.start - pad), min(len(x), b.stop + pad)
        seg = x[a2:b2]
        r = acquire(seg, Fs, pre, baud, beta, search=(0, b.start - a2 + 4*len(pre)*sps))
        if r is None:
            out.append(BurstResult(start=float(b.start), score=0.0, ok=False))
            continue
        t0, score, w, phase, gain = r
        if score < min_score:
            out.append(BurstResult(start=a2 + t0, score=score, ok=False))
            continue
        from ...sync.time.mat import centres_in
        c = centres_in(len(seg), t0, sps, 8)
        c = c[c <= b.stop - a2 + sps] # to the end of the gate
        y = _mf(seg, c, sps, beta)
        k = np.arange(len(y))
        y = y*np.exp(-1j*(w*k + phase))/max(gain, 1e-12)
        y = np.concatenate((y[:len(pre)], track(y[len(pre):], pts))) if len(y) > len(pre) else y
        idx = np.argmin(np.abs(y[:, None] - pts[None, :]), axis=1)
        dec = pts[idx]
        data = slice(len(pre), None)
        evm = float(np.sqrt(np.mean(np.abs(y[data] - dec[data])**2)/np.mean(np.abs(pts)**2))) if len(y) > len(pre) else 0.0
        out.append(BurstResult(start=a2 + t0, symbols=y, bits=_bits(idx, con),
                               cfo=w*baud/(2*np.pi), phase=phase, gain=gain, score=score, evm=evm, ok=True))
    # TDMA structure
    slots = []
    ok = [b for b in out if b.ok]
    if frame is None and len(ok) >= 3:
        from .waveform import bursts as wb
        per = wb(x, Fs)
        frame = per.period if per is not None and per.period else None
    if frame and ok:
        tol = slot_tol/baud
        starts = np.array([b.start/Fs for b in ok])
        t0 = starts.min()
        ph = (starts - t0) % frame
        for p in np.sort(ph):
            if not slots or min(abs(p - s) % frame for s in slots) > tol and min(frame - abs(p - s) % frame for s in slots) > tol:
                slots.append(float(p))
        slots = sorted(slots)
        for b, s, p in zip(ok, starts, ph):
            b.frame = int((s - t0 + tol)//frame)
            dist = [min(abs(p - q), frame - abs(p - q)) for q in slots]
            b.slot = int(np.argmin(dist))
    return Bursts(bursts=out, mod=mod, baud=baud, beta=beta, preamble=pre, frame=frame, slots=slots)
