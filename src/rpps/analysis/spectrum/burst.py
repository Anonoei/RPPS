"""TDMA bursts demodulated one by one: each acquired on the preamble the bursts share, as a TDMA receiver must

    r = rp.analysis.spectrum.burst.demodulate(x, Fs)              # or mod=, baud=, preamble= (symbols) when known
    print(r)       # "12 bursts, qpsk at 100.0 kBd, a 48-symbol preamble; frame 4.615 ms, 8 slots; EVM 6.1% median"
    b = r.bursts[0]
    b.symbols, b.bits, b.cfo, b.evm, b.slot, b.frame   # carrier-, timing- and phase-corrected symbols, decided bits

Every burst comes from another transmitter (or the same one after a gap), with its own carrier offset, phase and symbol
timing, and is too short to train loops on: so each is acquired on its own, data-aided, from the preamble.

  1. rpps.analysis.framing.frames.bursts gates the bursts (bursts that touch cut apart at a guard time or a power step:
     frames.split) and acquires each blind (rate, modulation, offset); bursts too short
     for that keep only their span. The modulation is the bursts' majority, the rate their median (or `mod` / `baud`).
  2. The preamble: the leading run of symbols the bursts agree on, as phase steps (rpps.analysis.framing.frames.preamble, or
     `shared_preamble` for a short one, whichever is longer), or `preamble` given. A network's burst types each carry a
     unique word of their own: when the bursts share fewer than `min_shared` symbols, they are grouped by length and
     each group's own preamble found (`burst_kinds`); each burst is acquired on its kind's. Known symbols elsewhere (a
     mid-amble, a postamble: `shared_segments`) serve the same way, at their offset; with none at all, each burst is
     returned as acquired blind (`Bursts.blind`: its phase known only to the constellation's symmetry). Its symbols are rebuilt from the steps and turned onto the constellation (one rotation for all
     bursts: the M-fold phase ambiguity is the same for every burst, so their bits agree with each other).
  3. In each burst the preamble is found by correlating its shaped waveform in pieces (the pieces' magnitudes added: a
     carrier offset costs little), then the fractional timing that best matches the preamble's phase steps (blind to
     the carrier), then from z_k = y_k conj(p_k) over the preamble: the carrier offset (the mean turn of z_k z*_{k-1})
     and the phase and gain (sum z_k with the turn taken out).
  4. The burst's symbols, from the preamble to the end of the gate, are taken at that timing and corrected. A burst of
     several `window`s has its symbol clock followed (`retime`): the matched filter's energy peak in each window
     (rpps.sync.time.mat.best_tau), a line through them anchored on the preamble (the clock offset, `clock`). A second-
     order decision-directed loop follows what is left of the carrier through the data; decisions give bits, the
     soft demapper LLRs (`llr`, noise from the burst's own symbols), and the error vector magnitude.
  5. The TDMA structure: the period of the bursts' starts (rpps.analysis.spectrum.waveform.on_off, else `frame`) and the slots,
     clusters of the starts within a frame (to within `slot_tol` symbols): each burst's frame and slot.
Limits: the clock is followed as a straight line (a constant offset over the burst), from windows of a few hundred
symbols; a burst whose preamble is weaker than the data's noise floor allows is reported with `ok` False and no bits.
"""
import numpy as np

from ..framing import frames

class BurstResult:
    """One burst: start (sample of its preamble's first symbol, fractional), symbols, bits, llr (per bit, log(P0/P1), the
    preamble's included, as bits), cfo (Hz), clock (its symbol clock's offset, a fraction: +1e-4 is 100 ppm fast), phase
    (rad), gain, score (the preamble's correlation 0..1), evm (rms, of the data after the preamble), frame, slot, kind
    (which of the preambles it was acquired on), known ((first, length): the known symbols in `symbols`), ok"""
    __slots__ = ("start", "symbols", "bits", "llr", "cfo", "clock", "phase", "gain", "score", "evm", "frame", "slot",
                 "kind", "known", "ok")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def payload(self):
        """The symbols without the known ones"""
        a, n = self.known if self.known else (0, 0)
        return np.concatenate((self.symbols[:a], self.symbols[a + n:]))

    def __repr__(self):
        if not self.ok:
            return f"<burst at {self.start:.0f}: not acquired (preamble score {self.score or 0:.2f})>"
        sl = f", frame {self.frame} slot {self.slot}" if self.slot is not None else ""
        ck = f", clock {1e6*self.clock:+.0f} ppm" if self.clock else ""
        return (f"<burst at {self.start:.1f}: {len(self.symbols)} symbols, cfo {self.cfo:+.1f} Hz{ck}, EVM "
                f"{100*self.evm:.1f}%{sl}>")

class Bursts:
    """`demodulate`: bursts [BurstResult], mod, baud, beta, preamble (symbols: the most common kind's), preambles (every
    kind's), offsets (each kind's known symbols from the bursts' start; negative: from their end), frame (s, None), slots
    (offsets within the frame, s), blind (no known symbols: each burst's phase only to the constellation's symmetry)"""
    __slots__ = ("bursts", "mod", "baud", "beta", "preamble", "preambles", "offsets", "frame", "slots", "blind")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        ok = [b for b in self.bursts if b.ok]
        kinds = [k for k in (self.preambles or [self.preamble]) if k is not None and len(k)]
        where = ""
        if self.offsets and len(kinds) == 1 and self.offsets[0]:
            o = self.offsets[0]
            where = f" {o} symbols in" if o > 0 else f" {-o} symbols before the end"
        s = (f"{len(self.bursts)} bursts ({len(ok)} demodulated), {self.mod} at {self.baud/1e3:.3f} kBd, "
             + ("blind (no known symbols they share)" if not kinds else
                f"{len(kinds[0])} known symbols{where}" if len(kinds) == 1 else
                f"{len(kinds)} burst kinds (preambles of {', '.join(str(len(p)) for p in kinds)} symbols)"))
        if self.frame:
            s += f"; frame {self.frame*1e3:.3f} ms, {len(self.slots)} slots"
        if ok:
            s += f"; EVM {100*np.median([b.evm for b in ok]):.1f}% median"
        return s

    __repr__ = __str__

    def llrs(self, kind=None, preamble=False):
        """Each demodulated burst's LLRs (of `kind` only, when given), without its known symbols' unless `preamble`: a
        list, the bursts' lengths differing"""
        out = []
        for b in self.bursts:
            if b.ok and b.llr is not None and (kind is None or b.kind == kind):
                m = len(b.llr)//len(b.symbols)
                a, n = b.known if b.known else (0, 0)
                out.append(b.llr if preamble else np.concatenate((b.llr[:a*m], b.llr[(a + n)*m:])))
        return out

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

class Segment:
    """`shared_segments`: steps (d[n] = s[n] conj(s[n-1]) the bursts agree on), offset (symbols from the bursts' start
    to the segment's first symbol; negative: from their end, a postamble), agree (|mean| of the steps over bursts)"""
    __slots__ = ("steps", "offset", "agree")
    def __init__(self, steps, offset, agree):
        self.steps, self.offset, self.agree = steps, offset, agree

    def __repr__(self):
        where = f"{self.offset} symbols from the start" if self.offset >= 0 else f"{-self.offset} symbols before the end"
        return f"<{len(self.steps) + 1} shared symbols, {where}, agreement {self.agree:.2f}>"

def _unit_steps(v):
    return v[1:]*np.conj(v[:-1])/(np.abs(v[1:]*v[:-1]) + 1e-12)

def shared_segments(burst_list, search=8, min_len=12, agree=0.8, refs=6):
    """Runs of symbols every burst carries at the same place: a preamble, a mid-amble, a postamble ([Segment], longest
    first). Phase steps, so each burst's own carrier phase drops out. Each burst is aligned on a reference by a lag
    within `search` symbols: the one with the best window of `min_len` steps (a known run of that length adds up to its
    length, random steps to its square root), once from the start and once from the end (bursts of several lengths
    sharing a postamble); the reference is the burst the others agree with most (of the first `refs`). A run is where
    the mean step over the bursts has magnitude above `agree`, `min_len` steps long at least"""
    S = [b.symbols for b in burst_list if b.symbols is not None and len(b.symbols) > 2*search + min_len + 2]
    if len(S) < 3:
        return []
    out = []
    for end in (False, True):
        D = [_unit_steps(v[::-1] if end else v) for v in S]
        best = None
        for r in range(min(refs, len(D))):
            ref = D[r]
            lags = []
            for d in D:
                n = min(len(ref), len(d)) - 2*search
                c = []
                for l in range(-search, search + 1):
                    pr = np.conj(ref[search:search + n])*d[search + l:search + l + n]
                    cs = np.concatenate(([0], np.cumsum(pr)))
                    c.append(np.abs(cs[min_len:] - cs[:-min_len]).max()) # the best window of min_len steps
                lags.append(int(np.argmax(c)) - search)
            lo = max(0, -min(lags))
            L = min(len(d) - l - lo for d, l in zip(D, lags))
            if L <= min_len:
                continue
            rows = np.array([d[l + lo:l + lo + L] for d, l in zip(D, lags)])
            mean = rows.mean(axis=0)
            m = np.abs(mean)
            if best is None or np.sum(m > agree) > best[0]:
                best = (int(np.sum(m > agree)), m, mean, lo)
        if best is None:
            continue
        _, m, mean, lo = best
        on = np.concatenate(([0], (m > agree).astype(np.int8), [0]))
        d = np.diff(on)
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a < min_len:
                continue
            steps = mean[a:b]
            if end: # reversed: steps of the reversed run are the conjugates, in reverse order
                steps = np.conj(steps[::-1])
                offset = -(a + lo + (b - a) + 1) # its first symbol, counted back from the last
            else:
                offset = a + lo
            out.append(Segment(steps, int(offset), float(np.mean(m[a:b]))))
    # The same run seen from both ends (bursts of one length): keep the start-anchored one
    keep = []
    for sg in sorted(out, key=lambda g: (-len(g.steps), g.offset < 0)):
        if all(not (sg.offset < 0 <= k.offset and abs(len(sg.steps) - len(k.steps)) <= 2) for k in keep):
            keep.append(sg)
    return keep

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

def retime(x, c, sps, beta, anchor=0, window=256):
    """Symbol centres c (samples of x, one a symbol) with the symbol clock's offset taken out, and that offset (the
    symbol rate's, a fraction: positive when the symbols come faster than sps): the matched filter's energy peak (best_tau) in each window of `window` symbols, unwrapped, a line
    fitted through them by least squares and turned about symbol `anchor` (where the preamble set the timing). c and 0
    when it holds fewer than three windows"""
    from ...sync.time.mat import best_tau
    nw = len(c)//window
    if nw < 3:
        return c, 0.0
    mids, offs = [], []
    for i in range(nw):
        a, b = i*window, (i + 1)*window - 1
        lo, hi = max(0, int(c[a] - sps)), min(len(x), int(np.ceil(c[b] + sps)) + 1)
        tau, _ = best_tau(x, sps, beta, lo=lo, hi=hi)
        m = (a + b)/2
        cm = c[0] + m*sps
        offs.append(2*np.pi*((tau - cm) % sps)/sps)
        mids.append(m)
    d = np.unwrap(np.array(offs))*sps/(2*np.pi)
    d = d - np.round(np.median(d)/sps)*sps # the lattice is periodic in sps: the nearest copy
    slope, _ = np.polyfit(np.array(mids), d, 1)
    k = np.arange(len(c))
    return c + slope*(k - anchor), float(-slope/sps) # sps (1 + e) samples a symbol: a rate 1/(1 + e), about 1 - e

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

def burst_kinds(B, sps, pts, min_shared=16, min_group=3):
    """(preambles [symbols], member [kind index or None per burst]): one shared by all the bursts, when they share at least
    `min_shared` symbols; else one per group of bursts of a length (lengths within 16 symbols or 2 %: a network's burst
    types, each with its own unique word), groups of at least `min_group` with at least `min_shared` symbols in common.
    Most bursts first. A burst in no group has None (it is tried on every kind's)"""
    def shared(g):
        # The gate opens within a few symbols of a short burst's start, but a long burst's (thousands of symbols) can be
        # tens of symbols out: each search a near one, then a wider one, the longest kept
        best = (0, None)
        for search in (8, 32):
            for n, d, _ in (frames.preamble(g, search=search), shared_preamble(g, search=search)):
                if n > best[0]:
                    best = (n, d)
        return best
    n, d = shared(B)
    if n >= min_shared or len(B) < 2*min_group:
        return ([preamble_symbols(d, pts)], [0]*len(B)) if n else ([], [None]*len(B))
    L = np.array([(b.stop - b.start)/sps for b in B])
    order = np.argsort(L)
    groups, cur = [], [order[0]]
    for i, j in zip(order[:-1], order[1:]):
        if L[j] - L[i] > max(16, 0.02*L[i]):
            groups.append(cur); cur = []
        cur.append(j)
    groups.append(cur)
    kinds, member = [], [None]*len(B)
    for g in sorted((g for g in groups if len(g) >= min_group), key=len, reverse=True):
        n, d = shared([B[i] for i in g])
        if n >= min_shared:
            for i in g:
                member[i] = len(kinds)
            kinds.append(preamble_symbols(d, pts))
    if not kinds and n: # (no group of its own: what they all share, however short)
        n, d = shared(B)
        if n and np.all(np.isfinite(d)):
            return [preamble_symbols(d, pts)], [0]*len(B)
        return [], [None]*len(B)
    return kinds, member

def demodulate(x, Fs, mod=None, baud=None, beta=None, preamble=None, offset=0, frame=None, slot_tol=4, min_score=0.5,
               min_shared=16, **kw):
    """Every burst of x, demodulated on its own from its preamble (see the module). Bursts of several kinds, each with a
    preamble of its own (burst_kinds: by length, when they share fewer than `min_shared` symbols), are each acquired on
    their kind's. With no preamble they share, known symbols elsewhere (`shared_segments`: a mid-amble, a postamble);
    with none at all, each burst blind (`Bursts.blind`). preamble: known symbols given, `offset` symbols after the
    burst's start (negative: before its end). kw: frames.gate's"""
    from ... import symbol as rsym
    x = np.asarray(x, dtype=np.complex128)
    B = frames.bursts(x, Fs, names=None if mod is None else [mod], baud=baud,
                      betas=(0.2, 0.25, 0.35) if beta is None else (beta,), **kw) # what is known isn't searched for
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
    sps = Fs/baud
    offsets = None
    if preamble is None:
        kinds, member = burst_kinds(B, sps, pts, min_shared)
        if len(kinds) <= 1: # one kind, or none: where the known symbols sit (a "preamble" may be a mid-amble)
            segs = shared_segments(B, min_len=max(8, min_shared - 1))
            top = segs[0] if segs else None
            if top is not None and (not kinds or len(top.steps) + 1 > len(kinds[0])
                                    or (len(top.steps) + 1 >= len(kinds[0]) - 2 and not 0 <= top.offset <= 8)):
                o = 0 if 0 <= top.offset <= 8 else top.offset # (within the gate's slack of the start: a preamble)
                kinds, member, offsets = [preamble_symbols(top.steps, pts)], [0]*len(B), [o]
        if not kinds or len(kinds[0]) < 8: # a symbol or two in common is chance: each burst blind
            return _blind(B, mod, baud, beta, con, pts, frame, slot_tol, x, Fs)
    else:
        kinds, member = [np.asarray(preamble, dtype=np.complex128)], [0]*len(B)
        offsets = [int(offset)]
    offsets = offsets or [0]*len(kinds)
    out = []
    for b, m in zip(B, member):
        tries = [m] if m is not None else range(len(kinds)) # A burst of no group: every kind's preamble, the best kept
        best = None
        for kind in tries:
            p, o = kinds[kind], offsets[kind]
            pad = int(np.ceil(16*sps)) + len(p)
            a2, b2 = max(0, b.start - pad), min(len(x), b.stop + pad)
            # The burst's own blind carrier offset off first: the preamble search correlates in pieces, which a carrier
            # tens of kHz out (a large share of a turn a symbol) turns past finding
            f0 = float(getattr(b.det, "cfo", 0.0) or 0.0) if b.det is not None else 0.0
            seg = x[a2:b2]*np.exp(-2j*np.pi*f0/Fs*np.arange(b2 - a2)) if f0 else x[a2:b2]
            if o == 0:
                search = (0, b.start - a2 + 4*len(p)*sps)
            else: # where the known run should be, from the gate's start (or its end: a postamble)
                e = (b.start - a2 if o > 0 else b.stop - a2) + o*sps
                margin = max(16, len(p))*sps
                search = (e - margin, e + margin)
            r = acquire(seg, Fs, p, baud, beta, search=search)
            if r is not None and (best is None or r[1] > best[0][1]):
                best = (r, kind, a2, seg, f0)
        if best is None:
            out.append(BurstResult(start=float(b.start), score=0.0, ok=False))
            continue
        r, kind, a2, seg, f0 = best
        pre, o = kinds[kind], offsets[kind]
        t0, score, w, phase, gain = r
        if score < min_score:
            out.append(BurstResult(start=a2 + t0, score=score, kind=kind, ok=False))
            continue
        # The burst's first symbol: `o` before the known run, or (a postamble) wherever the gate opened
        oe = o if o >= 0 else max(0, int(round((t0 - (b.start - a2))/sps)))
        oe = min(oe, int((t0 - 8*sps - 5)//sps)) # (the matched filter's span before it)
        from ...sync.time.mat import centres_in
        c = centres_in(len(seg), t0 - oe*sps, sps, 8)
        c = c[c <= b.stop - a2 + sps] # to the end of the gate
        n = len(pre)
        c, clock = retime(seg, c, sps, beta, anchor=oe + n/2)
        y = _mf(seg, c, sps, beta)
        k = np.arange(len(y)) - oe
        y = y*np.exp(-1j*(w*k + phase))/max(gain, 1e-12)
        if len(y) > oe + n: # the carrier loop from the known run out, both ways
            y = np.concatenate((track(y[:oe][::-1], pts)[::-1], y[oe:oe + n], track(y[oe + n:], pts)))
        idx = np.argmin(np.abs(y[:, None] - pts[None, :]), axis=1)
        dec = pts[idx]
        data = np.ones(len(y), bool)
        data[oe:oe + n] = False
        evm = float(np.sqrt(np.mean(np.abs(y[data] - dec[data])**2)/np.mean(np.abs(pts)**2))) if data.any() else 0.0
        out.append(BurstResult(start=a2 + c[0], symbols=y, bits=_bits(idx, con), llr=con.llr(y), cfo=f0 + w*baud/(2*np.pi),
                               clock=clock, phase=phase, gain=gain, score=score, evm=evm, kind=kind, known=(oe, n), ok=True))
    frame, slots = _structure(out, x, Fs, baud, frame, slot_tol)
    return Bursts(bursts=out, mod=mod, baud=baud, beta=beta, preamble=kinds[0], preambles=kinds, offsets=offsets, frame=frame,
                  slots=slots, blind=False)

def _structure(out, x, Fs, baud, frame, slot_tol):
    """The TDMA frame (given, or the period of the bursts' starts) and its slots; each burst's frame and slot set"""
    slots = []
    ok = [b for b in out if b.ok]
    if frame is None and len(ok) >= 3:
        from .waveform import on_off as wb
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
    return frame, slots

def _blind(B, mod, baud, beta, con, pts, frame, slot_tol, x, Fs):
    """Bursts that share no known symbols: each as frames.bursts acquired it blind (its own rate, timing, carrier, and a
    phase known only to the constellation's symmetry: bits may differ burst to burst by that rotation)"""
    out = []
    for b in B:
        if b.symbols is None:
            out.append(BurstResult(start=float(b.start), score=0.0, ok=False))
            continue
        y = b.symbols
        idx = np.argmin(np.abs(y[:, None] - pts[None, :]), axis=1)
        evm = float(np.sqrt(np.mean(np.abs(y - pts[idx])**2)/np.mean(np.abs(pts)**2)))
        out.append(BurstResult(start=float(b.start), symbols=y, bits=_bits(idx, con), llr=con.llr(y), cfo=float(b.det.cfo),
                               clock=0.0, phase=None, gain=None, score=None, evm=evm, kind=None, known=(0, 0), ok=True))
    frame, slots = _structure(out, x, Fs, baud, frame, slot_tol)
    none = np.zeros(0, dtype=np.complex128) # (no known symbols: one kind, of an empty preamble)
    for b in out:
        b.kind = 0 if b.ok else None
    return Bursts(bursts=out, mod=mod, baud=baud, beta=beta, preamble=none, preambles=[none], offsets=[0], frame=frame,
                  slots=slots, blind=True)
