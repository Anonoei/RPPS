"""Each burst acquired on its preamble and demodulated (`acquire`, `retime`, `demodulate`), and the TDMA frame and slots
the bursts' starts show. See rpps.analysis.burst"""
import numpy as np

from ..framing import frames
from .preamble import burst_kinds, periodic_word, preamble_symbols, shared_segments
from .plan import learn_plan

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
    (offsets within the frame, s), blind (no known symbols: each burst's phase only to the constellation's symmetry),
    period (symbols: a unique word repeating that often within each burst, at its own phase (offsets [None]); a burst's
    `known` is its first), frame (s: the period the bursts' pattern repeats at, every slot in the same frames of it; None
    when the starts fall on no slot grid), plan_frame (s: the burst time plan's frame, learn_plan's: half or less of
    `frame` where terminals get slots in alternate frames of it; None when not found or not fitting the starts)"""
    __slots__ = ("bursts", "mod", "baud", "beta", "preamble", "preambles", "offsets", "frame", "slots", "blind", "period",
                 "plan_frame")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        ok = [b for b in self.bursts if b.ok]
        kinds = [k for k in (self.preambles or [self.preamble]) if k is not None and len(k)]
        where = ""
        if self.offsets and len(kinds) == 1 and self.offsets[0] is None:
            where = f" every {self.period} symbols, at each burst's own phase"
        elif self.offsets and len(kinds) == 1 and self.offsets[0]:
            o = self.offsets[0]
            where = f" {o} symbols in" if o > 0 else f" {-o} symbols before the end"
        s = (f"{len(self.bursts)} bursts ({len(ok)} demodulated), {self.mod} at {self.baud/1e3:.3f} kBd, "
             + ("blind (no known symbols they share)" if not kinds else
                f"{len(kinds[0])} known symbols{where}" if len(kinds) == 1 else
                f"{len(kinds)} burst kinds (preambles of {', '.join(str(len(p)) for p in kinds)} symbols)"))
        if self.frame:
            s += f"; frame {self.frame*1e3:.3f} ms, {len(self.slots)} slots"
            if self.plan_frame and abs(self.plan_frame - self.frame) > 1e-6:
                s += f" (the plan's frame {self.plan_frame*1e3:.3f} ms)"
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

def retime(x, c, sps, beta, anchor=0, window=256, skip=None, offset=False):
    """Symbol centres c (samples of x, one a symbol) with the symbol clock's offset taken out, and that offset (the
    symbol rate's, a fraction: positive when the symbols come faster than sps): the matched filter's energy peak
    (best_tau) in each window of `window` symbols but those over the known symbols `skip` ((first, stop): a repeating
    pattern's energy can peak between its symbols), unwrapped, a line through them (Theil-Sen: the median slope between
    windows) turned about symbol `anchor` (where the preamble set the timing; with `offset`, the timing there from the
    windows too, for an acquisition on a short word). c and 0 when fewer than three windows"""
    from ...sync.time.mat import best_tau
    nw = len(c)//window
    if nw < 3:
        return c, 0.0
    mids, offs = [], []
    for i in range(nw):
        a, b = i*window, (i + 1)*window - 1
        if skip is not None and a < skip[1] and skip[0] <= b: # known symbols (a 1100 preamble's energy peaks between symbols)
            continue
        lo, hi = max(0, int(c[a] - sps)), min(len(x), int(np.ceil(c[b] + sps)) + 1)
        tau, _ = best_tau(x, sps, beta, lo=lo, hi=hi)
        m = (a + b)/2
        cm = c[0] + a*sps # (a lattice point: tau is a phase of the lattice, the window's middle may fall between)
        offs.append(2*np.pi*((tau - cm) % sps)/sps)
        mids.append(m)
    if len(offs) < 3:
        return c, 0.0
    d = np.unwrap(np.array(offs))*sps/(2*np.pi)
    d = d - np.round(np.median(d)/sps)*sps # the lattice is periodic in sps: the nearest copy
    mids = np.array(mids)
    i, j = np.triu_indices(len(mids), 1) # Theil-Sen: the median of the slopes between windows (one odd window can't tilt it)
    slope = float(np.median((d[j] - d[i])/(mids[j] - mids[i])))
    k = np.arange(len(c))
    if offset: # the windows' timing at the anchor too, not the acquisition's (a short word's can be half a symbol out)
        at = float(np.median(d - slope*(mids - anchor)))
        at -= np.round(at/sps)*sps
        return c + at + slope*(k - anchor), float(-slope/sps)
    return c + slope*(k - anchor), float(-slope/sps) # sps (1 + e) samples a symbol: a rate 1/(1 + e), about 1 - e

def _word_carrier(y, word, first, period, M=4):
    """Symbols y (one a symbol, a word every period) with the carrier the words show taken out: (y corrected, rad a
    symbol, phase at the first word, gain, the first word's position: found again by folding every word's correlation
    over the period, `first` only a start). The turn from word to word gives the offset (within half a turn
    a period of what the acquisition left: the blind estimate is far closer); each word's phase, unwrapped along it,
    anchors the phase there, and between words it is interpolated (phase noise followed word to word, which one
    word's line over thousands of symbols cannot)"""
    n = len(word)
    # the offset first from the whole burst's M-th power (thousands of symbols: far finer than the half turn a period
    # the words can tell apart), then the words for what is left
    d = y[1:]*np.conj(y[:-1])
    w0 = float(np.angle(np.sum(d**M/np.maximum(np.abs(d), 1e-12)**(M - 1))))/M
    y = y*np.exp(-1j*w0*(np.arange(len(y)) - first))
    # where the words are: the position in the period whose word correlations, summed in magnitude over every period,
    # are largest (the acquisition's hit can be a later word, or a chance match of a short word)
    from numpy.lib.stride_tricks import sliding_window_view
    corr = np.abs(sliding_window_view(y, n) @ np.conj(word)) if len(y) >= n else np.zeros(0)
    if len(corr) >= period:
        fold = np.bincount(np.arange(len(corr)) % period, corr, period)
        first = int(np.argmax(fold))
    at = np.arange(first, len(y) - n + 1, period)
    z = np.array([np.sum(y[a:a + n]*np.conj(word)) for a in at])
    if len(z) < 2:
        w = 0.0
        ph = np.angle(z) if len(z) else np.zeros(1)
    else:
        w = float(np.angle(np.sum(z[1:]*np.conj(z[:-1]))))/period
        ph = np.unwrap(np.angle(z*np.exp(-1j*w*(at - first)))) + w*(at - first)
    mid = at + (n - 1)/2
    k = np.arange(len(y))
    theta = np.interp(k, mid, ph) if len(at) > 1 else ph[0] + w*(k - mid[0])
    if len(at) > 1: # beyond the first and last word: their line's slope
        theta = np.where(k < mid[0], ph[0] + w*(k - mid[0]), theta)
        theta = np.where(k > mid[-1], ph[-1] + w*(k - mid[-1]), theta)
    y = y*np.exp(-1j*theta)
    gain = float(np.sqrt(np.mean(np.abs(y)**2)/np.mean(np.abs(word)**2))) # (the burst's power: the words' sums are noisy)
    return y/max(gain, 1e-12), w + w0, float(ph[0]), gain, first

def _follow(y, points, k1=0.05, k2=0.002):
    """What is left of the carrier in corrected symbols y (one a symbol) followed by a second-order decision-directed
    loop (rpps.sync.Costas on the constellation's points): the symbols with it taken off"""
    from ...sync.costas import Costas
    loop = Costas(k1, k2, points=points)
    return loop.step(y, loop.new_state(None), None)

def demodulate(x, Fs, mod=None, baud=None, beta=None, preamble=None, offset=0, frame=None, slot_tol=16, min_score=0.5,
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
    offsets, period = None, None
    if preamble is None:
        kinds, member = burst_kinds(B, sps, pts, min_shared)
        if len(kinds) <= 1: # one kind, or none: where the known symbols sit (a "preamble" may be a mid-amble)
            from ...sync.acquire import symmetry
            segs = shared_segments(B, min_len=max(8, min_shared - 1), M=max(2, symmetry(pts)))
            top = segs[0] if segs else None
            if top is not None and (not kinds or len(top.steps) + 1 > len(kinds[0])
                                    or (len(top.steps) + 1 >= len(kinds[0]) - 2 and not 0 <= top.offset <= 8)):
                o = 0 if 0 <= top.offset <= 8 else top.offset # (within the gate's slack of the start: a preamble)
                kinds, member, offsets = [preamble_symbols(top.steps, pts)], [0]*len(B), [o]
        if not kinds or len(kinds[0]) < 8: # long bursts of frames: the unique word each repeats, at its own phase
            pw = periodic_word(B, mod, pts)
            if pw is not None:
                kinds, member, offsets, period = [pw[0]], [0]*len(B), [None], pw[1]
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
            elif o is None: # a word every `period`: its first, within a period of the gate's opening
                search = (max(0, b.start - a2 - 8*sps), b.start - a2 + (period + len(p))*sps)
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
        oe = o if o is not None and o >= 0 else max(0, int(round((t0 - (b.start - a2))/sps)))
        oe = min(oe, int((t0 - 8*sps - 5)//sps)) # (the matched filter's span before it)
        from ...sync.time.mat import centres_in
        c = centres_in(len(seg), t0 - oe*sps, sps, 8)
        c = c[c <= b.stop - a2 + sps] # to the end of the gate
        n = len(pre)
        c, clock = retime(seg, c, sps, beta, anchor=oe + n/2, skip=(oe, oe + n), offset=o is None)
        y = _mf(seg, c, sps, beta)
        k = np.arange(len(y)) - oe
        if o is None and period: # a word every period: the carrier from all of them, not from the first alone
            from ...sync.acquire import symmetry
            y, w, phase, gain, oe = _word_carrier(y, pre, oe, period, symmetry(pts))
        else:
            y = y*np.exp(-1j*(w*k + phase))/max(gain, 1e-12)
        if len(y) > oe + n: # the carrier loop from the known run out, both ways
            y = np.concatenate((_follow(y[:oe][::-1], pts)[::-1], y[oe:oe + n], _follow(y[oe + n:], pts)))
        idx = np.argmin(np.abs(y[:, None] - pts[None, :]), axis=1)
        dec = pts[idx]
        data = np.ones(len(y), bool)
        data[oe:oe + n] = False
        evm = float(np.sqrt(np.mean(np.abs(y[data] - dec[data])**2)/np.mean(np.abs(pts)**2))) if data.any() else 0.0
        out.append(BurstResult(start=a2 + c[0], symbols=y, bits=_bits(idx, con), llr=con.llr(y), cfo=f0 + w*baud/(2*np.pi),
                               clock=clock, phase=phase, gain=gain, score=score, evm=evm, kind=kind, known=(oe, n), ok=True))
    frame, slots, plan = _structure(out, x, Fs, baud, frame, slot_tol)
    return Bursts(bursts=out, mod=mod, baud=baud, beta=beta, preamble=kinds[0], preambles=kinds, offsets=offsets, frame=frame,
                  slots=slots, blind=False, period=period, plan_frame=plan)

def _clustered(starts, frame, tol):
    """Share of starts whose phase in the frame lies within tol of two others' at least (a slot used again and again)"""
    ph = np.sort(starts % frame)
    ext = np.concatenate((ph - frame, ph, ph + frame))
    lo, hi = np.searchsorted(ext, ph - tol), np.searchsorted(ext, ph + tol, side="right")
    return float(np.mean(hi - lo - 1 >= 2))

def _slots_of(starts, frame, tol):
    """Slot index of each start in a frame (single linkage of the phases at tol, joined across the wrap)"""
    ph = starts % frame
    srt = np.sort(ph)
    edges = srt[np.flatnonzero(np.diff(srt) > tol) + 1]
    slot = np.searchsorted(edges, ph, side="right")
    if len(edges) and srt[0] + frame - srt[-1] <= tol:
        slot[slot == len(edges)] = 0
    return slot

def _multiple(starts, T0, tol, kmax=16, excess=0.2):
    """The frame as k*T0: the smallest k at which the slots of period T0 leave residues of the frame index mod k empty
    well beyond chance (a slot used only in some of the k parts of the true frame), else T0. Irregular use (a terminal
    in some frames only) spreads over every residue, so it does not multiply the period"""
    slot = _slots_of(starts, T0, tol)
    fr = np.floor((starts - starts.min())/T0 + tol/T0).astype(int)
    for k in range(2, kmax + 1):
        gain, tot = 0.0, 0
        for sl in np.unique(slot):
            f = fr[slot == sl]
            if len(f) < k:
                continue
            missing = k - len(set((f % k).tolist()))
            gain += missing - k*(1 - 1/k)**len(f)
            tot += k
        if tot and gain/tot > excess:
            return k*T0
    return T0

def _frame(ok, x, Fs, baud, slot_tol):
    """(frame, plan_frame) in seconds, each or both None. frame: the period the bursts' pattern repeats at (every slot
    used in the same frames of it); plan_frame: learn_plan's (the burst time plan's frame: every slot holds one kind),
    which is shorter where terminals are given slots in alternate frames of it. A base period: learn_plan's (the BTU grid of the starts, then the shortest
    period at which every slot holds one kind), else the on/off pattern's (rpps.analysis.spectrum.waveform.on_off),
    reduced to its shortest divisor the starts cluster at as well (`_clustered`); then `_multiple`: k times it where
    its slots are used in only some residues of the frame index mod k (the base was a divisor of the frame)"""
    from ..spectrum.waveform import on_off
    starts = np.array([b.start/Fs for b in ok])
    tol = slot_tol/baud
    T = plan = None
    try:
        rec = [{"freq": 0.0, "start": float(b.start/Fs), "length": len(b.symbols)/baud, "kind": b.kind, "rate": baud}
               for b in ok]
        T = plan = float(learn_plan(rec).evidence["frame"])
    except (ValueError, KeyError, ZeroDivisionError):
        per = on_off(x, Fs)
        T = float(per.period) if per is not None and per.period else None
    if T is None:
        return None, None
    if plan is not None and _clustered(starts, plan, tol) < 0.5:
        plan = None
    # the shortest divisor the starts cluster at as well as at the best of them (learn_plan wants bursts at two places
    # of a frame: one burst a period reads as two periods, which the starts fit worse), then up again where the slots show which part of a longer frame they belong to
    for base in (T, float(np.median(np.diff(np.sort(starts))))): # (else the starts' own spacing: one burst a period)
        cs = {base/k: _clustered(starts, base/k, tol) for k in range(1, 9) if base/k > 2*tol}
        c = max(cs.values(), default=0.0)
        if c >= 0.5:
            break
    else: # the starts fall in slots of none of them: bursts, but not on a slot grid
        return None, plan
    T0 = min(t for t, v in cs.items() if v >= c - 0.02)
    for _ in range(8): # (a factor at a time: 2 then 2 again for a quarter)
        T1 = _multiple(starts, T0, tol)
        if T1 == T0:
            break
        T0 = T1
    return T0, plan

def _structure(out, x, Fs, baud, frame, slot_tol):
    """The TDMA frame (given, or `_frame`'s) and its slots; each burst's frame and slot set"""
    slots = []
    ok = [b for b in out if b.ok]
    plan = None
    if frame is None and len(ok) >= 3:
        frame, plan = _frame(ok, x, Fs, baud, slot_tol)
    if frame and ok:
        tol = slot_tol/baud
        starts = np.array([b.start/Fs for b in ok])
        t0 = starts.min()
        ph = (starts - t0) % frame
        # slots: the phases split where they lie more than `slot_tol` symbols apart (single linkage: starts a few
        # symbols apart, the gate's and the preamble's jitter, stay one slot), around the frame's wrap too
        srt = np.sort(ph)
        gaps = np.flatnonzero(np.diff(srt) > tol)
        groups = np.split(srt, gaps + 1)
        if len(groups) > 1 and srt[0] + frame - srt[-1] <= tol: # (the last joins the first across the wrap)
            groups[0] = np.concatenate((groups[-1] - frame, groups[0]))
            groups = groups[:-1]
        slots = sorted(float(np.mean(g)) for g in groups) # (a slot across the wrap: just under 0, still first)
        for b, s, p in zip(ok, starts, ph):
            b.frame = int((s - t0 + tol)//frame)
            dist = [min(abs(p - q), frame - abs(p - q)) for q in slots]
            b.slot = int(np.argmin(dist))
    return frame, slots, plan

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
    frame, slots, plan = _structure(out, x, Fs, baud, frame, slot_tol)
    none = np.zeros(0, dtype=np.complex128) # (no known symbols: one kind, of an empty preamble)
    for b in out:
        b.kind = 0 if b.ok else None
    return Bursts(bursts=out, mod=mod, baud=baud, beta=beta, preamble=none, preambles=[none], offsets=[0], frame=frame,
                  slots=slots, blind=True, period=None, plan_frame=plan)

# ---- a burst time plan learned from bursts
