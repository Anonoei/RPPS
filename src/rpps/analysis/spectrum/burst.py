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

A network's burst time plan (rpps.plan) from the bursts rp.sync.Bursts detects: `learn_plan` (the BTU, the frame, each
carrier's slots and the kinds of burst in them), and whether a plan, learned or told, explains the bursts:
`check_plan` (each burst in a slot of its carrier and kind, the plan's origin and clock fitted to them; rp.viz.bursts
draws it, folded on the superframe or along the capture).
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

def shared_preamble(burst_list, search=8, win=12, agree=0.8, max_len=512):
    """The leading run of symbols the bursts share, as phase steps: (length in steps, steps, where it starts in each
    burst's symbols), or (0, None, None). Like rpps.analysis.framing.frames.preamble, but aligned on a short window (`win` steps
    near the start, so a preamble shorter than framing's 64-step window aligns too), each burst against a dozen others'
    starts (the longest, the first, some spread over the list), the reference being the burst the others agree with
    most"""
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
    best = None # References: the longest bursts (their gate opened earliest: the preamble whole), the first, and some
    # spread over the rest (a gate opening late cuts a burst's preamble short: no reference among those)
    longest = sorted(range(len(D)), key=lambda i: -len(D[i]))[:4]
    refs = list(dict.fromkeys(longest + list(range(min(len(D), 4))) + list(np.linspace(0, len(D) - 1, 4).astype(int))))
    for i in refs:
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

def shared_segments(burst_list, search=24, min_len=12, agree=0.8, refs=6, M=4, false_alarm=1e-3):
    """Runs of symbols every burst carries at the same place: a preamble, a mid-amble, a postamble ([Segment], longest
    first). Phase steps, so each burst's own carrier phase drops out. Each burst is aligned on a reference by a lag
    within `search` symbols: the one with the best window of `min_len` steps (a known run of that length adds up to its
    length, random steps to its square root), once from the start and once from the end (bursts of several lengths
    sharing a postamble); the reference is the burst the others agree with most (of the first `refs`). A run is where
    the bursts' steps lie within 45 degrees of their mean: all of 3, all but one of 4 or more (up to `agree`: a weak
    burst's decision error), gaps of a step or two where a majority still agrees bridged; at least `min_len` steps, and
    longer where chance would make one (M: the steps' alphabet, M-PSK; `false_alarm` over every position)"""
    S = [b.symbols for b in burst_list if b.symbols is not None and len(b.symbols) > 2*search + min_len + 2]
    if len(S) < 3:
        return []
    # How many must agree: all of 3, all but one of 4 or more (a weak burst's decision error), at most `agree`; and so
    # how long a run chance would not make: M-PSK steps of random data agree that well at a position with probability
    # about M P(Binomial(K, 1/M) >= need), over every position of two alignments
    from math import comb, ceil, log
    K = len(S)
    need = K if K < 4 else max(int(np.ceil(agree*K)), K - 1) if agree*K <= K - 1 else K - 1
    need = min(need, K)
    share = need/K - 1e-9
    pc = min(1.0, M*sum(comb(K, j)*(1/M)**j*(1 - 1/M)**(K - j) for j in range(need, K + 1)))
    positions = 2*max(len(v) for v in S)*(2*search + 1) # (every position of both alignments, each lag tried)
    if pc < 1:
        min_len = max(min_len, int(ceil(log(false_alarm/positions)/log(pc))))
    out = []
    for end in (False, True):
        D = [_unit_steps(v[::-1] if end else v) for v in S]
        best = None
        for r in range(min(refs, len(D))):
            ref = D[r]
            lags = []
            for d in D:
                c = []
                for l in range(-search, search + 1): # d[k + l] against ref[k], over all they overlap (a run at the very
                    r0, d0 = max(0, -l), max(0, l) #   start, a postamble's in the reversed pass, counts too)
                    n = min(len(ref) - r0, len(d) - d0)
                    if n <= min_len:
                        c.append(0.0)
                        continue
                    pr = np.conj(ref[r0:r0 + n])*d[d0:d0 + n]
                    cs = np.concatenate(([0], np.cumsum(pr)))
                    c.append(np.real(cs[min_len:] - cs[:-min_len]).max()) # the best window of min_len steps (real:
                    # steps drop each burst's phase, so a true alignment adds up positive; a periodic preamble one symbol
                    # off adds up as large, negative)
                lags.append(int(np.argmax(c)) - search)
            lo = max(0, -min(lags))
            L = min(len(d) - l - lo for d, l in zip(D, lags))
            if L <= min_len:
                continue
            rows = np.array([d[l + lo:l + lo + L] for d, l in zip(D, lags)])
            mean = rows.mean(axis=0)
            m = np.mean(np.abs(np.angle(rows*np.conj(mean)[None, :])) < np.pi/4, axis=0) # share near the mean step
            m[np.abs(mean) < 0.3] = 0 # (no direction to be near)
            if best is None or np.sum(m >= share) > best[0]:
                best = (int(np.sum(m >= share)), m, mean, lo)
        if best is None:
            continue
        _, m, mean, lo = best
        ok = m >= share
        gap = ~ok & (m >= 0.5) # a step or two where a weak burst slipped, between agreeing ones: bridged
        for g in (1, 2):
            fill = np.zeros_like(ok)
            for i in range(1, len(ok) - g):
                if ok[i - 1] and ok[i + g] and gap[i:i + g].all():
                    fill[i:i + g] = True
            ok |= fill
        on = np.concatenate(([0], ok.astype(np.int8), [0]))
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
    # The same run seen from both ends (bursts of one length): the start-anchored one
    starts = [g for g in out if g.offset >= 0]
    keep = starts + [g for g in out if g.offset < 0 and all(abs(len(g.steps) - len(k.steps)) > 8 for k in starts)]
    return sorted(keep, key=lambda g: -len(g.steps))

def _run(c, thr):
    """(start, length) of the longest circular run of c above thr"""
    on = c > thr
    if on.all():
        return 0, len(c)
    k = int(np.argmin(on)) # start the scan at an off position, so a run across the wrap is whole
    r = np.roll(on, -k)
    best, cur = (0, 0), 0
    for i, v in enumerate(r):
        cur = cur + 1 if v else 0
        if cur > best[1]:
            best = ((i - cur + 1 + k) % len(c), cur)
    return best

def _fold_coherence(y, lag):
    """frames.coherence by one fold: |sum over frames of y[n] conj(y[n+lag])| over the sum of magnitudes, at each
    position n mod lag"""
    pr = y[:-lag]*np.conj(y[lag:])
    k = np.arange(len(pr)) % lag
    num = np.abs(np.bincount(k, pr.real, lag) + 1j*np.bincount(k, pr.imag, lag))
    return num/(np.bincount(k, np.abs(pr), lag) + 1e-30)

def periodic_word(burst_list, mod, points, tries=4, min_frames=6, max_lag=20000, thr=0.8, min_len=4):
    """(unique word on the constellation, period in symbols) when long bursts hold frames with one unique word, each at
    its own phase; else None. Every lag from 16 to a `min_frames`-th of the shortest of the `tries` longest bursts is
    folded (`_fold_coherence`: O(n) a lag); at a lag, each burst's longest circular run of positions coherent from
    frame to frame above `thr` (over 8 frames a position passes 0.8 by chance one time in 170, so a run of `min_len`
    over every position and lag is rare, where a unique word's sit near 1). The first lag where two bursts hold such a
    run, of the same length and phase steps, is the period (its multiples pass too); the word is that run's symbols
    averaged over one burst's frames. A word of U symbols every P adds only U/P to an autocorrelation, which a comb
    over it finds about half the time on bursts of 8 frames; the fold finds it every time"""
    L = sorted((b for b in burst_list if b.symbols is not None), key=lambda b: -len(b.symbols))[:tries]
    if len(L) < 2:
        return None
    from ..framing.frames import _normalized
    n = len(L[-1].symbols)
    Y = [_normalized(b.symbols[:n]) for b in L]
    pts = np.asarray(points, dtype=np.complex128)
    for lag in range(16, min(max_lag, n//min_frames) + 1):
        runs = []
        for y in Y:
            st, ln = _run(_fold_coherence(y, lag), thr)
            if ln >= min_len:
                F = len(y)//lag
                fr = np.array([np.roll(y[f*lag:(f + 1)*lag], -st)[:ln] for f in range(F)])
                rot = np.exp(-1j*np.angle(np.sum(fr*np.conj(fr[0])[None, :], axis=1)))
                runs.append(np.mean(fr*rot[:, None], axis=0))
        for w in runs:
            same = [v for v in runs if abs(len(v) - len(w)) <= 2 and
                    abs(np.mean(_unit_steps(v[:min(len(v), len(w))])*np.conj(_unit_steps(w[:min(len(v), len(w))])))) > 0.8]
            if len(same) >= 2:
                uw = w/np.sqrt(np.mean(np.abs(w)**2))*np.sqrt(np.mean(np.abs(pts)**2))
                return pts[np.argmin(np.abs(uw[:, None] - pts[None, :]), axis=1)], int(lag)
    return None

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

class Units:
    """`units`: unit (symbols), err (its standard error: the lengths are known to a few symbols, so a unit is often
    only known to one either way), overhead (symbols besides the units: preamble, header, tail), counts ({units a burst:
    bursts}), fit (share of bursts within `tol` of overhead + n*unit)"""
    __slots__ = ("unit", "err", "overhead", "counts", "fit")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return (f"<{self.unit}-symbol units (+-{self.err:.1f}), {self.overhead} symbols besides; bursts of "
                f"{', '.join(f'{n} ({c})' for n, c in sorted(self.counts.items()))} units; fit {self.fit:.2f}>")

def units(lengths, tol=3, min_unit=16, hold=0.9):
    """The block a burst's length is made of: lengths (symbols a burst, e.g. len(b.symbols) of a kind's bursts) as
    overhead + n*unit, n the units in that burst (1, 2, 3...: MF-TDMA slots of one, two, three code blocks), or None.
    Lengths are known to within the gate's slack (`tol` symbols), so the unit is the largest one their differences are
    all near multiples of (in `hold` of the pairs), with at least two different multiples among them (a unit longer
    than every difference fits trivially), then unit and overhead refitted by least squares. What lengths cannot tell:
    the overhead is the least that fits (100 symbols and 84-symbol units read as 16 and one unit more), and a unit
    whose bursts all hold an even number of it reads as twice the size"""
    Ls = np.sort(np.asarray(lengths, dtype=np.float64))
    if len(Ls) < 3:
        return None
    d = np.abs(Ls[:, None] - Ls[None, :])[np.triu_indices(len(Ls), 1)]
    big = d[d > 2*tol]
    if len(big) < 2:
        return None
    for u in range(int(big.max()), min_unit - 1, -1):
        m = np.round(d/u)
        near = np.abs(d - m*u) <= 2*tol
        if np.mean(near) < hold or not np.any(m >= 1):
            continue
        n0 = np.floor((Ls - Ls[0])/u + 0.5) # units beyond the shortest burst's
        if len(np.unique(n0)) < 2:
            continue
        # the shortest burst's own units: as many as fit beside a non-negative overhead
        base = int(np.floor((Ls[0] + tol)/u))
        best = None
        for b0 in range(base, 0, -1):
            n = n0 + b0
            over = np.median(Ls - n*u)
            fit = float(np.mean(np.abs(Ls - n*u - over) <= tol))
            if over >= -tol and (best is None or fit > best[2]):
                best = (n, over, fit)
                break
        if best is None:
            continue
        n, over, fit = best
        if fit < hold:
            continue
        # unit and overhead refitted by least squares on the bursts that fit (the search steps a symbol at a time, and
        # the largest unit within tolerance runs high)
        uf, of = float(u), float(over)
        for _ in range(4):
            ok = np.abs(Ls - n*uf - of) <= tol
            if ok.sum() < 3:
                break
            (uf, of), *_ = np.linalg.lstsq(np.vstack((n[ok], np.ones(ok.sum()))).T, Ls[ok], rcond=None)
        ok = np.abs(Ls - n*uf - of) <= tol
        r = Ls[ok] - n[ok]*uf - of
        sn = np.sum((n[ok] - n[ok].mean())**2)
        err = float(np.sqrt(np.sum(r**2)/max(ok.sum() - 2, 1)/sn)) if sn > 0 else float("inf")
        u, over = int(round(uf)), int(round(of))
        fit = float(np.mean(np.abs(Ls - n*u - over) <= tol))
        vals, cnt = np.unique(n.astype(int), return_counts=True)
        return Units(unit=u, err=err, overhead=over, counts={int(v): int(c) for v, c in zip(vals, cnt)}, fit=fit)
    return None

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
            y = np.concatenate((track(y[:oe][::-1], pts)[::-1], y[oe:oe + n], track(y[oe + n:], pts)))
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
    from .waveform import on_off
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

class NoPlan(ValueError):
    """learn_plan found no frame period; `scores`: what it tried (as a Guess's, for rp.viz.plan_evidence)"""
    scores = None

def learn_plan(bursts, quantum=None, max_btus=4096, clock=27e6, purity=0.95, tol=0.1, carriers=None):
    """A burst time plan (rpps.plan) from detected bursts, for a network whose forward link is not heard: its BTU, frame
    period, a frame type per carrier, its slots and their tx types. Returns a Guess to `save` (config/plan/guess/<name>)
    and load as a plan (`rp.plan.load("guess/<name>")`), whose `plan` is the Plan itself and `scores` what it was chosen
    on (each BTU's score, each frame period's purity and why it was passed over: rp.viz.plan_evidence draws them).

    bursts: rp.sync.Bursts' output (each its ctx["burst"]: carrier freq, start; its length) or dicts {freq, start, length
    [s], kind (a burst type: its unique word), rate}. quantum: the BTU [s] when known. carriers: [{freq, width, rate}]
    the bursts were detected on (rp.sync.Bursts'): each frame type's bandwidth and symbol rate.

      1. The BTU: the quantum of the bursts' start differences (every slot starts on the grid of its frame): the largest q
         whose multiples the differences sit on, to `tol` of q, then made exact over the whole capture (each start's
         count of BTUs, over a span doubling from the first burst: the slope of a line through them).
      2. The frame: the shortest period of M BTUs at which every (carrier, BTU in the frame) holds one kind of burst
         (`purity` of them; length clusters stand for kinds not given), with bursts at two or more places of it.
      3. A frame type per carrier: its slots where bursts were seen (as many BTUs as the longest burst of the kind needs),
         single-BTU slots between; a tx type per kind (`kind` in its fields, and `word` when the kind is a word's name), its offset the bursts' median offset from
         the slot's start. The superframe starts at the first burst's frame start.
    Only what the bursts show is learned: slots never used are single BTUs, carriers never used are not in it."""
    from ..result import Guess
    from ...plan import Plan, TxType, FrameType, Sequence, Frame
    rec = _records(bursts)
    if len(rec) < 4:
        raise ValueError("learn_plan: a few bursts at least")
    starts = np.array([r["start"] for r in rec])
    # Kinds: given, else clusters of length per carrier (a new one where the sorted lengths jump by more than 10%)
    for fr in {r["freq"] for r in rec if r["kind"] is None}:
        mine = sorted((r for r in rec if r["freq"] == fr and r["kind"] is None), key=lambda r: r["length"])
        c, last = 0, None
        for r in mine:
            if last is not None and r["length"] > 1.1*last:
                c += 1
            r["kind"], r["auto"], last = f"{fr:g}/{c}", True, r["length"]
    curve = None
    if quantum:
        q = float(quantum)
    else:
        q, curve = _quantum(starts, tol)
        q = _stretch(starts, q, tol)
    t0 = float(starts.min())
    k = np.round((starts - t0)/q).astype(np.int64)
    off = (starts - t0) - k*q
    need = {}
    for r in rec: # BTUs each kind's longest burst takes
        need[str(r["kind"])] = max(need.get(str(r["kind"]), 1), int(np.ceil(r["length"]/q - 1e-9)))
    best, tried = None, [] # tried: (M, purity, why not: "" taken, "purity", "repeats", "short", "overlap")
    for M in range(2, max_btus + 1):
        cells = {}
        for r, kk in zip(rec, k):
            cells.setdefault((r["freq"], int(kk % M)), []).append(r["kind"])
        major = sum(max(np.unique([str(x) for x in v], return_counts=True)[1]) for v in cells.values())
        repeated = sum(len(v) > 1 for v in cells.values())
        why = ("purity" if major < purity*len(rec) else "repeats" if repeated < 2 else
               "short" if M*q < max(r["length"] for r in rec) else "overlap" if _overlap(cells, need, M) else "")
        tried.append((M, major/len(rec), why)) # (the overlap check only where the rest pass: it costs most)
        if why:
            continue
        best = (M, cells)
        break
    if best is None:
        e = NoPlan("learn_plan: no frame period found (too few bursts, or no plan): rp.viz.plan_evidence(None, error) "
                   "shows why each was passed over")
        e.scores = {"quantum": curve, "btu": q, "tol": tol, "purity": purity, "frames": [(M_*q, pur, why) for M_, pur, why in tried],
                    "frame": None}
        raise e
    M, cells = best
    # The frame's start: where no carrier's slot crosses from one frame into the next (the first burst need not be first)
    slots = [(b, need[max(set(str(x) for x in v), key=[str(x) for x in v].count)]) for (f, b), v in cells.items()]
    shift = next((r for r in range(M) if all((b - r) % M + n <= M for b, n in slots)), 0)
    t0 += shift*q
    k = np.round((starts - t0)/q).astype(np.int64)
    off = (starts - t0) - k*q
    cells = {}
    for r, kk in zip(rec, k):
        cells.setdefault((r["freq"], int(kk % M)), []).append(r["kind"])
    kinds = sorted({str(r["kind"]) for r in rec})
    tx_id = {kd: 128 + i for i, kd in enumerate(kinds)}
    longest = {kd: max(r["length"] for r in rec if str(r["kind"]) == kd) for kd in kinds}
    btus = {kd: max(1, int(np.ceil(longest[kd]/q - 1e-9))) for kd in kinds}
    offs = {kd: float(np.median([o for r, o in zip(rec, off) if str(r["kind"]) == kd])) for kd in kinds}
    T = lambda s: int(round(s*clock))
    word = {str(r["kind"]): r["kind"] for r in rec if isinstance(r["kind"], str) and not r.get("auto")} # (a word's name)
    txs = {tx_id[kd]: TxType(tx_id[kd], btus[kd], max(T(offs[kd]), 0), {"kind": kd, **({"word": word[kd]} if kd in word else {})})
           for kd in kinds}
    fts, frames = {}, []
    for ci, fr in enumerate(sorted({r["freq"] for r in rec})):
        here = sorted((b_, max(set(v), key=v.count)) for (f_, b_), v in cells.items() if f_ == fr)
        sections, t = [], 0
        for b_, kd in here:
            kd = str(kd)
            if b_ < t:
                continue # (a slot inside the one before: the purity let it through)
            if b_ > t:
                sections.append([0, b_ - t - 1])
            sections.append([tx_id[kd], 0])
            t = b_ + btus[kd]
        if t < M:
            sections.append([0, M - t - 1])
        c = next((c for c in (carriers or []) if abs(float(c["freq"]) - fr) < 1.0), {})
        rate = c.get("rate") or next((r["rate"] for r in rec if r["freq"] == fr and r["rate"]), None) or 1.0/q
        fts[ci + 1] = FrameType(ci + 1, T(M*q), T(q), float(c.get("width") or rate), float(rate), M, [0.0],
                                [tuple(s) for s in sections if s[1] >= 0])
        frames.append(Frame(ci + 1, 0, fr))
    seq = Sequence(0, T(t0), T(M*q), 0.0, 0, frames)
    plan = Plan(clock, txs, fts, {0: seq}, name="learned")
    for ft in fts.values():
        ft.slots(txs) # (checked: the sections fill the frame)
    g = Guess("plan", plan.definition("burst time plan learned from detected bursts (rpps.analysis.spectrum.burst.learn_plan)"),
              {"btu": q, "frame": M*q, "bursts": len(rec), "carriers": len(fts), "kinds": len(kinds)})
    g.plan = plan
    g.scores = {"quantum": curve, "btu": q, "tol": tol, "purity": purity,
                "frames": [(M_*q, pur, why) for M_, pur, why in tried], "frame": M*q}
    return g

def _records(bursts):
    """Detected bursts as dicts {freq, start, length [s], kind, rate}: from rp.sync.Bursts' output (its ctx["burst"]) or
    dicts of those keys"""
    rec = []
    for b in bursts:
        f = getattr(b, "ctx", None)
        if f is not None:
            g = f.get("burst", {})
            rec.append({"freq": float(g["freq"]), "start": float(g["start"]), "length": len(b)/b.meta.Fs,
                        "kind": g.get("word") or g.get("tx_type"), "rate": None})
        else:
            rec.append({"freq": float(b["freq"]), "start": float(b["start"]), "length": float(b["length"]),
                        "kind": b.get("kind"), "rate": b.get("rate")})
    return rec

def _overlap(cells, need, M):
    """Whether the slots of a frame of M BTUs collide: on a carrier, a burst's BTUs reaching into the next burst's"""
    by = {}
    for (f, b), v in cells.items():
        kd = max(set(str(x) for x in v), key=[str(x) for x in v].count)
        by.setdefault(f, []).append((b, need[kd]))
    for slots in by.values():
        slots.sort()
        for (b, n), (b2, _) in zip(slots, slots[1:] + [(slots[0][0] + M, 0)]):
            if b + n > b2:
                return True
    return False

def _stretch(starts, q, tol, grid=1e-4):
    """q to the precision the whole capture holds: the starts' BTU counts from the first burst, counted over a span
    that doubles, and q the least-squares slope of the starts on them (the starts more than tol*q off their count left
    out). _quantum's q is right to its grid, `grid` of q: over thousands of BTUs that is half of one, so the first span
    is as many BTUs as that error leaves within half the tolerance, and each fit makes the next span's count right"""
    s = np.sort(np.asarray(starts, dtype=np.float64))
    t0, span = s[0], 0.5*tol/grid*q
    while True:
        d = s[s - t0 <= span] - t0
        k = np.round(d/q)
        good = np.abs(d - k*q) <= tol*q
        if good.sum() >= 4 and np.ptp(k[good]) >= 32:
            q = float(np.polyfit(k[good], d[good], 1)[0])
        if span >= s[-1] - t0:
            return q
        span *= 2

def _quantum(starts, tol, chunk=1024, most=1 << 21):
    """The largest q whose multiples the start differences sit on (to tol*q): scored by the mean cosine of 2 pi d / q
    over the differences d of nearby bursts (each to its next 8), from the longest q down. A difference of N BTUs
    passes only within about tol/N of q: so q steps evenly in 1/q, by tol/4 of the longest differences' (sparse bursts
    sit hundreds of BTUs apart, their peaks too narrow for a fixed grid), evaluated `chunk` at a time down to the first
    that passes (at most `most` of them). Returns q and the curve (q's tried, their scores)"""
    s = np.sort(starts)
    d = np.concatenate([s[i + 1:i + 9] - s[i] for i in range(len(s))])
    d = d[d > 0]
    if not len(d):
        raise ValueError("learn_plan: the bursts start together")
    if len(d) > 8192: # (a spread sample: the score is a mean)
        d = np.sort(d)[np.linspace(0, len(d) - 1, 8192).astype(int)]
    lo = max(np.min(d)/64, 1e-7)
    hi = float(np.median(d))
    df = tol/(4*float(np.percentile(d, 99)))
    thr = np.cos(2*np.pi*tol)
    qs, score, found = [], [], None
    f = 1/hi
    while f < 1/lo and len(qs)*chunk < most:
        fr = f + df*np.arange(chunk)
        fr = fr[fr < 1/lo]
        sc = np.cos(2*np.pi*np.outer(fr, d)).mean(axis=1)
        qs.append(1/fr)
        score.append(sc)
        good = np.nonzero(sc >= thr)[0]
        if len(good):
            found = 1/fr[good[0]]
            break
        f = fr[-1] + df
    qs, score = np.concatenate(qs), np.concatenate(score)
    if found is None:
        raise ValueError("learn_plan: the bursts' starts sit on no grid (give `quantum`)")
    # Refine: the q whose cosine sum is largest over the peak the first passing q is on the edge of (its half-width
    # about tol*q/d over the typical difference d), below it (the descent comes from above)
    width = 2*tol*found*found/float(np.median(d))
    fine = np.linspace(found + 0.25*width, found - width, 401)
    return float(fine[np.argmax(np.cos(2*np.pi*np.outer(1/fine, d)).sum(axis=1))]), (qs, score)

# ---- how well a plan explains the bursts

SLOT, KIND, LONG, NONE = "slot", "kind", "long", "none"

class PlanCheck:
    """check_plan's answer: how well a burst time plan explains the bursts detected.

    plan; origin (plan clock ticks at the capture's first sample) and rate (the network clock's offset from the
    capture's, a fraction): a capture time t [s] is plan tick origin + t*clock*(1 + rate); cf (the capture's centre in
    the plan's frequencies); bursts (the bursts as records: freq, start, length, kind); per burst its `status`:
      "slot"  in a slot of its carrier (its start to `tol` of the slot's burst start), of its kind, and fitting it
      "kind"  in a slot whose tx type is another kind's (its unique word, `kind` or `word` in the tx type's fields)
      "long"  in a slot, but running past the slot's end by more than `tol`
      "none"  no slot of its carrier starts near it: the plan does not explain it
    its `window` (rpps.plan.Window; None where "none") and `error` (s: its start minus its window's burst start, nan
    where "none"); `windows`, every window starting within the bursts' span, and `filled`, how many bursts each holds"""
    __slots__ = ("plan", "origin", "rate", "cf", "tol", "bursts", "status", "window", "error", "windows", "filled")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def tick(self, t):
        """The plan's clock at capture time t [s]"""
        return self.origin + np.asarray(t)*self.plan.clock*(1 + self.rate)

    def time(self, tick):
        """The capture time [s] of plan tick `tick`"""
        return (np.asarray(tick) - self.origin)/(self.plan.clock*(1 + self.rate))

    @property
    def counts(self):
        return {s: sum(1 for x in self.status if x == s) for s in (SLOT, KIND, LONG, NONE)}

    @property
    def explained(self):
        """The fraction of the bursts in a slot of their own kind that they fit"""
        return self.counts[SLOT]/max(len(self.status), 1)

    @property
    def rms(self):
        """The rms start error [s] of the bursts in slots"""
        e = np.asarray([e for e, s in zip(self.error, self.status) if s != NONE])
        return float(np.sqrt(np.mean(e**2))) if len(e) else float("nan")

    def __str__(self):
        c, n = self.counts, len(self.status)
        used = int(np.sum(self.filled > 0)) if self.filled is not None else 0
        return (f"{c[SLOT]} of {n} bursts in a slot of their kind, {c[KIND]} in another kind's, {c[LONG]} overrunning "
                f"theirs, {c[NONE]} in none; start error {1e6*self.rms:.1f} us rms; {used} of "
                f"{len(self.windows or [])} slots used; clock {1e6*self.rate:+.1f} ppm")

def check_plan(plan, bursts, origin=None, rate=0.0, cf=0.0, tol=None, fit=True, votes=48):
    """How well a burst time plan (rpps.plan: learned, a definition or from a forward link) explains detected bursts:
    each burst placed in the slot of its carrier whose burst start is nearest its own, and judged (PlanCheck: "slot",
    "kind", "long", "none"). A right plan puts every burst in a slot of its kind; a wrong BTU or frame leaves bursts in
    none, or drifting through their slots (`error` growing with time); a wrong slot layout puts them in other kinds'.

    bursts: rp.sync.Bursts' output or dicts {freq [Hz from the capture's centre], start [s], length [s], kind}, as
    learn_plan takes. origin: plan ticks at the capture's first sample (a forward link's clock reference); None: voted
    for (known only to a whole superframe: the one nearest the capture's start is taken), as rp.sync.Bursts acquires it (each of `votes` bursts spread over the capture votes for the origins that put it
    on a slot of its carrier, of its kind when both are known; the most voted wins). rate: the network clock's offset.
    fit: refine the origin and rate on the bursts placed (a line through their start errors, its slope taken where it is
    over three standard errors), placing them again.
    cf: the capture's centre in the plan's frequencies. tol: how far [s] a burst's start may be from its slot's (None:
    half the shortest BTU). Plans whose sequences have different superframe durations are voted on the longest."""
    from ...sync import track
    rec = _records(bursts)
    if not rec:
        raise ValueError("check_plan: no bursts")
    clock = plan.clock
    if tol is None:
        tol = 0.5*min(ft.btu_duration for ft in plan.frame_types.values())/clock
    rate = float(rate)
    if origin is None:
        origin = _vote_origin(plan, rec, cf, tol, votes, track)
    if origin is None: # (no slot on any burst's carrier)
        nan = float("nan")
        return PlanCheck(plan=plan, origin=nan, rate=rate, cf=cf, tol=tol, bursts=rec, status=[NONE]*len(rec),
                         window=[None]*len(rec), error=np.full(len(rec), nan), windows=[], filled=np.zeros(0, int))
    origin = float(origin)
    for it in range(3 if fit else 1):
        out = _place(plan, rec, origin, rate, cf, tol)
        if not fit or it == 2:
            break
        ok = [(r["start"], e) for r, s, e in zip(rec, out[0], out[2]) if s != NONE]
        if not ok:
            break
        t, e = np.asarray(ok).T
        a, b = float(np.mean(e)), 0.0
        if len(t) >= 8 and np.ptp(t) > 0: # (the rate only where the start errors lean beyond their scatter)
            (b1, a1), cov = np.polyfit(t, e, 1, cov=True)
            if abs(b1) > 3*np.sqrt(cov[0, 0]):
                a, b = a1, b1
        # e = (tick(t) - window)/clock = a + b t: the window's tick is origin - a clock + t clock (1 + rate - b)
        origin, rate = origin - a*clock, rate - b
    status, window, error, windows, filled = out
    return PlanCheck(plan=plan, origin=origin, rate=rate, cf=cf, tol=tol, bursts=rec, status=status, window=window,
                     error=error, windows=windows, filled=filled)

def _kind_of(w, plan):
    tx = plan.tx_types.get(w.tx_type)
    f = tx.fields if tx else {}
    return f.get("word") or f.get("kind")

def _same_kind(kind, w, plan):
    """Whether a burst of `kind` (None: unknown) may be in window w"""
    if kind is None:
        return True
    k = _kind_of(w, plan)
    return k is None or str(k) == str(kind) or str(kind) == str(w.tx_type)

def _rank(kind, w, plan):
    """How well a burst of `kind` fits window w's type: 0 its own kind, 1 a window of no kind (a single BTU an
    assignment types, a learned plan's unused BTU; or the burst's kind unknown), 2 another kind's"""
    if kind is None:
        return 1
    k = _kind_of(w, plan)
    if str(kind) == str(w.tx_type) or (k is not None and str(k) == str(kind)):
        return 0
    return 1 if k is None else 2

def _vote_origin(plan, rec, cf, tol, votes, track):
    clock = plan.clock
    span = lambda s: s.duration*(len(s.hop.channels) if s.hop is not None else 1)
    P = max(span(s) for s in plan.sequences.values() if s.duration > 0)
    ws = [w for s in plan.sequences.values() if s.duration > 0 for w in plan.windows(s.start, s.start + span(s), sequences=[s.id])]
    idx = np.unique(np.linspace(0, len(rec) - 1, min(votes, len(rec))).round().astype(int))
    C, B = [], []
    for j, i in enumerate(idx):
        r = rec[i]
        mine = [w for w in ws if abs(w.freq - cf - r["freq"]) <= 0.5*max(w.bw, 1.0)]
        ranks = [_rank(r["kind"], w, plan) for w in mine]
        best = min(ranks, default=2)
        for w, rk in zip(mine, ranks): # (its own kind's windows where the carrier has them: an untyped one too would
            if rk == best and rk < 2: # let the origin slip a BTU onto the filler around them)
                C.append(w.burst - r["start"]*clock)
                B.append(j)
    if not C:
        return None
    C, B = np.asarray(C), np.asarray(B)
    first = np.r_[0, np.nonzero(np.diff(B))[0] + 1]
    T = tol*clock
    best, n_best = 0, -1
    for a in range(0, len(C), 256): # (each candidate: how many bursts have a candidate within T of it)
        hit = (np.abs(track.wrap(C[None, :] - C[a:a + 256, None], P)) <= T).astype(np.uint8)
        n = np.maximum.reduceat(hit, first, axis=1).sum(axis=1)
        k = int(np.argmax(n))
        if n[k] > n_best:
            best, n_best = a + k, int(n[k])
    d = track.wrap(C - C[best], P)
    near = [d[(B == j) & (np.abs(d) <= T)] for j in range(len(idx))]
    near = [x[np.argmin(np.abs(x))] for x in near if len(x)]
    return float(track.wrap(C[best] + np.mean(near), P)) # (to a whole superframe: the one nearest the capture's start)

def _place(plan, rec, origin, rate, cf, tol):
    clock = plan.clock
    T = tol*clock
    k = clock*(1 + rate)
    starts = np.asarray([r["start"] for r in rec])
    longest = max(ft.duration for ft in plan.frame_types.values())
    t0, t1 = origin + starts.min()*k, origin + starts.max()*k
    ws = plan.windows(t0 - longest - T, t1 + longest + T)
    span = np.asarray([t0 - T <= w.burst <= t1 + T for w in ws], dtype=bool)
    by = {}
    for i, w in enumerate(ws):
        by.setdefault(w.freq, []).append(i)
    by = {f: (np.asarray([ws[i].burst for i in v], dtype=np.float64), np.asarray(v)) for f, v in by.items()}
    bw = {f: max(ws[v[0]].bw, 1.0) for f, (_, v) in by.items()}
    status, window, error = [], [], np.full(len(rec), np.nan)
    count = np.zeros(len(ws), int)
    for n, r in enumerate(rec):
        tick = origin + r["start"]*k
        best = None # (rank, distance, window): its own kind's window first, of those within the tolerance
        for f, (b, v) in by.items():
            if abs(f - cf - r["freq"]) > 0.5*bw[f]:
                continue
            j = int(np.searchsorted(b, tick))
            for jj in range(max(j - 2, 0), min(j + 2, len(b))):
                d = abs(b[jj] - tick)
                if d > T:
                    continue
                cand = (_rank(r["kind"], ws[v[jj]], plan), d, v[jj])
                if best is None or cand[:2] < best[:2]:
                    best = cand
        if best is None:
            status.append(NONE)
            window.append(None)
            continue
        w = ws[best[2]]
        count[best[2]] += 1
        error[n] = (tick - w.burst)/clock
        end = origin + (r["start"] + r["length"])*k
        if not _same_kind(r["kind"], w, plan):
            status.append(KIND)
        elif end > w.start + w.length + 2*T:
            status.append(LONG)
        else:
            status.append(SLOT)
        window.append(w)
    keep = np.nonzero(span)[0]
    return status, window, error, [ws[i] for i in keep], count[keep]
