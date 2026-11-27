"""What a TDMA network's bursts share: the preamble (`shared_preamble`, `preamble_symbols`), known symbols elsewhere
(`shared_segments`: a mid-amble, a postamble), a word repeated through each burst (`periodic_word`), and burst kinds,
each with a word of its own (`burst_kinds`). See rpps.analysis.burst"""
import numpy as np

from ...math.mask import runs
from ..framing import frames

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
    if 0 < pc < 1: # (0: underflowed, many bursts: chance agreeing that well is nothing, no longer run needed)
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
        for a, b in zip(*runs(ok)):
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
