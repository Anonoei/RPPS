"""The finder's steps 4-6 (rpps.analysis.spectrum.find): a line raised from the floor; a signal comes up and holds
its sides

In a region the detector marked, a line is raised from the floor a step at a time (STEP_K noise deviations, at least
STEP_DB), and at each the runs over it are found. A run whose sides hold from one step to the next (to HOLD of its
width) is one thing going on up; a run that comes up out of a wider one, or out of the floor, is a new thing standing
on it. A thing is what held its sides for a step or more (a hump's move in at every step: it is none) and is not only
what comes up out of it (FILL of it or more: two carriers side by side come up as one run and part at the notch; a
carrier's top comes up out of it); what stands on a pedestal leaves most of it. Its top is the middle of its own
points over where its sides last held (not those of what stands on it), its sides where it stands halfway up from
where it came up (or drop_db under its top). Lines (LINE_BINS wide or less) are taken out first (a median over twice that): no
signal. A box passes the detector at its width and has an SNR of MIN_SNR or more.

In a Finder each region's reading is kept from step to step: one whose span and level hold keeps its things, measured
again, and is read again when they move or once a window (see boxes).
"""
import numpy as np

from ...math.mask import runs

STEP_K = 3.0       # the line's step [noise deviations of the spectrum read: finer breaks sides on the noise,
                   # coarser misses weak signals]
STEP_DB = 1.0      # and at least [dB]
HOLD = 0.05        # a run's sides hold when they move under this of its width (2 points at least) in a step
FILL = 0.75        # a thing whose sides what comes up out of it fills this much of is only them
LINE_BINS = 4
MIN_SNR = 3.0      # [dB]

DEBUG = None # a list: boxes() appends why it dropped each box (reason, lo, hi [Hz])

def _why(c, reason, a, b):
    if DEBUG is not None:
        DEBUG.append((reason, (a - c.nfft//2)*c.bin_hz, (b - c.nfft//2)*c.bin_hz))

def _read(y, step, drop_db=10.0):
    """The things in y (dB over the floor at points), the line raised `step` dB at a time: dicts lo, hi (sides,
    points), top, base (dB: where it came up), over (index of the thing it stands on, or None); what others stand on
    before them"""
    things = []
    open_ = [] # (the things going on up: their run's sides, the index in things)
    L = step
    top_y = float(np.max(y)) if len(y) else 0.0
    while L < top_y:
        s0, s1 = runs(y > L)
        wide = (s1 - s0) > LINE_BINS # (a run as narrow as a line is noise or a line: nothing to follow)
        us = np.array([u for (u, v, t) in open_], int)
        new_open = []
        for a, b in zip(s0[wide].tolist(), s1[wide].tolist()):
            # (the thing this run came up in, or out of: the open one holding it; the open runs are apart, in order)
            j = int(np.searchsorted(us, a, side="right")) - 1
            k = open_[j][2] if j >= 0 and b <= open_[j][1] else None
            if k is not None:
                t = things[k]
                tol = max(2, HOLD*(t["s1"] - t["s0"]))
                if abs(a - t["s0"]) <= tol and abs(b - t["s1"]) <= tol: # (its sides hold: it goes on up)
                    t["held"] += 1
                    t["s0"], t["s1"], t["at"] = a, b, L
                    new_open.append((a, b, k))
                    continue
            # (a new thing, come up out of the thing k or the floor)
            things.append(dict(s0=a, s1=b, lo0=a, hi0=b, base=L - step, at=L, over=k, held=0))
            new_open.append((a, b, len(things) - 1))
        open_ = new_open
        L += step
    # Which are things: held a step or more, and not only what comes up out of them
    keep = [False]*len(things)
    for i in reversed(range(len(things))): # (a thing's kids come up after it: judged first)
        t = things[i]
        kids = [u for j, u in enumerate(things) if u["over"] == i and keep[j]]
        if kids and sum(u["hi0"] - u["lo0"] for u in kids) >= FILL*(t["s1"] - t["s0"]):
            # (it is only what comes up out of it: things side by side, or its own top; they take its place)
            l, r = min(u["lo0"] for u in kids), max(u["hi0"] for u in kids)
            for u in kids:
                u["over"], u["base"] = t["over"], t["base"]
                if u["lo0"] == l:
                    u["lo0"] = min(u["lo0"], t["lo0"])
                if u["hi0"] == r:
                    u["hi0"] = max(u["hi0"], t["hi0"])
            t["held"] = -1
        keep[i] = t["held"] >= 1
    out, ids = [], {}
    for i, t in enumerate(things):
        if not keep[i]:
            continue
        over = t["over"]
        while over is not None and not keep[over]:
            over = things[over]["over"]
        # (its top: the middle of its own points over where its sides last held, not those of what stands on it)
        seg = y[t["lo0"]:t["hi0"]]
        own = seg > t["at"]
        for j, u in enumerate(things):
            if keep[j] and u["over"] == i:
                own[max(0, u["lo0"] - t["lo0"]):max(0, u["hi0"] - t["lo0"])] = False
        top = float(np.median(seg[own])) if own.any() else t["at"]
        thr = max(top - drop_db, 0.5*(top + t["base"]))
        pk = t["lo0"] + int(np.argmax(np.where(own, seg, -np.inf))) if own.any() else t["lo0"] + int(np.argmax(seg))
        lo, hi = _side(y, pk, t["lo0"] - 1, -1, thr), _side(y, pk, t["hi0"], 1, thr)
        ids[i] = len(out)
        out.append(dict(lo=lo, hi=hi, top=top, base=t["base"], over=over))
    for t in out:
        t["over"] = ids.get(t["over"]) if t["over"] is not None else None
    order = sorted(range(len(out)), key=lambda i: _depth(out, i)) # (what others stand on first)
    re = {old: new for new, old in enumerate(order)}
    out = [out[i] for i in order]
    for t in out:
        t["over"] = re[t["over"]] if t["over"] is not None else None
    return out

def _depth(things, i):
    d = 0
    while things[i]["over"] is not None:
        i, d = things[i]["over"], d + 1
    return d

def _side(y, pk, stop, step, thr):
    """Where y first rises over thr coming in from `stop` (exclusive) toward the peak pk (fractional)"""
    seg = y[pk:stop][::-1] if step > 0 else y[stop + 1:pk + 1]
    if not len(seg):
        return float(pk)
    k = int(np.argmax(seg > thr)) if (seg > thr).any() else len(seg) - 1
    far = stop - 1 if step > 0 else stop + 1
    i = far - step*k # (the first point over it)
    j = i + step     # (the one before, under it)
    if 0 <= j < len(y) and y[i] > y[j]:
        return i + step*float(np.clip((y[i] - thr)/(y[i] - y[j]), 0, 1))
    return float(i)

def boxes(c, occ, groups, min_snr=None, state=None):
    """find's steps 4-6 by levels: the Signals of context c (see find._find_frames), the detector's marks occ.
    state: a Finder's: each region's reading kept from step to step; one whose span and level hold (find._same) keeps
    its things, measured again (find._remeasure), and is read again when they move or a window (state["refresh"]
    frames) on, the regions' turns spread over it"""
    from .find import _copies, _dof, _on, _remeasure, _same
    from .signal import Signal
    if min_snr is None:
        min_snr = MIN_SNR
    s0, s1 = runs(occ)
    keep = (s1 - s0) >= max(c.min_bins, LINE_BINS)
    s0, s1 = s0[keep], s1[keep]
    if len(s0) > 1: # (spans a gap of under a quarter of the narrower apart: one region)
        gap = s0[1:] - s1[:-1]
        narrow = np.minimum(s1[:-1] - s0[:-1], s1[1:] - s0[1:])
        new = np.concatenate(([True], gap > np.maximum(4, narrow//4)))
        s0, s1 = s0[new], s1[np.concatenate((new[1:], [True]))]
    recs = state.get("levels", []) if state is not None else []
    now, refresh = (state.get("frame", 0), state.get("refresh")) if state is not None else (0, None)
    out, kept = [], []
    for j, (a, b) in enumerate(zip(s0, s1)):
        a, b = int(a), int(b)
        lvl = 10*np.log10(max(float(c.Z[a:b].mean()), 1e-30))
        rec = _same(recs, a, b, lvl)
        if rec is not None and refresh is not None and now < rec["due"]: # (as it was: its things measured again)
            sigs = _remeasure(c, rec["sig"])
            wait = rec["due"] - now
        else:
            mg = int(np.clip((b - a)//4, 4, 2048))
            r0 = max(c.cut[0], a - mg, (int(s1[j - 1]) + a)//2 if j > 0 else 0)
            r1 = min(c.cut[1], b + mg, (b + int(s0[j + 1]))//2 if j + 1 < len(s0) else c.nfft)
            sigs = _region(c, a, b, r0, r1, Signal, _on, _dof, min_snr)
            # (read again a window on; a region's first time a fraction of one, set by where it is, so the turns
            # spread over the window)
            wait = refresh or 0
            if rec is None and refresh:
                wait = refresh//2 + (a*7919) % max(1, refresh//2)
            lvl = 10*np.log10(max(float(c.Z[a:b].mean()), 1e-30))
        if state is not None:
            kept.append(dict(a=a, b=b, lvl=lvl, sig=sigs, due=now + wait))
        out += _copies(sigs)
    if state is not None:
        state["levels"] = kept
    return out

def _region(c, a, b, r0, r1, Signal, _on, _dof, min_snr):
    """The Signals of one region (the detector's span a..b, read over r0..r1)"""
    if r1 - r0 < LINE_BINS:
        return []
    on, bursts, duty = _on(c, a, b, 1.0)
    Mon = int(on.sum()) if on is not None else c.M
    spec = c.Pw[on, r0:r1].mean(axis=0, dtype=np.float64) if on is not None else c.Z[r0:r1]
    # (a line is no thing to read: a median over 2 LINE_BINS + 1 points takes out what is LINE_BINS wide or less, and
    # the noise with it, keeping the edges of what is wider)
    h = LINE_BINS
    pad = np.concatenate((np.full(h, spec[0]), spec, np.full(h, spec[-1])))
    ys = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2*h + 1), axis=1)
    y = 10*np.log10(np.maximum(ys, 1e-30))
    # (a point's deviation: the median of 2 LINE_BINS + 1 is 1.15 times a mean of as many, measured on noise)
    sd1 = 1.15*4.343/np.sqrt(_dof(c.nfft, c.hop, Mon, 2*LINE_BINS + 1))
    things = _read(y, max(STEP_DB, STEP_K*sd1), c.drop_db)
    sigs = []
    for t in things:
        e0, e1 = r0 + t["lo"], r0 + t["hi"]
        if e1 - e0 < max(c.min_bins, LINE_BINS):
            _why(c, "line", e0, e1)
            sigs.append(None)
            continue
        b0 = int(np.clip(np.floor(e0 + 0.5), 0, c.nfft - 1))
        b1 = int(np.clip(np.ceil(e1 + 0.5), b0 + 1, c.nfft))
        over = sigs[t["over"]] if t["over"] is not None else None
        local = 10**(things[t["over"]]["top"]/10) if over is not None else 1.0 # (what it stands on)
        on_k, bursts_k, duty_k = _on(c, b0, b1, local)
        sel = on_k if on_k is not None else slice(None)
        bp = c.band(b0, b1)[sel]
        m = max(1, int(bp.sum()**2/max(np.sum(bp**2), 1e-30)))
        z = (c.Pw[on_k, b0:b1].mean(axis=0) if on_k is not None else c.Z[b0:b1])/local
        hi_t, _ = c.det.thresholds(_down(len(z)), _down(m))
        if float(np.mean(z)) <= hi_t:
            _why(c, "detector", e0, e1)
            sigs.append(None)
            continue
        snr = 10*np.log10(max(float(np.mean(z)) - 1.0, 1e-30))
        if snr < min_snr:
            _why(c, f"snr {snr:.1f} dB", e0, e1)
            sigs.append(None)
            continue
        lo, hi = (e0 - c.nfft//2)*c.bin_hz, (e1 - c.nfft//2)*c.bin_hz
        band = c.A[b0:b1]
        a_use, b_use = c.cut
        d = int(np.clip((b1 - b0)//32, 1, 8))
        sg = Signal(0.5*(lo + hi), hi - lo, float(np.median(10*np.log10(c.floor[b0:b1]))) + 10*np.log10(local),
                    float(10*np.log10(max(float(z.max()), 1e-30))), float(snr), lo=lo, hi=hi,
                    power=float(10*np.log10(band.sum()*c.bin_hz + 1e-30)),
                    peak_freq=float(c.freq[b0 + int(np.argmax(band))]), duty=duty_k, resolved=(e1 - e0) >= c.min_across,
                    Fs=c.Fs, cf=c.cf, bin_hz=c.bin_hz, clipped=e0 <= a_use + d + 1 or e1 >= b_use - d - 1)
        sg.bursts = bursts_k
        sg.over = over
        if over is not None:
            over.holds = (over.holds or 0) + 1
        sg.holds = 0
        sg.t0, sg.t1 = (bursts_k[0][0], bursts_k[-1][1]) if bursts_k else (0.0, c.T)
        sg.score = float(np.max(c.score[b0:b1]))
        sg._range = (b0, b1, b0, b1, local)
        sg._snr0, sg._excess = sg.snr, float(np.sum(c.Z[b0:b1]/local - 1.0)) # (find._remeasure's start)
        sigs.append(sg)
    return [s for s in sigs if s is not None]

def _down(k):
    """k rounded down to a grid 2^(1/8) apart (exact under 16)"""
    k = int(k)
    if k < 16:
        return max(k, 1)
    return int(2**(np.floor(8*np.log2(k))/8))
