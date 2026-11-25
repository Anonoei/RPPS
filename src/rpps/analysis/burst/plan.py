"""A network's burst time plan (rpps.plan) from the bursts rp.sync.Bursts detects: the code blocks a burst is made of
(`units`), the plan learned (`learn_plan`), and whether a plan explains the bursts (`check_plan`). See rpps.analysis.burst"""
import numpy as np

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
         whose multiples the differences sit on, to `tol` of q, made exact over the whole capture (each start's count of
         BTUs, over a span doubling from the first burst: the slope of a line through them), and taken only where 90% of
         the starts sit on its grid (a mean of differences passes near a q they only nearly fit).
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
    g = Guess("plan", plan.definition("burst time plan learned from detected bursts (rpps.analysis.burst.learn_plan)"),
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

def _quantum(starts, tol, chunk=1024, most=1 << 21, on_grid=0.9, peaks=32):
    """The largest q whose multiples the starts sit on (to tol*q). Candidates: q scored by the mean cosine of 2 pi d / q
    over the differences d of nearby bursts (each to its next 8), from the longest q down. A difference of N BTUs
    passes only within about tol/N of q: so q steps evenly in 1/q, by tol/4 of the longest differences' (sparse bursts
    sit hundreds of BTUs apart, their peaks too narrow for a fixed grid), evaluated `chunk` at a time (at most `most`).
    A mean passes near a q the differences only nearly fit (slots of 2.4 ms read as 2.5: 2.4 and 4.8 within tol of 1
    and 2 of it), so each candidate, refined over its peak and made exact over the capture (`_stretch`), stands only
    where `on_grid` of the starts sit on its grid; else the search goes on past its peak (up to `peaks` of them).
    Returns q (exact) and the curve (q's tried, their scores)"""
    s = np.sort(starts)
    d = np.concatenate([s[i + 1:i + 9] - s[i] for i in range(len(s))])
    d = d[d > 0]
    if not len(d):
        raise ValueError("learn_plan: the bursts start together")
    if len(d) > 8192: # (a spread sample: the score is a mean)
        d = np.sort(d)[np.linspace(0, len(d) - 1, 8192).astype(int)]
    lo = max(np.min(d)/64, 1e-7)
    med = float(np.median(d))
    df = tol/(4*float(np.percentile(d, 99)))
    thr = np.cos(2*np.pi*tol)
    qs, score = [], []
    f, n, skip, failed = 1/med, 0, 0.0, 0 # skip: 1/q up to which a peak was looked at and failed
    while f < 1/lo and n < most and failed < peaks:
        fr = f + df*np.arange(chunk)
        fr = fr[fr < 1/lo]
        sc = np.cos(2*np.pi*np.outer(fr, d)).mean(axis=1)
        qs.append(1/fr)
        score.append(sc)
        n += len(fr)
        for i in np.nonzero(sc >= thr)[0]:
            if fr[i] <= skip:
                continue
            found = 1/fr[i]
            # Refine over the peak the first passing q is on the edge of (its half-width about tol*q/d over the typical
            # difference d), below it (the descent comes from above); then exact over the capture, and checked
            width = 2*tol*found*found/med
            fine = np.linspace(found + 0.25*width, found - width, 401)
            q = _stretch(s, float(fine[np.argmax(np.cos(2*np.pi*np.outer(1/fine, d)).sum(axis=1))]), tol)
            r = (s - s[0])/q
            if np.mean(np.abs(r - np.round(r)) <= tol) >= on_grid:
                return q, (np.concatenate(qs), np.concatenate(score))
            skip, failed = 1/(found - width), failed + 1 # (past this peak)
            if failed >= peaks:
                break
        f = max(fr[-1] + df, skip)
    raise ValueError("learn_plan: the bursts' starts sit on no grid (give `quantum`)")

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
