"""What each bit of a set of frames (or bursts) is: constant, a counter, an id, a length, a few values, or random

    m = rp.analysis.framing.fields.map(frames, groups=terminal)   # frames: bit rows as sent, aligned at their start
    print(m)          # one line a field: "bits 0..15 constant 0x1acf", "bits 32..43 counter +1 a frame (msb)", ...
    m.guess.save("mine")   # config/header/link/guess/mine.json: the fields, named by kind and place

Frames are read up to the shortest (or `length`). Each field is a run of columns, found in this order, each test against
what chance gives over that many frames:

  - constant: a column one value in `hold` of the frames (a bit error or two allowed); runs of them, with their value
  - id: with `groups`, a run constant within every group and differing between groups (columns constant everywhere
    inside it included; found before counters: within
    a group an id and the counter after it step together). A counter's top bits that never change within a group over
    the record read as id too: only a carry through them tells them apart
  - counter: a value stepping by the same amount (mod 2^width) from frame to frame, in either bit order. Its lowest
    bit that changes flips every frame (step odd times 2^k); the field is grown from it both ways while the step holds
    in `hold` of the frames and each new bit changes at all (the counter's top bits never reached in the record read as
    constant). With `groups` (a terminal per frame, e.g. from rpps.analysis.spectrum.burst), steps within each group:
    frames in record order, each group its own count
  - length: with `lengths` (each frame's own, in bits or symbols), a run whose value is a*length + b for one a, b
  - enum: a run taking at most `max_values` values over the frames (a type, a flag set), grown while it stays few
  - random: the rest, each run's bias (mean of its bits) given; "varying" for a lone column between fields
"""
import numpy as np

from ..result import Guess

class Field:
    """start (bit), width, kind (constant, counter, id, length, enum, random, varying), detail (dict: value, step,
    order, values {value: count}, scale, offset, bias)"""
    __slots__ = ("start", "width", "kind", "detail")
    def __init__(self, start, width, kind, **detail):
        self.start, self.width, self.kind, self.detail = int(start), int(width), kind, detail

    @property
    def stop(self):
        return self.start + self.width

    def __repr__(self):
        d = self.detail
        what = {"constant": lambda: f"constant {d['value']:#x}" if self.width > 1 else f"constant {d['value']}",
                "counter": lambda: f"counter {d['step']:+d} a frame ({d['order']} first)",
                "id": lambda: f"id, {d['ids']} values across {d['groups']} groups",
                "length": lambda: f"length: {d['scale']:g} x length {d['offset']:+g}",
                "enum": lambda: f"{len(d['values'])} values {dict(sorted(d['values'].items(), key=lambda kv: -kv[1])[:4])}",
                "random": lambda: f"random (bias {d['bias']:.2f})",
                "varying": lambda: "varying"}[self.kind]()
        return f"bits {self.start}..{self.stop - 1}: {what}"

class FieldMap:
    """`map`: fields (covering bits 0 .. length - 1 in order), length, frames"""
    __slots__ = ("fields", "length", "frames")
    def __init__(self, fields, length, frames):
        self.fields, self.length, self.frames = fields, length, frames

    def of(self, kind):
        return [f for f in self.fields if f.kind == kind]

    @property
    def guess(self):
        """The fields as a link header definition (config/header/link/guess/<name>.json when saved): each named by its
        kind and first bit, an enum's values listed, the rest of the frame after them"""
        fs = []
        for f in self.fields:
            e = {"name": f"{f.kind}_{f.start}", "bits": f.width}
            if f.kind == "counter" and f.detail["order"] == "lsb":
                e["order"] = "lsb"
            if f.kind == "enum":
                e["enum"] = {str(v): f"v{v}" for v in sorted(f.detail["values"])}
            fs.append(e)
        return Guess("header", {"base": "header", "type": "link", "order": "msb", "fields": fs},
                     {"frames": self.frames, "length": self.length})

    def __str__(self):
        return f"{self.frames} frames, {self.length} bits\n" + "\n".join(f"  {f}" for f in self.fields)

    __repr__ = __str__

def _value(cols, order="msb"):
    """Each row's value of the bit columns cols (rows x width), int64"""
    w = cols.shape[1]
    pw = 1 << (np.arange(w - 1, -1, -1) if order == "msb" else np.arange(w))
    return cols.astype(np.int64) @ pw

def _steps(v, w, groups):
    """Frame-to-frame differences of values v (mod 2^w), within each group when given"""
    if groups is None:
        return np.diff(v) % (1 << w)
    out = []
    for g in np.unique(groups):
        vg = v[groups == g]
        if len(vg) > 1:
            out.append(np.diff(vg) % (1 << w))
    return np.concatenate(out) if out else np.zeros(0, np.int64)

def _mode_share(d):
    if len(d) == 0:
        return 0, 0.0
    vals, cnt = np.unique(d, return_counts=True)
    i = int(np.argmax(cnt))
    return int(vals[i]), float(cnt[i]/len(d))

def _counters(F, const, groups, hold, max_width):
    """[Field] counters: grown from columns that flip every frame (within groups), over columns not in `const` (the
    constant ones, and ids)"""
    N, L = F.shape
    found, used = [], np.zeros(L, bool)
    flips = []
    for j in range(L):
        if const[j]:
            continue
        d = _steps(F[:, j].astype(np.int64), 1, groups)
        if len(d) >= 3 and np.mean(d == 1) >= hold:
            flips.append(j)
    for j in flips:
        if used[j]:
            continue
        best = None
        for order in ("msb", "lsb"):
            # msb first: the counter's low bit is its last column, the field grows to the left; lsb first: to the right
            lo = hi = j
            step = None
            while True:
                grew = False
                for nlo, nhi in ((lo - 1, hi), (lo, hi + 1)) if order == "msb" else ((lo, hi + 1), (lo - 1, hi)):
                    if nlo < 0 or nhi >= L or nhi - nlo + 1 > max_width or used[nlo:nhi + 1].any():
                        continue
                    new = nlo if nlo < lo else nhi
                    if const[new]:
                        continue
                    # the new bit must sit above the low bit in this order
                    if (order == "msb" and new > j) or (order == "lsb" and new < j):
                        continue
                    w = nhi - nlo + 1
                    v = _value(F[:, nlo:nhi + 1], order)
                    s, share = _mode_share(_steps(v, w, groups))
                    if share >= hold and s != 0:
                        lo, hi, step, grew = nlo, nhi, s, True
                        break
                if not grew:
                    break
            if step is None: # a lone flipping bit: a 1-bit counter is just a toggle
                continue
            w = hi - lo + 1
            if best is None or w > best[1] - best[0] + 1:
                signed = step - (1 << w) if step >= 1 << (w - 1) else step
                best = (lo, hi, signed, order)
        if best is not None and best[1] - best[0] + 1 >= 3:
            lo, hi, s, order = best
            used[lo:hi + 1] = True
            found.append(Field(lo, hi - lo + 1, "counter", step=int(s), order=order))
    return found, used

def _inside(fields, L):
    m = np.zeros(L, bool)
    for f in fields:
        m[f.start:f.stop] = True
    return m

def _runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

def map(frames, groups=None, lengths=None, length=None, hold=0.95, max_width=32, max_values=8):
    """FieldMap of the frames (bit rows, aligned at their first bit). groups: a label a frame (a terminal), for counters
    within each and ids; lengths: each frame's length, for length fields; length: bits to map (default: the shortest
    frame's)"""
    rows = [np.asarray(f).astype(bool).reshape(-1) for f in frames]
    L = min(len(r) for r in rows) if length is None else int(length)
    rows = [r for r in rows if len(r) >= L]
    F = np.array([r[:L] for r in rows], dtype=np.uint8)
    N = len(F)
    if N < 4:
        raise ValueError(f"{N} frames: need 4 at least to tell fields from chance")
    groups = None if groups is None else np.asarray(groups)
    p1 = F.mean(axis=0)
    const = (p1 >= hold) | (p1 <= 1 - hold)
    fields, taken = [], const.copy()
    # ids: constant within every group of 2 frames or more, not across them
    if groups is not None:
        within = np.ones(L, bool)
        G = [F[groups == g] for g in np.unique(groups)]
        many = [g for g in G if len(g) >= 2]
        for g in many:
            q = g.mean(axis=0)
            within &= (q >= hold) | (q <= 1 - hold)
        if len(many) >= 2: # runs constant within the groups, through columns constant everywhere, ends that vary
            for a, b in _runs(within):
                vary = np.flatnonzero(~const[a:b])
                if len(vary) == 0:
                    continue
                a, b = a + int(vary[0]), a + int(vary[-1]) + 1
                vals = {int(v) for v in _value(F[:, a:b])}
                fields.append(Field(a, b - a, "id", ids=len(vals), groups=len(G)))
                taken[a:b] = True
    counters, used = _counters(F, taken, groups, hold, max_width)
    fields += counters
    taken |= used
    # length fields
    if lengths is not None:
        ln = np.asarray([l for l, r in zip(lengths, frames) if len(np.asarray(r).reshape(-1)) >= L], dtype=np.float64)
        if len(np.unique(ln)) >= 3:
            for a, b in _runs(~taken):
                best = None
                for w in range(2, min(16, b - a) + 1):
                    for o in range(a, b - w + 1):
                        v = _value(F[:, o:o + w]).astype(np.float64)
                        if len(np.unique(v)) < 3:
                            continue
                        A = np.vstack((ln, np.ones_like(ln))).T
                        (k, c), *_ = np.linalg.lstsq(A, v, rcond=None)
                        if k != 0 and np.mean(np.abs(A @ (k, c) - v) < 0.5) >= hold and (best is None or w > best[1]):
                            best = (o, w, k, c)
                if best is not None:
                    o, w, k, c = best
                    fields.append(Field(o, w, "length", scale=float(np.round(k, 6)), offset=float(np.round(c, 3))))
                    taken[o:o + w] = True
    # enums: runs of few values, grown while they stay few
    for a, b in _runs(~taken):
        o = a
        while o < b:
            e = o + 1
            while e < b and len(np.unique(_value(F[:, o:e + 1]))) <= max_values and e + 1 - o <= 16:
                e += 1
            vals, cnt = np.unique(_value(F[:, o:e]), return_counts=True)
            if e - o >= 2 and len(vals) <= max_values and len(vals) < 2**(e - o) and N >= 4*len(vals):
                fields.append(Field(o, e - o, "enum", values={int(v): int(c) for v, c in zip(vals, cnt)}))
                taken[o:e] = True
            o = e
    # constants, random and what is left
    for a, b in _runs(const & ~used & ~_inside(fields, L)):
        fields.append(Field(a, b - a, "constant", value=int(_value((p1[None, a:b] > 0.5).astype(np.uint8))[0])))
    for a, b in _runs(~taken):
        if b - a == 1:
            fields.append(Field(a, 1, "varying"))
        else:
            fields.append(Field(a, b - a, "random", bias=float(F[:, a:b].mean())))
    return FieldMap(sorted(fields, key=lambda f: f.start), L, N)
