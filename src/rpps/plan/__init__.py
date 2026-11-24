"""Burst time plans: when, and on which carrier, each burst of a TDMA / MF-TDMA return link is, and what it is

A plan is what a network's controller tells its terminals (DVB-RCS2's SCT, FCT2, BCT and TBTP2, ETSI EN 301 545-2
clause 6.2.2; DVB-RCS's SCT/FCT/TCT/TBTP, EN 301 790): RPPS holds it in that model, whatever it came from.

    sequences    superframe sequences: superframe `count` starts at `start` (clock ticks) and each lasts `duration`;
                 a centre frequency; frames at time and frequency offsets from the superframe's, each of a frame type.
                 Frames are numbered from 0 by ascending lowest burst frequency, then start time, then frame type
                 (EN 301 545-2 7.5.1.1)
    frame_types  a grid of bandwidth-time units (BTUs: `btu` duration in ticks, bandwidth, symbol rate) `time_units` long
                 on each carrier (`grid`: the carriers' offsets from the frame's centre); timeslot sections (tx type,
                 repeat) fill the grid in time on the lowest carrier, then the next (7.5.1.3): a slot is as many BTUs as
                 its tx type's `btus`, a tx type 0 one BTU (a slot an assignment gives a type to)
    tx_types     burst formats: `btus`, `offset` (ticks from the slot's start to the burst's), and the `fields` a profile
                 needs to receive it (its preamble word, modulation, code, length...): what controls drive per burst
    assignments  per (sequence, superframe count): per frame, from slot `offset` on, (assignment id, tx type or 0) in
                 slot order; a tx type given to a single-BTU slot takes as many BTUs as it needs (the TBTP2's rule)

`Plan.windows(t0, t1)` lists the bursts' places in a span of clock time; `clock` is ticks a second (27e6: the NCR; 1:
seconds). Where a plan comes from:

  - over the air, from the tables a forward link carries: `rp.plan.builder("dvbrcs2/forward")` takes their decoded
    messages and gives a Plan, updated as new versions and each superframe's assignments come, and the network clock
    (rpps.plan.air: a builder is a definition too, mapping the tables onto this model)
  - a definition, `config/plan/<family>/<name>.json`, of the same model (a standard's fixed structure, a test, a plan
    `rp.analysis.spectrum.burst` learned from detected bursts): `rp.plan.load("demo/mftdma")`

Hopping: a sequence's `hop` ({"channels": [Hz, ...], "carrier": "(k*hop + f) % 37", "params": {"hop": 7}}, or
"algorithm": "ble_csa1") moves each frame's centre to a channel, superframe by superframe; a train (bursts at a fixed
spacing for a while, then a long pause) is a superframe whose frames start where the train's bursts do: a frame's
"repeat": n and "every": ticks make n frames that far apart.

A receiver places its bursts in the plan (rpps.sync.Burst `plan`), and a transmitter its bursts on it
(rpps.sync.PlanTx). The plan's time is the network's: where its clock is in a capture (the origin) is the receiver's to
find (from the forward link's clock reference, or from the bursts' preambles).

    {"base": "plan", "clock": 27e6,
     "tx_types": {"1": {"btus": 2, "offset": 270, "fields": {"word": "demo_uw1", "units": 1}}},
     "frame_types": {"1": {"duration": 2700000, "btu": {"duration": 108000, "bw": 600e3, "symbol_rate": 500e3},
                           "time_units": 25, "grid": [-1e6, 0, 1e6], "sections": [[1, 36]]}},
     "sequences": {"0": {"start": 0, "duration": 2700000, "centre": 0, "count": 0,
                         "frames": [{"type": 1, "start": 0, "offset": 0}]}}}
"""
from dataclasses import dataclass, field

from .. import _config
from ..errors import ConfigError

COUNT_MOD = 1 << 16 # superframe_count wraps


@dataclass
class TxType:
    id: int
    btus: int = 1
    offset: int = 0 # ticks from the slot's start to the burst's
    fields: dict = field(default_factory=dict)

@dataclass
class Slot:
    number: int
    carrier: int        # index into the frame type's grid, ascending frequency
    freq: float         # offset from the frame's centre [Hz]
    start: int          # BTUs from the frame's start
    btus: int
    tx_type: int        # 0: a single BTU an assignment gives a type to

@dataclass
class FrameType:
    id: int
    duration: int
    btu_duration: int
    btu_bw: float
    symbol_rate: float
    time_units: int
    grid: list
    sections: list      # [(tx type, repeat)]: repeat+1 slots of that type
    _slots: list = None

    def slots(self, tx_types):
        """The frame's timeslots, numbered from 0: in time on the lowest carrier, then the next (EN 301 545-2 7.5.1.4)"""
        if self._slots is not None:
            return self._slots
        grid = sorted(self.grid)
        out, c, t = [], 0, 0
        for tx, repeat in self.sections:
            n = 1 if not tx else tx_types[tx].btus if tx in tx_types else None
            if n is None:
                raise ConfigError(f"frame type {self.id}: tx type {tx} is not in the plan's tx types")
            for _ in range(int(repeat) + 1):
                if t + n > self.time_units:
                    c, t = c + 1, 0
                if c >= len(grid):
                    raise ConfigError(f"frame type {self.id}: its sections need more than its {len(grid)} carriers x "
                                      f"{self.time_units} BTUs")
                out.append(Slot(len(out), c, float(grid[c]), t, n, int(tx)))
                t += n
        self._slots = out
        return out

@dataclass
class Frame:
    type: int
    start: int          # ticks from the superframe's start
    offset: float       # Hz from the superframe's centre

@dataclass
class Sequence:
    id: int
    start: int          # ticks: the start of superframe `count`
    duration: int
    centre: float
    count: int
    frames: list        # Frame, in frame number order
    hop: object = None  # Hop: the frames' channel changes superframe by superframe (frequency hopping)

class Hop:
    """Frequency hopping: each frame's centre moved to one of `channels` (Hz from the sequence's centre), the channel
    an expression of the superframe's index `k` (superframes since the sequence's `count`) and the frame number `f`
    (`"carrier": "(k*hop) % 37"`, the hop's params in scope), or a registered algorithm (`"algorithm": "ble_csa1"`)"""
    def __init__(self, channels, carrier=None, algorithm=None, params=None):
        self.channels = [float(c) for c in channels]
        self.carrier, self.algorithm, self.params = carrier, algorithm, dict(params or {})
        if not self.channels or (carrier is None) == (algorithm is None):
            raise ConfigError("plan hop: channels, and a carrier expression or an algorithm (one of the two)")
        if algorithm is not None and algorithm not in HOPS:
            raise ConfigError(f"plan hop: no algorithm {algorithm!r} (have {sorted(HOPS)})")

    def index(self, k, f):
        if self.algorithm is not None:
            return int(HOPS[self.algorithm](k, f, len(self.channels), **self.params)) % len(self.channels)
        return int(_config.evaluate(str(self.carrier), {**self.params, "k": int(k), "f": int(f)}, "plan hop")) % len(self.channels)

    def offset(self, k, f):
        return self.channels[self.index(k, f)]

HOPS = {}

def hop_algorithm(name):
    """Register a hopping algorithm: fn(k, f, n_channels, **params) -> channel index"""
    def deco(fn):
        HOPS[name] = fn
        return fn
    return deco

@hop_algorithm("ble_csa1")
def _ble_csa1(k, f, n, hop=5, used=None):
    """Bluetooth LE channel selection #1 (Core Vol 6 Part B 4.5.8.2): unmapped = (k+1)*hop mod 37 for connection event k
    (the last unmapped channel starting at 0); a channel the map does not use is remapped to used[unmapped mod #used]"""
    unmapped = ((int(k) + 1)*int(hop)) % 37
    used = list(range(37)) if used is None else sorted(int(u) for u in used)
    return unmapped if unmapped in used else used[unmapped % len(used)]

@dataclass
class Window:
    """Where one burst may be: its superframe sequence and count, frame number, slot number, carrier [Hz, absolute: the
    sequence's centre + frame + carrier offsets], its slot's start and length and the burst's start [ticks], symbol
    rate, tx type and the assignment (id, None when the plan has none)"""
    sequence: int
    superframe: int
    frame: int
    slot: int
    freq: float
    start: int
    length: int
    burst: int
    symbol_rate: float
    bw: float
    tx_type: int
    assignment: object = None

    def fields(self, plan):
        """The burst's fields: where it is, and its tx type's own (what a profile's controls drive)"""
        tx = plan.tx_types.get(self.tx_type)
        return {"sequence": self.sequence, "superframe": self.superframe, "frame": self.frame, "slot": self.slot,
                "freq": self.freq, "tx_type": self.tx_type, "assignment": self.assignment, **(tx.fields if tx else {})}


class Plan:
    """A burst time plan (see the module). clock: ticks a second"""
    def __init__(self, clock=1.0, tx_types=None, frame_types=None, sequences=None, assignments=None, name="plan"):
        self.clock, self.name = float(clock), name
        self.tx_types = dict(tx_types or {})
        self.frame_types = dict(frame_types or {})
        self.sequences = dict(sequences or {})
        self.assignments = dict(assignments or {}) # {(sequence, count): {frame: (offset, [(id, tx_type)])}}

    def __repr__(self):
        return (f"<Plan {self.name}: {len(self.sequences)} superframe sequence(s), {len(self.frame_types)} frame type(s), "
                f"{len(self.tx_types)} tx type(s)>")

    def seconds(self, ticks):
        return ticks/self.clock

    def ticks(self, seconds):
        return seconds*self.clock

    def carriers(self, sequences=None):
        """Every carrier [Hz] a burst of the plan can be on (a hopping sequence's every channel), with its bandwidth and
        symbol rate: {freq: (bw, rate)}"""
        out = {}
        for sid, s in self.sequences.items():
            if sequences is not None and sid not in sequences:
                continue
            hops = [0.0] if s.hop is None else s.hop.channels
            for f in s.frames:
                ft = self.frame_types[f.type]
                for h in hops:
                    for g in ft.grid:
                        out.setdefault(s.centre + f.offset + h + g, (ft.btu_bw, ft.symbol_rate))
        return dict(sorted(out.items()))

    def frames(self, seq):
        """A sequence's frames in frame-number order: by lowest burst frequency, start, frame type (7.5.1.1)"""
        s = self.sequences[seq]
        def low(f):
            ft = self.frame_types[f.type]
            return f.offset + min(ft.grid)
        return sorted(s.frames, key=lambda f: (low(f), f.start, f.type))

    def assign(self, sequence, count, frame, assignments, offset=0):
        """Assignments of one superframe's frame: [(assignment id, tx type or 0)] in slot order from slot `offset`"""
        self.assignments.setdefault((int(sequence), int(count) % COUNT_MOD), {})[int(frame)] = (int(offset), list(assignments))

    def windows(self, t0, t1, sequences=None, assigned=False):
        """The bursts' places that start in [t0, t1) ticks: every slot of every frame of every superframe there (the
        assignments, where the plan has them, merging single-BTU slots into a tx type's and naming whose slot it is).
        assigned: only the slots an assignment gives"""
        out = []
        for sid, s in self.sequences.items():
            if sequences is not None and sid not in sequences:
                continue
            if s.duration <= 0:
                continue
            k0 = int((t0 - s.start)//s.duration) - 1
            k1 = int((t1 - s.start)//s.duration) + 1
            frames = self.frames(sid)
            for k in range(k0, k1 + 1):
                sf0 = s.start + k*s.duration
                count = (s.count + k) % COUNT_MOD
                given = self.assignments.get((sid, count), {})
                for fnum, f in enumerate(frames):
                    ft = self.frame_types[f.type]
                    f0 = sf0 + f.start
                    if s.hop is not None: # (this superframe's channel)
                        f = Frame(f.type, f.start, f.offset + s.hop.offset(k, fnum))
                    for w in self._slots(sid, count, fnum, f, ft, f0, s.centre, given.get(fnum), assigned):
                        if t0 <= w.start < t1:
                            out.append(w)
        out.sort(key=lambda w: (w.start, w.freq))
        return out

    def _slots(self, sid, count, fnum, f, ft, f0, centre, given, assigned):
        slots = ft.slots(self.tx_types)
        def window(sl, btus, tx, who):
            txt = self.tx_types.get(tx)
            start = f0 + sl.start*ft.btu_duration
            return Window(sid, count, fnum, sl.number, centre + f.offset + sl.freq, start, btus*ft.btu_duration,
                          start + (txt.offset if txt else 0), ft.symbol_rate, ft.btu_bw, tx, who)
        if given is None:
            return [] if assigned else [window(sl, sl.btus, sl.tx_type, None) for sl in slots]
        offset, items = given
        out, i = [], offset
        for who, tx in items:
            if i >= len(slots):
                break
            sl = slots[i]
            tx = tx or sl.tx_type
            n = self.tx_types[tx].btus if (tx and not sl.tx_type) else sl.btus # (a typed single-BTU slot takes its type's BTUs)
            out.append(window(sl, n, tx, who))
            i += n if not sl.tx_type else 1
        return out

    # ---- as a definition

    @staticmethod
    def load(name, obj):
        """From a definition (see the module)"""
        def num(v, what):
            try:
                return float(v)
            except (TypeError, ValueError):
                raise ConfigError(f"plan {name}: {what} is a number, not {v!r}") from None
        clock = obj.get("clock", 1.0)
        tx = {}
        for k, v in (obj.get("tx_types") or {}).items():
            tx[int(k)] = TxType(int(k), int(v.get("btus", 1)), int(round(num(v.get("offset", 0), f"tx type {k} offset"))),
                                dict(v.get("fields", {})))
        fts = {}
        for k, v in (obj.get("frame_types") or {}).items():
            btu = v.get("btu") or {}
            for need in ("duration", "symbol_rate"):
                if need not in btu:
                    raise ConfigError(f"plan {name}: frame type {k} needs btu.{need}")
            fts[int(k)] = FrameType(int(k), int(round(num(v.get("duration", 0), "frame duration"))),
                                    int(round(num(btu["duration"], "btu duration"))), num(btu.get("bw", btu["symbol_rate"]), "btu bw"),
                                    num(btu["symbol_rate"], "btu symbol_rate"), int(v.get("time_units", 1)),
                                    [num(g, "grid offset") for g in (v.get("grid") or [0.0])],
                                    [(int(t), int(r)) for t, r in (v.get("sections") or [])])
        seqs = {}
        for k, v in (obj.get("sequences") or {}).items():
            frames = [Frame(int(f["type"]), int(round(num(f.get("start", 0), "frame start") + j*num(f.get("every", 0), "frame every"))),
                            num(f.get("offset", 0), "frame offset"))
                      for f in v.get("frames", []) for j in range(int(f.get("repeat", 1)))] # (repeat, every: a train of frames)
            for f in frames:
                if f.type not in fts:
                    raise ConfigError(f"plan {name}: sequence {k} names frame type {f.type}, not one of {sorted(fts)}")
            hop = None
            if v.get("hop"):
                h = v["hop"]
                hop = Hop(h["channels"], h.get("carrier"), h.get("algorithm"), h.get("params"))
            seqs[int(k)] = Sequence(int(k), int(round(num(v.get("start", 0), "start"))), int(round(num(v["duration"], "duration"))),
                                    num(v.get("centre", 0), "centre"), int(v.get("count", 0)), frames, hop)
        p = Plan(num(clock, "clock"), tx, fts, seqs, name=name)
        for a in obj.get("assignments") or []:
            p.assign(a["sequence"], a["superframe"], a["frame"], [tuple(x) for x in a["slots"]], a.get("offset", 0))
        for ft in fts.values():
            ft.slots(tx) # (checked now)
        return p

    def definition(self, comment=""):
        """The plan as a definition (what `load` reads): to save one learned or built"""
        return {"base": "plan", "comment": comment, "clock": self.clock,
                "tx_types": {str(t.id): {"btus": t.btus, "offset": t.offset, "fields": dict(t.fields)} for t in self.tx_types.values()},
                "frame_types": {str(f.id): {"duration": f.duration, "btu": {"duration": f.btu_duration, "bw": f.btu_bw,
                                                                           "symbol_rate": f.symbol_rate},
                                            "time_units": f.time_units, "grid": list(f.grid),
                                            "sections": [list(s) for s in f.sections]} for f in self.frame_types.values()},
                "sequences": {str(s.id): {"start": s.start, "duration": s.duration, "centre": s.centre, "count": s.count,
                                          "frames": [{"type": f.type, "start": f.start, "offset": f.offset} for f in s.frames],
                                          **({"hop": {"channels": list(s.hop.channels), "carrier": s.hop.carrier,
                                                      "algorithm": s.hop.algorithm, "params": dict(s.hop.params)}}
                                             if s.hop is not None else {})}
                              for s in self.sequences.values()}}


from .air import Builder # (after Plan: a builder makes one)

DOMAIN = _config.register_domain("plan", {"plan": Plan.load, "builder": Builder.load})

def load(name, **params):
    """A plan definition by name: `rp.plan.load("demo/mftdma")`"""
    p = _config.load("plan", name, params=params)
    if not isinstance(p, Plan):
        raise ConfigError(f"plan {name} is a plan builder: rp.plan.builder({name!r})")
    return p

def init(name, obj):
    """A plan from a definition object"""
    return _config.load_obj("plan", name, obj)

def ls():
    """The plans there are, and the builders: {family: [names]}"""
    return _config.ls("plan")

def builder(name, **params):
    """A new builder of plans from a forward link's tables and clock references (rpps.plan.air):
    `rp.plan.builder("dvbrcs2/forward")`"""
    b = _config.load("plan", name, params=params)
    if not isinstance(b, Builder):
        raise ConfigError(f"plan {name} is a plan, not a builder: rp.plan.load({name!r})")
    return b
