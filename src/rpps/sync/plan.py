"""Bursts on many carriers, placed in a burst time plan (rpps.plan): the return link of an MF-TDMA network

    for b in src(1 << 16) + rp.sync.Bursts(carriers=[{"freq": -1e6, "width": 600e3}, {"freq": 0, "width": 600e3}]):
        print(b.ctx["burst"])          # {"carrier": 0, "freq": -1e6, "start": 0.0123}

    bursts = rp.sync.Bursts(plan=rp.plan.load("demo/mftdma"))   # the plan's carriers; each burst labelled with its place

Detection comes first: each carrier is brought to baseband (rpps.signal.channelize) and cut into the bursts that rise
out of its own noise floor (rpps.sync.Burst), at absolute positions, so a capture with no plan still gives its bursts,
and a plan learned from them (rpps.analysis.burst) or told by a forward link (rpps.plan) places them:

  - `plan`: each burst is labelled with its window (superframe, frame, slot, tx type, assignment) and the fields of its
    tx type (what a profile's controls drive per burst). A burst no window explains is given unlabelled, and counted.
  - the plan's time in the capture, `origin` (plan clock ticks at the capture's first sample): given (from a forward
    link's clock reference, rpps.plan), or acquired. Acquiring is block estimation: each detected burst votes for the
    origins that would put it on a slot of its carrier; the most voted, once `lock` bursts agree on it (to `tol`), is
    taken, and then tracked burst by burst (rpps.sync.track, its rate the clock offset between the network and the
    capture). The precise start of each burst comes back from the preamble its profile found (`feedback`, which a
    profile's burst runner calls): energy edges only seed it, as they smear at low SNR.
  - `gate`: "energy" (default) or "plan": once the origin is locked, cut every window the plan gives (assigned ones, when
    the plan has assignments) whether its energy stands out or not, `margin` seconds either side.

Each burst is a Data of its own stream (the steps after it start over), at the carrier's rate, its first sample at its
position in the carrier's stream; `ctx["burst"]` holds its fields: carrier (index), freq (Hz from the capture's centre),
start (seconds from the capture's first sample), and with a plan its window's fields. `sps` (or a carrier's own) with
its `rate` brings each carrier to exactly that many samples a symbol (decimated by a whole factor, the rest resampled:
a capture at any rate), so one profile serves carriers of several rates. Estimates in est["bursts"]:
per carrier the floor and bursts, the origin, its rate, locked, labelled and unlabelled counts.

The rest of the burst code: rpps.sync.burst (one carrier's bursts by energy), rpps.profile.burst (a profile's burst step),
rpps.analysis.burst (bursts demodulated blind; plans learned and checked: its docstring maps it all).
"""
from types import SimpleNamespace

import numpy as np

from ..base.rpps import Pipe
from ..base.types import Category, Rep
from ..base.meta import Stream
from ..base.data import Data
from . import track
from .burst import Burst, BurstTx

class Bursts(Pipe):
    CATEGORY = Category.SYNC
    CONTROLS = {"plan": "live", "origin": "live"} # A forward link's tables and clock reference say them

    def __init__(self, carriers=None, plan=None, origin=None, gate="energy", tol=None, lock=8, margin=None,
                 oversample=3.0, bw=0.25, sps=None, pad=None, **burst):
        if carriers is None and plan is None:
            raise ValueError("Bursts: the carriers, or a plan that has them")
        if gate not in ("energy", "plan"):
            raise ValueError(f"Bursts: gate is energy or plan, not {gate!r}")
        self.carriers = None if carriers is None else [dict(c) for c in carriers]
        self.plan, self.origin, self.gate = plan, origin, gate
        self.tol, self.lock, self.margin, self.oversample, self.bw = tol, int(lock), margin, float(oversample), float(bw)
        self.sps = None if sps is None else int(sps)
        self.pad = pad # Seconds kept either side of an energy burst (its edges sit under its pulse's tails); None: 16
        # symbols where the rate is known, else 4 windows
        self.burst = dict(burst)
        Burst(**self.burst) # (checked now)

    def __str__(self):
        return f"Bursts({len(self.carriers) if self.carriers else 'plan'} carriers{', plan' if self.plan else ''})"

    def takes(self, encode=True):
        return {Rep.SAMPLES}

    def control(self, plan=None, origin=None):
        """A plan, or its time in the capture, from what a forward link says (rpps.plan): the origin seeds the tracker,
        which the bursts' preambles still refine (the forward link's clock does not know the path to this receiver)"""
        if plan is not None:
            self.plan = plan
        if origin is not None:
            self.origin = float(origin)
            self._reseed = True

    _reseed = False
    _given = False # (no longer frozen: a given origin is where tracking starts)

    # ---- the carriers, and each one's chain

    def _carriers(self, cf):
        """[{freq, width, rate}] from the capture's centre: given, or the plan's (each carrier its windows' frequency)"""
        if self.carriers is not None:
            return self.carriers
        return [{"freq": round(f - (cf or 0.0), 3), "width": bw, "rate": r} for f, (bw, r) in self.plan.carriers().items()]

    def new_state(self, meta):
        from ..signal.channelize import Channelize, channelize
        from ..signal.filters.filters import low_pass_kaiser
        chans = []
        carriers = self._carriers(meta.cf)
        for c in carriers:
            width = float(c.get("width") or c.get("rate") or meta.Fs)
            over, sps, rate = self.oversample, c.get("sps", self.sps), c.get("rate")
            ratio = None # What is left to resample after decimating
            if sps and rate: # Decimated to sps samples a symbol: by a whole factor, the rest by resampling
                target = float(sps)*float(rate)
                D = meta.Fs/target
                if abs(D - round(D)) < 1e-9 and round(D) >= 1:
                    D = int(round(D))
                else: # (a capture at 100 MS/s / 2^k: no whole factor; the largest that stays above the target)
                    D = max(1, int(np.floor(D)))
                    ratio = target*D/meta.Fs
                over = meta.Fs/((D + 0.5)*width)
            from ..base import Pipeline
            chan = Channelize(c["freq"], width, over)
            Fs_c = channelize(c["freq"], width, meta.Fs, over)[1]
            near = min((abs(float(o["freq"]) - float(c["freq"])) for o in carriers if o is not c), default=None)
            stop = min(0.5*Fs_c, near - 0.5*width) if near else 0.5*Fs_c
            if stop > 0.6*width: # The neighbours out: channelizing only keeps them from aliasing onto the channel
                chan = Pipeline(chan, low_pass_kaiser(0.5*(0.5*width + stop), stop - 0.5*width, 50, Fs_c))
            if ratio is not None and abs(ratio - 1) > 1e-12:
                from ..signal.sample.fractional import FractionalPolyphase
                chan = Pipeline(chan, FractionalPolyphase(ratio))
            chans.append(SimpleNamespace(freq=float(c["freq"]), width=width, chan=chan,
                                         cut=Burst(**self.burst), stream=Stream(meta.Fs), Fs=None, delay=0.0, held=None,
                                         held_pos=0, sps=float(sps) if sps and rate else None, rate=rate, pending=[]))
        return SimpleNamespace(chans=chans, origin=None, tr=track.new(), votes=[], labelled=0, unlabelled=0, cut=set(),
                               Fs=meta.Fs, cf=meta.cf or 0.0, waiting=[], fb=[])

    def __radd__(self, data):
        from ..base.data import _Data, expect
        if not isinstance(data, _Data):
            return NotImplemented
        expect(self, data, self.takes(), True)
        return self.split(data)

    def split(self, data):
        """The bursts that end in this block, on every carrier, in order of their starts (a list of Data)"""
        st = data.meta.state_of(self)
        x = np.asarray(data.data)
        out = []
        for i, ch in enumerate(st.chans):
            inner = Data(x, Fs=data.meta.Fs, sps=data.meta.sps, cf=data.meta.cf, stream=ch.stream)
            inner.flush = data.flush
            y = inner + ch.chan
            ch.Fs, ch.delay = y.meta.Fs, y.meta.delay
            self._hold(ch, y)
            pad = self._pad(ch)
            warm = int(np.ceil(2*ch.delay*ch.Fs)) + 1 - (y.meta.pos - ch.held0) # The filters' start: not cut (the
            if warm > 0: # cutter would take those near-silent samples for its noise floor); it starts after them
                k = min(warm, len(y))
                m = y.meta.fork()
                m.pos = y.meta.pos + k
                y = type(y)(np.asarray(y.data)[k:], y.CAT, y.DT, dict(y.ctx), m, y.REP)
            for b in y + ch.cut:
                for a, n in self._split(st, ch, b.meta.pos, len(b)):
                    start = a/ch.Fs - ch.delay # (seconds from the capture's first sample)
                    f = self._label(st, i, ch, {"carrier": i, "freq": ch.freq, "start": start}, n/ch.Fs)
                    if not f.get("_dup"):
                        ch.pending.append((a - pad, a + n + pad, f))
            cs = y.meta.state_of(ch.cut) # (the energy cutter's: a burst still open holds the history from its start)
            ch.open = None
            if cs is not None and getattr(cs, "pos", None) is not None and not len(cs.hist): # still settling: all it holds
                ch.open = cs.pos
            elif cs is not None and getattr(cs, "on", False) and cs.start is not None:
                ch.open = cs.start
            if self.gate == "plan" and self._locked(st):
                ch.pending += self._windows(st, i, ch, data.flush)
        if st.waiting and (self._locked(st) or data.flush): # The bursts before the origin was found: placed now
            for i, f, dur in st.waiting:
                f.pop("_wait", None)
                if self._locked(st):
                    f.update(self._label(st, i, st.chans[i], {k: f[k] for k in ("carrier", "freq", "start")}, dur))
                else:
                    st.unlabelled += 1
            st.waiting = []
        for ch in st.chans:
            out += self._emit(ch, data.flush)
        out.sort(key=lambda o: o[0])
        bursts = []
        for start, f, b in out:
            if f.get("_dup"):
                continue
            b.ctx["burst"] = {k: v for k, v in f.items() if not k.startswith("_")}
            b.ctx["burst_window"] = f.get("_window")
            bursts.append(b)
        data.meta.stream.est["bursts"] = {"carriers": [{"freq": ch.freq} for ch in st.chans], "origin": self._origin(st),
                                          "rate": st.tr.rate, "locked": self._locked(st), "labelled": st.labelled,
                                          "unlabelled": st.unlabelled}
        return bursts

    def _pad(self, ch):
        if self.pad is not None:
            return int(round(self.pad*ch.Fs))
        if ch.rate:
            return int(round(16*ch.Fs/ch.rate))
        return 4*int(self.burst.get("win", 100))

    def _hold(self, ch, y):
        x = np.asarray(y.data)
        if ch.held is None:
            ch.held, ch.held_pos, ch.held0 = x, y.meta.pos, y.meta.pos
        else:
            ch.held = np.concatenate((ch.held, x))

    def _emit(self, ch, flush):
        """The pending bursts of a carrier whose samples are all in (the stream's end: all), cut from its history"""
        end = ch.held_pos + len(ch.held)
        out, keep = [], []
        for a, b, f in sorted(ch.pending, key=lambda q: q[0]):
            if (b > end or f.get("_wait")) and not flush:
                keep.append((a, b, f))
                continue
            ia, ib = max(a - ch.held_pos, 0), min(b - ch.held_pos, len(ch.held))
            if ib <= ia:
                continue
            stream = Stream(ch.Fs)
            stream.pos = ch.held_pos + ia
            out.append((f["start"], f, Data(np.ascontiguousarray(ch.held[ia:ib]), flush=True, Fs=ch.Fs, sps=ch.sps,
                                            stream=stream, delay=ch.delay)))
        ch.pending = keep
        span = self._pad(ch) + (int(round(self._max_len(self.plan)*ch.Fs)) if self.plan is not None else 0)
        cut = min([a for a, _, _ in keep] + [end - span - self._pad(ch) - int(self.burst.get("win", 100))]
                  + ([ch.open - self._pad(ch)] if getattr(ch, "open", None) is not None else []))
        cut = max(cut - ch.held_pos, 0)
        if cut:
            ch.held, ch.held_pos = ch.held[cut:], ch.held_pos + cut
        return out

    # ---- placing a burst in the plan

    def _locked(self, st):
        return st.tr.started or self.origin is not None

    def _origin(self, st, t=0.0):
        """The plan's clock at capture time t [s] minus clock*t: where the plan's ticks start (tracked from a given
        origin or from the one the bursts voted for)"""
        if (self.origin is not None and not st.tr.started) or self._reseed:
            st.tr, st.origin, self._reseed = track.new(), float(self.origin), False
            track.update(st.tr, float(self.origin), 0.0)
        if not st.tr.started:
            return None
        return track.at(st.tr, t)

    def _tol(self, st, w=None):
        """How far [s] a burst's start may be from its window's"""
        if self.tol is not None:
            return float(self.tol)
        if w is not None and st is not None:
            return 0.5*w.length/self.plan.clock
        return 0.5*min(ft.btu_duration for ft in self.plan.frame_types.values())/self.plan.clock

    def _split(self, st, ch, a, n):
        """An energy burst as the bursts it holds: bursts in adjacent slots whose guard the noise filled are one to the
        detector; with the plan's origin known, a burst longer than its first window is cut at the window's end, and so on"""
        if self.plan is None or self._origin(st, a/ch.Fs - ch.delay) is None:
            return [(a, n)]
        p, out = self.plan, []
        while n > 0:
            t = a/ch.Fs - ch.delay
            w, d = self._near(st, None, ch, t, self._origin(st, t))
            end = None
            if w is not None and d <= self._tol(st, w):
                end = int(round(((w.start + w.length - self._origin(st, t))/p.clock + ch.delay)*ch.Fs))
            if end is None or end - a >= n or end <= a + int(0.25*n): # (fits, or no window to cut at)
                out.append((a, n))
                break
            out.append((a, end - a))
            n -= end - a
            a = end
        return out

    def _fits(self, w, dur):
        """Whether a burst `dur` seconds long (None: unknown) fits window w: it fills its slot to within a BTU (a slot is
        the fewest BTUs the burst needs: EN 301 545-2 6.4.6.1, tx_block_size)"""
        if dur is None:
            return True
        btu = w.length/max(self.plan.tx_types[w.tx_type].btus if w.tx_type in self.plan.tx_types else 1, 1)
        return w.length - btu - self._tol(None)*self.plan.clock <= dur*self.plan.clock <= w.length + self._tol(None)*self.plan.clock

    def _near(self, st, i, ch, t, origin, dur=None):
        """The window of carrier i whose burst start is nearest capture time t (a burst `dur` long fits it), and the
        distance [s]"""
        p = self.plan
        tick = origin + t*p.clock
        span = max((ft.duration for ft in p.frame_types.values()), default=0) + 2*self._tol(st)*p.clock
        best, dist = None, None
        for w in p.windows(tick - span, tick + span):
            if abs(w.freq - st.cf - ch.freq) > 0.5*max(w.bw, 1.0) or not self._fits(w, dur):
                continue
            d = abs(w.burst - tick)/p.clock
            if dist is None or d < dist:
                best, dist = w, d
        return best, dist

    def _label(self, st, i, ch, f, dur=None):
        if self.plan is None:
            return f
        origin = self._origin(st, f["start"])
        if origin is None:
            self._vote(st, i, ch, f["start"], dur)
            origin = self._origin(st, f["start"])
            if origin is None: # Held until the origin is found, then placed with the rest
                f["_wait"] = True
                st.waiting.append((i, f, dur))
                return f
        w, d = self._near(st, i, ch, f["start"], origin, dur)
        if w is None or d > self._tol(st, w):
            st.unlabelled += 1
            return f
        key = (w.sequence, w.superframe, w.frame, w.slot)
        if key in st.cut: # (a window cut by the plan already: not twice)
            return {**f, "_dup": True}
        st.cut.add(key)
        st.labelled += 1
        return {**f, **w.fields(self.plan), "_window": w}

    def _vote(self, st, i, ch, t, dur=None):
        """Acquiring the origin: every window of the carrier in one superframe would put the burst at origin o = burst
        tick - t*clock; once `lock` bursts agree on one o (to the tolerance), it is the origin"""
        p = self.plan
        for s in p.sequences.values():
            span = s.duration*(len(s.hop.channels) if s.hop is not None else 1) # (a hopping carrier: its whole cycle)
            cands = [w.burst - t*p.clock for w in p.windows(s.start, s.start + span, sequences=[s.id])
                     if abs(w.freq - st.cf - ch.freq) <= 0.5*max(w.bw, 1.0) and self._fits(w, dur)]
            if s.hop is not None:
                s_dur = span
            else:
                s_dur = s.duration
            st.votes.append((s.id, s_dur, np.asarray(cands, dtype=np.float64)))
        st.votes = st.votes[-max(4*self.lock, 32):]
        if len(st.votes) < self.lock:
            return
        tol = self._tol(st)*p.clock
        best, n_best = None, 0
        for sid, dur, cands in st.votes:
            for o in cands:
                n = sum(np.any(np.abs(track.wrap(c - o, d2)) <= tol) for s2, d2, c in st.votes if s2 == sid)
                if n > n_best:
                    best, n_best = (o, dur), n
        if best is not None and n_best >= self.lock:
            st.origin = best[0]
            st.tr = track.new()
            track.update(st.tr, best[0], 0.0, period=best[1], bw=self.bw)

    def feedback(self, meta, ctx, start):
        """A burst's precise start [s from the capture's first sample], from its preamble: the tracked origin follows
        it (a profile's burst runner calls this, with the burst's ctx, with what its frame sync found). The preambles of
        a superframe make one measurement (their median, at their median time): a block estimate the tracker takes once a
        superframe, as bursts on several carriers come at once"""
        st = meta.state_of(self)
        w = (ctx or {}).get("burst_window")
        if self.plan is None or w is None or not st.tr.started:
            return
        dur = self.plan.sequences[w.sequence].duration
        pred = self._origin(st, start)
        st.fb.append((start, pred + track.wrap(w.burst - start*self.plan.clock - pred, dur)))
        if start - st.fb[0][0] >= dur/self.plan.clock:
            t = np.array([a for a, _ in st.fb])
            o = np.array([b for _, b in st.fb])
            track.update(st.tr, float(np.median(o)), float(np.median(t)), period=dur, bw=self.bw)
            st.fb = []

    # ---- gate "plan": the windows cut whether their energy shows or not

    def _windows(self, st, i, ch, flush):
        """The plan's windows of carrier i in what the carrier's history holds, not cut yet: (first, last sample, fields)"""
        p = self.plan
        end_s = (ch.held_pos + len(ch.held))/ch.Fs - ch.delay
        start_s = ch.held_pos/ch.Fs - ch.delay
        margin = self.margin if self.margin is not None else self._pad(ch)/ch.Fs
        o = self._origin(st, start_s)
        out = []
        for w in p.windows(o + start_s*p.clock, o + end_s*p.clock, assigned=bool(p.assignments)):
            if abs(w.freq - st.cf - ch.freq) > 0.5*max(w.bw, 1.0):
                continue
            key = (w.sequence, w.superframe, w.frame, w.slot)
            if key in st.cut:
                continue
            a = (w.start - o)/p.clock - margin
            b_ = (w.start + w.length - o)/p.clock + margin
            st.cut.add(key)
            st.labelled += 1
            out.append((int(round((a + ch.delay)*ch.Fs)), int(round((b_ + ch.delay)*ch.Fs)),
                        {"carrier": i, "freq": ch.freq, "start": (w.burst - o)/p.clock, **w.fields(p), "_window": w}))
        return out

    @staticmethod
    def _max_len(p):
        return max((ft.duration for ft in p.frame_types.values()), default=0)/p.clock


class PlanTx(BurstTx):
    """Bursts placed by a plan: each burst's fields name its window (sequence, superframe, frame, slot; or `window`),
    and it goes where the plan puts it: `at` its burst start, `freq` its carrier, from the plan's time `origin` (clock
    ticks at the output's first sample) and the output's centre frequency `cf`

        tx = rp.sync.PlanTx(plan, origin=0, cf=14e9, Fs=8e6)
        tx.place(burst, {"superframe": 3, "frame": 0, "slot": 7})"""
    def __init__(self, plan, origin=0.0, cf=0.0, Fs=None, ramp_s=0.0):
        super().__init__(Fs, 0.0, ramp_s)
        self.plan, self.origin, self.cf = plan, float(origin), float(cf)
        # Bursts on several carriers start together, and each is placed so that its first symbol (not its pulse's tail) is
        # on its slot: a BTU held back lets one go a little before another
        self.lag = max((ft.btu_duration for ft in plan.frame_types.values()), default=0)/plan.clock

    def __str__(self):
        return f"PlanTx({self.plan.name})"

    def where(self, fields):
        """(at [s], freq [Hz from the centre], window) of a burst whose fields name its window"""
        p = self.plan
        f = dict(fields or {})
        seq = int(f.get("sequence", next(iter(p.sequences))))
        s = p.sequences[seq]
        count = int(f["superframe"])
        k = (count - s.count) % (1 << 16)
        t0 = s.start + k*s.duration
        ws = [w for w in p.windows(t0, t0 + s.duration, sequences=[seq])
              if w.superframe == count and w.frame == int(f["frame"]) and w.slot == int(f["slot"])]
        if not ws:
            raise ValueError(f"PlanTx: no window for {f} in {p.name}")
        w = ws[0]
        return (w.burst - self.origin)/p.clock, w.freq - self.cf, w

    def place(self, burst, fields):
        at, freq, _ = self.where(fields)
        self.add(burst, at, freq)
