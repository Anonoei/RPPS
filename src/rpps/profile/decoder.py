"""A selected signal's samples to a profile's messages, block by block: for a program that shows what a signal carries

    dec = rp.profile.Decoder("adsb/mode_s", Fs=20e6, offset=-3.2e6, cf=1093.2e6)
    for start, x, t in blocks:                       # the analyser's blocks: its own sample index, the block's time
        for m in dec.feed(x, start=start, time=t):   # rp.profile.Message, placed in the caller's index and time
            print(m.kind, m.crc_ok, m.fields)
    dec.status                                       # {"state": "searching" | "measuring" | "locked" | "lost", "why", ...}

The Decoder brings the signal at `offset` from the samples' centre to the profile's rate (rp.profile.info: requires):
a shift and staged decimation (rp.signal.channelize), then a fractional resampler (FractionalPolyphase) to the exact rate;
an FM demodulator between them for a profile that takes an FM channel's audio (AX.25). A profile whose symbol rate the
standard leaves open (DVB-S, CCSDS) takes `baud`, or the Decoder measures it first (rp.sync.Lock.find on the channel, `bw`
wide), holding the samples until it can, and feeds them through once it has.

Blocks need not follow on: `start` is the caller's index of each block's first sample. An overlap is cut off; a gap up to
`fill` seconds is filled with zeros (the profile's loops coast, frames keep their place, the frames in it are lost); a
longer one starts the profile's stream over (a measured rate is kept). Work is about proportional to the samples fed (and
the zeros of a filled gap); `status["load"]` is the time a feed took over the time its samples span.

One Decoder for one signal; not thread-safe. A retune, a new sample rate or another signal: `reset()` or a new Decoder.
"""
import time as _time

import numpy as np

from ..errors import NoLock
from .catalogue import info as _info
from .message import _plain

class Decoder:
    """A profile's receiver behind a front end that brings a selected signal to it (see module)

    name: the profile; Fs: the rate of the samples fed; offset: the signal's centre from the samples' centre [Hz];
    cf: the samples' centre frequency [Hz] (for Message.freq, and the params that follow from it: BLE's channel, NR's fc);
    baud: the symbol rate, for a profile that leaves it open; bw: the channel's width when there is no rate to work it out
    from (for measuring one: the selected signal's width); sync: the profile's sync settings; fill: the longest gap filled
    with zeros [s]; params: the profile's params (those `cf` would give, given here, win)
    Raises ValueError when Fs can't hold what the profile needs.
    """
    WINDOW = 2.0 # A good message within this many messages' worth of samples (min_samples)...
    WINDOW_S = 2.0 # ...or seconds, whichever is longer, is "locked" (a bursty signal: squitters, advertisements)

    def __init__(self, name, Fs, offset=0.0, cf=None, baud=None, bw=None, sync=None, fill=0.05, **params):
        self.name, self.Fs, self.offset, self.cf = name, float(Fs), float(offset), cf
        self.sync, self.fill = sync, float(fill)
        self._given = dict(params)
        desc = _info(name, baud=baud, **params) # Checks the params and the rate
        self.info = desc
        req = desc.requires
        if cf is not None and req["from_cf"]: # Params the carrier gives, unless given
            from .. import _config
            values = {**{k: v["default"] for k, v in desc.params.items()}, **params, "cf": float(cf) + self.offset}
            for k, e in req["from_cf"].items():
                if k not in params:
                    params[k] = _config.evaluate(e, values, f"profile/{name} from_cf")
            desc = _info(name, baud=baud, **params)
        self.params, self.req = params, desc.requires
        self.bw = float(bw) if bw is not None else self.req["bw"]
        if self.bw is not None and self.bw > self.Fs:
            raise ValueError(f"{name} needs {self.bw/1e6:g} MHz of channel; samples at {self.Fs/1e6:g} MS/s hold "
                             f"{self.Fs/1e6:g} MHz")
        if self.req["baud"] is None and self.bw is None:
            self.bw = self.Fs
        self._anchors, self._load = [], None
        self._counts = {"messages": 0, "good": 0, "failed": 0}
        self._last_good = None
        self.reset()

    # ---- the stream

    def reset(self):
        """Start the profile's stream over (a seek, a pause, a retune): the next block starts a new one. A measured rate
        is kept"""
        self._front = self._rx = self._stream = None
        self._next = None      # The caller's index the next sample should have
        self._seg = None       # The caller's index of the profile stream's first sample
        self._held, self._held_at, self._why = [], None, None
        if self.req["baud"] is not None:
            self._build()

    def _build(self):
        """The front end and the profile's receiver, once the rate is known"""
        from .. import signal
        from ..signal.sample.fractional import FractionalPolyphase
        from ..wave import FMDemod
        from .profile import load
        req = self.req
        Fs_req = req["Fs"]
        bw = self.bw or req["bw"]
        if self.Fs < bw:
            raise ValueError(f"{self.name} needs {bw/1e6:g} MHz of channel; samples at {self.Fs/1e6:g} MS/s hold {self.Fs/1e6:g} MHz")
        front, Fs_mid, _ = signal.channelize(self.offset, bw, self.Fs, oversample=max(Fs_req/bw, 1.0))
        stages = [front]
        if req["input"] == "audio":
            stages.append(FMDemod(gain=1.0)) # The channel's audio (rad/sample: the profile scales it itself)
        if not np.isclose(Fs_req, Fs_mid, rtol=1e-9):
            stages.append(FractionalPolyphase(Fs_req/Fs_mid))
        from ..base import Pipeline
        self._front = Pipeline(*stages)
        self._Fs_req = Fs_req
        self._rx = load(self.name, sync=self.sync, **self.params).rx()
        self._stream = None
        self._delay = None

    # ---- feeding

    def feed(self, samples, start=None, time=None, flush=False):
        """The messages the samples complete (rp.profile.Message, in the caller's index and time).

        start: the caller's index of samples[0] (None: right after the last block); time: its time [s] (a radio's
        timestamp, the host clock), so messages carry it; flush: the stream ends here (frames held come out)"""
        t_wall = _time.perf_counter()
        x = np.asarray(samples)
        if x.ndim != 1:
            raise ValueError("samples: one channel, a 1-D array")
        if start is None:
            start = self._next if self._next is not None else 0
        start = int(start)
        if time is not None:
            self._anchors = (self._anchors + [(start, float(time))])[-64:]
        if self._next is not None and start != self._next: # Not where the last block ended
            gap = start - self._next
            if gap < 0: # An overlap: what was fed already is cut off
                x = x[min(-gap, len(x)):]
                start = self._next
            elif gap <= self.fill*self.Fs and (self._rx is not None):
                x = np.concatenate((np.zeros(gap, np.complex64), x.astype(np.complex64)))
                start = self._next
            else:
                self._restart()
        if self._seg is None:
            self._seg = start
        self._next = start + len(x)
        if self._rx is None: # The rate is not known yet: measure it on what has come
            x = self._measure(x, start)
            if self._rx is None:
                return []
            start = self._seg
        msgs = self._run(x, flush)
        span = len(samples)/self.Fs
        if span > 0:
            load = (_time.perf_counter() - t_wall)/span
            self._load = load if self._load is None else 0.8*self._load + 0.2*load
        return msgs

    def close(self):
        """The end of the stream: what the profile holds comes out"""
        if self._rx is None:
            return []
        return self._run(np.zeros(0, np.complex64), True)

    def _restart(self):
        keep = self._rx is not None or self.req["baud"] is not None
        self._front = self._rx = self._stream = None
        self._next = self._seg = None
        self._held, self._held_at = [], None
        if keep:
            self._build()

    def _run(self, x, flush):
        from ..base import Data
        sps = {**{k: v["default"] for k, v in self.info.params.items()}, **self.params}.get("sps")
        sps_in = sps*self.Fs/self._Fs_req if sps else 0.0 # (the stages scale it with the rate)
        d = Data(x.astype(np.complex64), Fs=self.Fs, sps=sps_in, cf=(self.cf + self.offset) if self.cf is not None else 0.0,
                 stream=self._stream, flush=flush)
        self._stream = d.stream
        y = d + self._front
        if self._delay is None and len(y):
            self._delay = y.meta.delay
        self._rx(y)
        out = []
        for m in self._rx.take_messages():
            out.append(self._place(m))
        return out

    def _place(self, m):
        """A message from the profile's stream (its rate, from its start) in the caller's index and time"""
        k = self._seg + (m.sample/self._Fs_req - (self._delay or 0.0))*self.Fs
        m = m.placed(k, self.Fs)
        anchors = [a for a in self._anchors if a[0] <= m.sample] or self._anchors[:1]
        if anchors:
            s0, t0 = anchors[-1]
            m.time = t0 + (m.sample - s0)/self.Fs
        else:
            m.time = m.sample/self.Fs
        m.freq = (self.cf + self.offset) if self.cf is not None else None
        self._counts["messages"] += 1
        if m.crc_ok is False:
            self._counts["failed"] += 1
        elif m.crc_ok or m.crc_ok is None:
            self._counts["good"] += 1
            self._last_good = m.sample
        return m

    # ---- a rate to measure

    def _measure(self, x, start):
        """Hold the channel until Lock.find gives the rate; then build the receiver and give back what was held (from
        its start, which becomes the stream's)"""
        from .. import signal, sync
        if self._held_at is None:
            self._held_at = start
            self._meas, self._Fs_meas, _ = signal.channelize(self.offset, self.bw, self.Fs, oversample=4.0)
            self._meas_stream, self._meas_x = None, []
        self._held.append(x)
        from ..base import Data
        d = Data(x.astype(np.complex64), Fs=self.Fs, stream=self._meas_stream)
        self._meas_stream = d.stream
        self._meas_x.append(np.asarray((d + self._meas).data))
        need = max(self.req["min_samples"]*self._Fs_meas/self.Fs*4, 16384)
        have = sum(len(a) for a in self._meas_x)
        if have < need:
            self._why = f"measuring the symbol rate: {have} of {int(need)} samples"
            return x[:0]
        z = np.concatenate(self._meas_x)
        try:
            lk = sync.Lock.find(z[-int(need):], self._Fs_meas, min_samples=1024)
        except NoLock as e:
            self._why = e.reason
            self._meas_x = [z[-int(need)//2:]] # Try again when as much again has come
            raw = np.concatenate(self._held)
            keep = int(need//2*self.Fs/self._Fs_meas)
            self._held_at += max(len(raw) - keep, 0)
            self._held = [raw[-keep:]]
            return x[:0]
        self._why = None
        self.req = requirements_with(self.info, self.params, lk.det.baud)
        self.measured = lk
        self._build()
        held, at = np.concatenate(self._held), self._held_at
        self._held, self._held_at = [], None
        self._seg = at
        return held

    # ---- what is going on

    @property
    def status(self):
        """{"state": "measuring" (the rate) | "searching" (no good message yet) | "locked" (one lately) | "lost" (none
        lately), "why": the reason when there is one, "baud", "Fs": the profile's rate, "load": a feed's time over its
        samples' time, "messages"/"good"/"failed": counts, "est": the profile's estimates (plain values)}"""
        if self._rx is None:
            state = "measuring"
        elif self._last_good is None:
            state = "searching"
        else:
            window = max(self.WINDOW*self.req["min_samples"]*self.Fs/self.req["Fs"], self.WINDOW_S*self.Fs)
            state = "locked" if (self._next or 0) - self._last_good <= window else "lost"
        est = {}
        if self._stream is not None:
            est = {k: _plain(v) for k, v in self._stream.est.items() if isinstance(v, dict)}
        return {"state": state, "why": self._why, "baud": self.req["baud"], "Fs": self.req["Fs"], "load": self._load,
                **self._counts, "est": est}

def requirements_with(desc, params, baud):
    """A ProfileInfo's requires again, the rate now known"""
    return _info(desc.name, baud=baud, **params).requires
