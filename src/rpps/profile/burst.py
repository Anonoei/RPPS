"""A profile around a `{"sync": "burst"}` step: per-burst demodulation (BurstRunner), and burst transmission (BurstTxRunner)

The steps before the burst step run on the stream, as any profile's do. The burst step cuts the stream into its bursts
(rpps.sync.Burst), and every burst runs through the steps after it in a runner of its own, built for it: whatever a stage,
a decoder, a marker or a header holds starts over on every burst (each burst is a stream of its own: a packet acquired
fresh), and the controls whose header comes after the burst step drive that burst's steps only.

A call gives a list, one output per burst that ended in the block (Blocks flattens it: `for pdu in src(n) + p.rx()`).

Transmitting is the mirror: every burst through a runner of its own of the steps after the burst step, the bursts placed
on one stream (rpps.sync.BurstTx: gaps, start times, ramps, a carrier each), the steps before it on that stream.
"""
from ..errors import ConfigError


class BurstRunner:
    """pre: the runner of the steps before the burst step (None when there are none), burst: the rpps.sync.Burst,
    make_post(): a new runner of the steps after it. `last` is the runner of the last burst (its controls, reports)"""
    kinds = None # The message kinds handed out (Profile.rx(kinds=...)), None: all

    def __init__(self, pre, burst, make_post, name, entry=None):
        self.pre, self.burst, self.make_post, self.name = pre, burst, make_post, name
        self.entry = entry # The controls entry of the burst step: its fields per burst (rpps.sync.Bursts) drive the burst's steps
        self.last = None
        self.bursts = 0
        self._msgs, self._input = [], (1.0, 0.0, None)

    @property
    def ops(self):
        """[(how, operator)] of every step: those on the stream, the burst step, those of a burst (a runner built to show them)"""
        from .steps import STAGE
        return (list(self.pre.ops) if self.pre is not None else []) + [(STAGE, self.burst)] + list(self.make_post().ops)

    def __iter__(self):
        return iter(self.ops)

    def __repr__(self):
        return f"<BurstRunner {self.name}: {self.pre!r} | {self.burst} | per burst>"

    def __call__(self, data, fields=None):
        if fields is not None:
            raise ConfigError(f"{self.name}: fields are for transmitting")
        self._input = (data.meta.Fs or 1.0, data.meta.cf, data.meta.stream.t0)
        if self.pre is not None:
            data = self.pre(data)
        out = []
        for b in data + self.burst:
            run = self.make_post()
            fields = b.ctx.get("burst")
            if fields and self.entry is not None and self.entry.on in ("both", "rx"):
                run.inject(self.entry, fields, b.meta.pos, ("burst", b.meta.pos))
            out.append(run(b))
            if fields and hasattr(self.burst, "feedback"):
                start = self._preamble_at(run, b)
                if start is not None:
                    self.burst.feedback(data.meta, b.ctx, start)
            msgs = run.take_messages() # (a burst's positions are absolute in the burst step's input)
            if fields:
                for m in msgs: # Where the burst was: its carrier, and its place in a plan
                    m.fields.setdefault("burst", dict(fields))
            self._msgs += msgs
            self.last = run
            self.bursts += 1
        return out

    @staticmethod
    def _preamble_at(run, b):
        """Where the burst's frame sync found its preamble [s from the capture's first sample], or None"""
        from ..sync import FrameSync
        from ..base.control import of
        log = of(run._stream).log if run._stream is not None else []
        for e in log:
            slots = run._step_slots.get(e["step"], [])
            op = run.slots[slots[0]].op if slots else None
            if isinstance(op, FrameSync) and "frame" in e["fields"]:
                pos = run._back(e["at"], slots[0], 0) # (the burst's input: samples of its carrier's stream)
                delay = run._stream.est.get(op.name, {}).get("delay", b.meta.delay) # (all the delay before the frame sync)
                return pos/b.meta.Fs - delay
        return None

    def take_messages(self):
        """The frames decoded since the last call, every burst's, placed in the stream given"""
        Fs, cf, t0 = self._input
        out = list(self.pre.take_messages()) if self.pre is not None else []
        for m in self._msgs:
            out.append(m.placed(self.pre.to_input(m.sample) if self.pre is not None else m.sample, Fs, cf, t0))
        self._msgs = []
        return sorted((m for m in out if self.kinds is None or m.kind in self.kinds), key=lambda m: m.sample)

    def __radd__(self, data):
        """`block + runner`, as a runner goes where a stage does"""
        return self(data)


class BurstTxRunner:
    """The transmit side of a profile with a burst step: a call sends one burst

        tx = p.tx()
        out = tx(rp.Data(bits, Fs=1e6, sps=1), fields={"type": 2})       # the samples up to this burst's start
        out = tx(rp.Data(bits2, Fs=1e6, sps=1), at=4e-3, freq=2e6, flush=True)  # the rest: the stream ends

    Each burst runs through a runner of its own of the steps after the burst step (make_post(): built anew, so every
    scrambler, header, marker and pulse starts over, and the burst ends whole: its data is sent as a stream of its own,
    flushed). The burst step's rpps.sync.BurstTx places it on the output stream: `at` [s from the stream's start], else
    `gap_s` after the burst before; `freq` [Hz from the centre], else the step's `hop`. The steps before the burst step
    (pre, None for none) run on that stream. With a plan (rpps.sync.PlanTx) the burst's fields name its window: it goes
    there, at its symbol rate, and its window's fields (its tx type's) drive the burst's steps. A call gives the samples no later burst can reach any more; `flush=True` (or
    `flush()`) gives the rest and ends the stream. A list of blocks is sent burst by burst (fields, at, freq: one each, or
    one for all). `last` is the runner of the last burst, `sent` the BurstTx's record: (start sample, samples, freq)"""

    def __init__(self, make_post, burst_tx, pre, name, entry=None):
        self.make_post, self.burst_tx, self.pre, self.name = make_post, burst_tx, pre, name
        self.entry = entry # The controls entry of the burst step: a plan's fields of the burst's window drive its steps
        self.last = None
        self.bursts = 0
        burst_tx.reset()

    @property
    def sent(self):
        return self.burst_tx.sent

    @property
    def ops(self):
        """[(how, operator)] in the order they run: a burst's steps, the burst step's BurstTx, those on the stream"""
        from .steps import STAGE
        return list(self.make_post().ops) + [(STAGE, self.burst_tx)] + (list(self.pre.ops) if self.pre is not None else [])

    def __iter__(self):
        return iter(self.ops)

    def __repr__(self):
        return f"<BurstTxRunner {self.name}: per burst | {self.burst_tx} | {self.pre!r}>"

    def __call__(self, data, fields=None, at=None, freq=None, flush=False):
        from ..base.data import _Data
        if isinstance(data, _Data):
            self._send(data, fields, at, freq)
        else:
            data = list(data)
            each = lambda v: list(v) if isinstance(v, (list, tuple)) else [v]*len(data)
            fields, at, freq = each(fields), each(at), each(freq)
            if not len(fields) == len(at) == len(freq) == len(data):
                raise ValueError(f"{self.name}: one fields, at and freq a burst (or one for all)")
            for d, f, a, fr in zip(data, fields, at, freq):
                self._send(d, f, a, fr)
        return self._take(flush)

    def flush(self):
        """The rest of the stream, ending it"""
        return self._take(True)

    def _send(self, data, fields, at, freq):
        from ..base.data import _Data
        if not isinstance(data, _Data):
            raise TypeError(f"{self.name}: a burst is a Data block (rp.Data(bits, Fs=..., sps=...)), not {type(data).__name__}")
        d = data.derive(data.data, data.CAT, data.DT, rep=data.REP) # A stream of its own, ended: the burst goes out whole
        from ..base.meta import Meta
        m = data.meta
        w = None
        if at is None and freq is None and hasattr(self.burst_tx, "where"): # A plan places it (rpps.sync.PlanTx), at its rate
            at, freq, w = self.burst_tx.where(fields)
        d.meta = Meta(m.Fs if w is None else w.symbol_rate, m.sps, m.cf, m.delay)
        d.flush = True
        run = self.make_post()
        if w is not None and self.entry is not None and self.entry.on in ("both", "tx"): # its window's fields drive its steps
            run.inject(self.entry, {**w.fields(self.burst_tx.plan), **(fields or {})}, d.meta.pos, ("burst", self.bursts))
        out = run(d, fields=fields)
        if w is not None: # The plan's burst start is its first symbol's: the samples before it (a pulse's tail) go earlier
            at -= out.meta.delay or 0.0
        self.burst_tx.add(out, at, freq)
        self.last = run
        self.bursts += 1

    def _take(self, flush):
        out = self.burst_tx.take(flush)
        return self.pre(out) if self.pre is not None else out
