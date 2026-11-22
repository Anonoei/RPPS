"""A profile's receive side around a `{"sync": "burst"}` step: per-burst demodulation

The steps before the burst step run on the stream, as any profile's do. The burst step cuts the stream into its bursts
(rpps.sync.Burst), and every burst runs through the steps after it in a runner of its own, built for it: whatever a stage,
a decoder, a marker or a header holds starts over on every burst (each burst is a stream of its own: a packet acquired
fresh), and the controls whose header comes after the burst step drive that burst's steps only.

A call gives a list, one output per burst that ended in the block (Blocks flattens it: `for pdu in src(n) + p.rx()`).
"""
from ..errors import ConfigError


class BurstRunner:
    """pre: the runner of the steps before the burst step (None when there are none), burst: the rpps.sync.Burst,
    make_post(): a new runner of the steps after it. `last` is the runner of the last burst (its controls, reports)"""
    def __init__(self, pre, burst, make_post, name):
        self.pre, self.burst, self.make_post, self.name = pre, burst, make_post, name
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
            out.append(run(b))
            self._msgs += run.take_messages() # (a burst's positions are absolute in the burst step's input)
            self.last = run
            self.bursts += 1
        return out

    def take_messages(self):
        """The frames decoded since the last call, every burst's, placed in the stream given"""
        Fs, cf, t0 = self._input
        out = list(self.pre.take_messages()) if self.pre is not None else []
        for m in self._msgs:
            out.append(m.placed(self.pre.to_input(m.sample) if self.pre is not None else m.sample, Fs, cf, t0))
        self._msgs = []
        return sorted(out, key=lambda m: m.sample)

    def __radd__(self, data):
        """`block + runner`, as a runner goes where a stage does"""
        return self(data)
