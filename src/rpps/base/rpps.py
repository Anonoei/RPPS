"""RPPS parent classes"""
import numpy as np


def align(ctx, key, x, block, pad, log=None):
    """Whole blocks of x for a stage that works on `block` items at a time

    ctx: the stream state (a dict this call updates) holding the leftover of the previous call under `key`.
    A leftover waits for the next call. Only when the stream is flushed is it zero padded up to a whole block
    (pad: encoding) or dropped (decoding, where padding would invent data).
    """
    x = np.asarray(x)
    prev = ctx.pop(key, None)
    if prev is not None and len(prev):
        x = np.concatenate((prev, x))
    n = len(x)//block*block
    rest = x[n:]
    if len(rest):
        if not ctx.get(FLUSH):
            ctx[key] = rest
        elif pad:
            x = np.concatenate((x[:n], rest, np.zeros(block - len(rest), dtype=x.dtype)))
            n = len(x)
        elif log is not None:
            log.warn(f"Dropped {len(rest)} trailing items short of a block of {block}")
    return x[:n]

FLUSH = "flush" # ctx entry: this is the end of the stream, so nothing is held back (set by Data)


class Pipe:
    """Parent class for RPPS implementations

    A pipe works on data with the operators: `data * pipe` encodes, `data / pipe` decodes (and `data + pipe` for
    stages that process a signal). Two pipes multiplied/added together give a `Pipeline`, a reusable chain:

        link = scr * ecc * mod          # scrambler, then code, then modulation
        syms = bits * link              # same as bits * scr * ecc * mod
        bits = syms / link              # the decoders, run backwards
    """
    __array_ufunc__ = None # numpy defers to the operators here instead of broadcasting over the object

    def __str__(self) -> str:
        return "Pipe"
    def encode(self, data):
        """Encode data"""
        return NotImplemented
    def decode(self, data):
        """Decode data"""
        return NotImplemented
    def __rmul__(self, data):
        """Encode data"""
        return NotImplemented
    def __rtruediv__(self, data):
        """Decode data"""
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Pipe):
            return Pipeline(self, other)
        return NotImplemented
    def __add__(self, other):
        if isinstance(other, Pipe):
            return Pipeline(self, other)
        return NotImplemented


class Pipeline(Pipe):
    """Pipes (and stages) applied in order. Build one by combining pipes: `a * b * c` or `a + b + c`

    data * pipeline: every item encodes in order. data / pipeline: every item decodes, last item first.
    data + pipeline: every item processes in order (stages). Pipelines nest and flatten.
    """
    __slots__ = ("items",)
    def __init__(self, *items):
        flat = []
        for it in items:
            flat += it.items if isinstance(it, Pipeline) else [it]
        self.items = tuple(flat)

    def __str__(self):
        return " | ".join(str(it) for it in self.items)

    def __repr__(self):
        return f"<Pipeline {self}>"

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, i):
        return Pipeline(*self.items[i]) if isinstance(i, slice) else self.items[i]

    def encode(self, data):
        for it in self.items:
            data = data * it
        return data

    def decode(self, data):
        for it in reversed(self.items):
            data = data / it
        return data

    def __rmul__(self, data):
        return self.encode(data)
    def __rtruediv__(self, data):
        return self.decode(data)

    def __radd__(self, data):
        from .data import _Data
        if not isinstance(data, _Data):
            return NotImplemented
        for it in self.items:
            data = data + it
        return data


class Stage(Pipe):
    """A DSP stage operating on a continuous stream of blocks: `block + stage`

    Settings (constructor args) live on the stage. Anything that must carry from one block to the
    next lives in a state object returned by `new_state`, kept in the stream the block belongs to, so the stage
    itself stays reusable. Stages never see Meta being created or reset: the data carries it.

    A stage must accept any block length (including 0), and neither drop nor repeat samples across blocks.
    """
    def new_state(self, meta):
        """Create this stage's carried state"""
        return None

    def step(self, samples, st, meta):
        """Process a block, returning the output. Set meta.op (the output rates) and add to meta.delay as needed"""
        raise NotImplementedError()

    def __radd__(self, data):
        from .data import _Data
        if not isinstance(data, _Data):
            return NotImplemented
        return data.apply(self)
