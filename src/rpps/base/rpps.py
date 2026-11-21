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
        else:
            ctx.setdefault(DROPPED, []).append(len(rest)) # (a parallel region's worker reports it: a frame cut short)
            if log is not None:
                log.warn(f"Dropped {len(rest)} trailing items short of a block of {block}")
    return x[:n]

STREAM, FRAME = "stream", "frame"
# Scope of an operator's state (README.dev.md, Parallelism). STREAM: it carries state from one frame to the next (trackers,
# frame sync, convolutional decoders), so frames can't be processed apart. FRAME: it holds nothing across a frame start, so
# a frame is a pure function of its data and the parameters controls gave it. Default STREAM (safe).

DROPPED = "dropped" # ctx entry: items a flushed decoder had to drop, as [count, ...]
FLUSH = "flush" # ctx entry: this is the end of the stream, so nothing is held back (set by Data)
def failed_of(data):
    """The data's `failed` mask (True where a decoder before could not decode it), or None: nothing known to have failed"""
    m = getattr(data, "failed", None)
    return m if m is not None and len(m) == len(data) else None


class Pipe:
    """Parent class for RPPS implementations

    A pipe works on data with the operators: `data * pipe` encodes, `data / pipe` decodes (and `data + pipe` for
    stages that process a signal). Two pipes multiplied/added together give a `Pipeline`, a reusable chain:

        link = scr * ecc * mod          # scrambler, then code, then modulation
        syms = bits * link              # same as bits * scr * ecc * mod
        bits = syms / link              # the decoders, run backwards
    """
    __array_ufunc__ = None # numpy defers to the operators here instead of broadcasting over the object
    CATEGORY = None # The rpps.base.types.Category of what it does (stages without one count as SIGNAL)
    SCOPE = STREAM  # STREAM or FRAME: where its state lives (see above)
    SCOPE_NEEDS = () # FRAME only while a control drives these parameters (a scrambler restarts at a frame start by `length`)

    def takes(self, encode):
        """The representations (rpps.base.types.Rep) this accepts to encode (True) or decode (False); None: not checked"""
        return None

    def __str__(self) -> str:
        return "Pipe"
    def encode(self, data):
        """Encode data"""
        return NotImplemented
    def decode(self, data):
        """Decode data"""
        return NotImplemented
    def __rmul__(self, data):
        """data * pipe: encode, after checking the pipe takes what data is"""
        from .data import expect
        expect(self, data, self.takes(True), True)
        return self.encode(data)
    def __rtruediv__(self, data):
        """data / pipe: decode, after checking the pipe takes what data is"""
        from .data import expect
        expect(self, data, self.takes(False), False)
        return self.decode(data)

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
    data + pipeline: every item processes in order (stages). Pipelines nest and flatten. A stage that splits the data
    (`rp.sync.Burst`) makes the rest run on every piece, and the pipeline gives the list of what they made.
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
        out, split = [data], False
        for it in self.items:
            nxt = []
            for d in out:
                r = d + it
                if isinstance(r, list):
                    nxt += r
                    split = True
                else:
                    nxt.append(r)
            out = nxt
        return out if split else out[0]


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

    def takes(self, encode=True):
        """Representations this stage processes (rpps.base.types.Rep): samples and symbols"""
        from .types import Rep
        return {Rep.SAMPLES, Rep.SYMBOLS}

    def __radd__(self, data):
        from .data import _Data
        if not isinstance(data, _Data):
            return NotImplemented
        return data.apply(self)

class BitStage(Stage):
    """A stage on a stream of bits: channel bits or data bits, as LLRs or hard bits (sync on bits: alignment, polarity)

    `step` gets a numpy array: float LLRs (positive means 0) or bools, and gives the same form back. The output keeps the
    input's representation and form; state carries across blocks as for any stage, so any block length works.
    """
    def takes(self, encode=True):
        from .types import Rep
        return {Rep.CHANNEL, Rep.DATA}

def tag_stages(package, category):
    """Give every Stage class defined in `package` (a module or package name, submodules included) the Category `category`,
    unless the class sets its own. In this architecture the package is the category (README.dev.md)"""
    import importlib, pkgutil, sys
    root = importlib.import_module(package)
    names = [package]
    if hasattr(root, "__path__"):
        names += [m.name for m in pkgutil.walk_packages(root.__path__, package + ".")]
    for name in names:
        mod = sys.modules.get(name) or importlib.import_module(name)
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, Stage) and obj.__module__ == name and "CATEGORY" not in vars(obj):
                obj.CATEGORY = category

class Tee(Pipe):
    """The same blocks through several branches: `data + Tee(name=stage_or_pipeline, ...)` gives {name: data out}

        out = mpx + rp.Tee(mono=audio, stereo=pilot + audio)     # one `audio` pipeline serves both
        left, right = out["mono"].data + out["stereo"].data, ...

    Each branch runs in a stream of its own (kept in the stream's state under the tee), so a stage can sit in several
    branches, each with its state, and every branch is as independent of the block cut as a chain is. A branch's
    estimates and states are on its output (`out["stereo"].state(pilot)`, `out["stereo"].stream.est`). A tee ends a
    chain: what follows takes one branch's output."""
    def __init__(self, **branches):
        if not branches:
            raise ValueError("a Tee needs at least one branch: Tee(name=stage, ...)")
        self.branches = branches

    def __str__(self):
        return "Tee(" + ", ".join(f"{k}={v}" for k, v in self.branches.items()) + ")"

    def __radd__(self, data):
        from .data import _Data
        from .meta import Stream
        if not isinstance(data, _Data):
            return NotImplemented
        parent = data.meta.stream
        subs = parent.states.get(self)
        if subs is None: # The branches' streams, made when the tee first runs in this stream
            subs = parent.states[self] = {k: Stream(parent.src_Fs) for k in self.branches}
            for s in subs.values():
                s.t0 = parent.t0
        out = {}
        for name, branch in self.branches.items():
            meta = data.meta.fork()
            meta.state = data.meta.state.copy() # (the block as it is, not as a stage left it)
            meta.op = data.meta.state.copy()
            meta.stream, meta.gen, meta.flush = subs[name], subs[name].gen, data.meta.flush
            out[name] = type(data)(data.data, data.CAT, data.DT, dict(data.ctx), meta, data.REP) + branch
        return out
