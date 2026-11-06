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
    """Parent class for RPPS implementations"""
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


class Stage(Pipe):
    """A DSP stage operating on a continuous stream of blocks

    Settings (constructor args) live on the stage. Anything that must carry from one block to the
    next lives in a state object returned by `new_state` and stored in `meta.ctx`, so the stage
    itself stays reusable, and `meta.restart()` resets the whole pipeline.

    A stage must accept any block length (including 0), and neither drop nor repeat samples across blocks.
    """
    def new_state(self, meta):
        """Create this stage's carried state"""
        return None

    def step(self, samples, st, meta):
        """Process a block, returning the output. Update meta.op (and meta.stream.delay) as needed"""
        raise NotImplementedError()

    def __radd__(self, meta):
        meta.obj = self.step(meta.obj, meta.state_of(self), meta)
        meta.finish()
        return meta

    def __rsub__(self, meta):
        return self.__radd__(meta)
