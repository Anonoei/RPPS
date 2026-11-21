"""Batch kernels of the code operators: `BlockDecode` of a block code or interleaver, `XorMask` of an additive scrambler

Checked row for row against the operator run frame by frame (tests/unit/test_batch_kernels.py); they use only what
numpy and cupy share, so `xp` can be either (README.dev.md, Parallelism)
"""
import numpy as np

from ..base.data import Digital, Mod
from ..base.kernel import Form, Kernel
from ..base.types import dtype, Rep
from ..errors import RPPSError


class BlockDecode(Kernel):
    """A decoder or deinterleaver of whole blocks, over a batch: the frames' blocks go through the coder in one call (a block code's
    decoder already runs its codewords on several threads, so a batch is more of them at once, not more calls). Each block is
    independent of the others, so a row comes out as it would alone. A frame is whole blocks; items past the last block are
    dropped and reported, as the operator does at the end of a flushed stream.

    coder: what the operator decodes with (`decode(bits or LLRs) -> bits or LLRs`, any number of blocks); block: items a
    block takes in; soft: it takes LLRs when it is given them"""
    def __init__(self, coder, block, soft, category, rep, name="block"):
        self.coder, self.block, self.soft, self.category, self.rep = coder, int(block), bool(soft), category, rep
        self.name = f"decode:{name}"

    def shape(self, n_in):
        return None # a code says how many items a block gives, not every coder how its input and output relate

    def _in(self, b, X):
        f = b.form
        if f.dt == dtype.LLR:
            return X if (self.soft and issubclass(f.cls, Mod)) else X < 0
        return X.astype(bool)

    def run(self, xp, b):
        N, L = b.x.shape
        used = L//self.block*self.block
        for i in range(N):
            if used < L and not b.bad[i]:
                b.notes.setdefault(i, []).append({"what": "cut short", "items": int(L - used), "at": int(b.pos[i] + b.n_in)})
        if used == 0:
            b.x = np.zeros((N, 0), dtype=bool)
            b.form = Form(Digital, self.category, dtype.BITS, self.rep, *b.form[4:])
            return b
        x = self._in(b, b.x[:, :used])
        try:
            y = np.asarray(self.coder.decode(x.reshape(-1))).reshape(N, -1)
        except RPPSError: # one frame's block was bad: find which, the others go on
            rows = []
            for i in range(N):
                try:
                    rows.append(np.asarray(self.coder.decode(x[i])).reshape(-1))
                except RPPSError as e:
                    b.fail(i, f"{type(e).__name__}: {e}")
                    rows.append(None)
            ok = next((r for r in rows if r is not None), None)
            if ok is None:
                b.x = np.zeros((N, 0), dtype=bool)
                return b
            y = np.zeros((N, len(ok)), dtype=ok.dtype)
            for i, r in enumerate(rows):
                if r is not None:
                    y[i] = r
        soft = y.dtype.kind == "f"
        b.x = y
        b.form = Form(Mod if soft else Digital, self.category, dtype.LLR if soft else dtype.BITS, self.rep, *b.form[4:])
        errs = getattr(self.coder, "errors", None) # This decoder's verdict per row (the operator's FAILED mask, per frame)
        if errs is not None and len(errs) and len(errs) % N == 0:
            b.code_failed = (np.asarray(errs).reshape(N, -1) < 0).any(axis=1)
        return b

class XorMask(Kernel):
    """Bits XOR a fixed mask repeated from the start of each frame (a scrambler restarted every `len(mask)` bits). LLRs
    stay LLRs, their signs flipped (as the additive scrambler descrambles them)"""
    def __init__(self, mask, category, rep, name="xor"):
        self.mask, self.category, self.rep = np.asarray(mask, dtype=bool), category, rep
        self.name = f"xor:{name}"

    def shape(self, n_in):
        return n_in

    def consts(self):
        return {"mask": self.mask}

    def run(self, xp, b):
        L = b.x.shape[1]
        m = xp.asarray(self.mask[np.arange(L) % len(self.mask)])[None, :]
        if b.form.dt == dtype.LLR and self.rep == Rep.DATA: # Descrambling (encoding decides first, as the operator does)
            b.x = (b.x*(1.0 - 2.0*m)).astype(b.x.dtype)
            b.form = Form(Mod, self.category, dtype.LLR, self.rep, *b.form[4:])
        else:
            b.x = (b.x < 0 if b.form.dt == dtype.LLR else b.x.astype(bool)) ^ m
            b.form = Form(Digital, self.category, dtype.BITS, self.rep, *b.form[4:])
        return b
