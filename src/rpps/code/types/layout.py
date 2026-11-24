"""Where a block code's bits go in the word sent, and what is done to them on the way: keys of any block definition

    "layout": the sent position of each codeword bit (codeword order: the code's own, message first for linear, cyclic,
              bch, rs...), as a list (or {"each": ..., "n": N}), or {"message": [...], "parity": [...]} with "rest" for
              one of them (the positions the other leaves, in order)
    "xor":    a fixed pattern on the sent word (a coset: inverted or masked parity), a number of n bits (first sent bit
              most significant) or a list of bits
    "append": check bits after the codeword (before the layout), each the XOR of the codeword bits it lists (indices
              into the codeword and the bits appended before it): P25's NID bit 64 is [[14, 15]]. Checked after the
              inner decode: a mismatch is one more error, a failure past the code's `t`
    "extend": an overall parity bit, of every bit before it (an extended code); a code with matrices (linear, cyclic,
              expr) builds it into them, any other gets it appended here

On encode: the code, the appended bits, the layout, the pattern; on decode the same undone in reverse, on bits or LLRs alike.
"""
import numpy as np

from ._code import _code

def layout_perm(spec, n):
    """`to` map of a layout spec: codeword bit j goes to sent position perm[j], for an n-bit word"""
    if isinstance(spec, dict):
        if set(spec) - {"message", "parity", "comment"} or not {"message", "parity"} <= set(spec):
            raise ValueError(f"layout {spec}: give \"message\" and \"parity\" (one of them may be \"rest\")")
        msg, par = spec["message"], spec["parity"]
        if msg == "rest" and par == "rest":
            raise ValueError("layout: only one of message / parity can be \"rest\"")
        given = [int(p) for p in (par if msg == "rest" else msg)]
        rest = [p for p in range(n) if p not in set(given)]
        perm = rest + given if msg == "rest" else given + ([int(p) for p in par] if par != "rest" else rest)
    else:
        perm = [int(p) for p in spec]
    perm = np.asarray(perm, dtype=np.int64)
    if len(perm) != n or not np.array_equal(np.sort(perm), np.arange(n)):
        raise ValueError(f"layout: not a placement of the {n} codeword bits on {n} positions (repeated, missing or out of range)")
    return perm

def xor_mask(spec, n):
    from ...sequence.word import word_bits
    m = word_bits(spec, n) if not isinstance(spec, (list, tuple)) else np.asarray(spec).astype(bool)
    if len(m) != n:
        raise ValueError(f"xor: {len(m)} bits for a {n}-bit word")
    return m

class arranged(_code):
    """A block code with check bits appended (`append`, `extend`), its bits placed (`layout`) and masked (`xor`) as sent.
    `inner`: the code (its encoder), `dec`: what decodes it (default: the code itself). Decode status is the decoder's,
    plus each appended bit that disagrees with the decision"""
    def __init__(self, log, inner, layout=None, xor=None, extend=False, dec=None, append=None):
        eqs = [np.asarray(e, dtype=np.int64) for e in (append or [])]
        n0 = inner.den
        for j, e in enumerate(eqs):
            if len(e) == 0 or e.min() < 0 or e.max() >= n0 + j:
                raise ValueError(f"append {j}: codeword bits 0..{n0 + j - 1}, not {e.tolist()}")
        if extend: # Overall parity: of every bit before it
            eqs.append(np.arange(n0 + len(eqs)))
        n = n0 + len(eqs)
        super().__init__(log, getattr(inner, "name", None), inner.num, n)
        self.inner, self.eqs = inner, eqs
        self.dec = inner if dec is None else dec
        self.soft = getattr(self.dec, "soft", False)
        self.t = getattr(self.dec, "t", None)
        if self.t is None:
            self.t = getattr(inner, "t", None)
        self.to = None if layout is None else layout_perm(layout, n) # codeword bit j -> sent position to[j]
        self.mask = None if xor is None else xor_mask(xor, n)

    def with_decoder(self, dec):
        """The same arrangement decoded by `dec` (a decoder built from `inner`)"""
        import copy
        out = copy.copy(self)
        out.dec, out.soft = dec, getattr(dec, "soft", False)
        out.t = getattr(dec, "t", None) if getattr(dec, "t", None) is not None else self.t
        return out

    def __getattr__(self, k):
        if k in ("encode_batch", "decode_batch"): # (they would skip the arrangement)
            raise AttributeError(k)
        return getattr(self.__dict__["inner"], k) if "inner" in self.__dict__ else object.__getattribute__(self, k)

    def _appended(self, c):
        for e in self.eqs:
            c = np.concatenate((c, np.bitwise_xor.reduce(c[:, e], axis=1)[:, None]), axis=1)
        return c

    def encode(self, bits):
        c = self._appended(np.asarray(self.inner.encode(bits)).astype(bool).reshape(-1, self.inner.den))
        if self.to is not None:
            out = np.empty_like(c)
            out[:, self.to] = c
            c = out
        if self.mask is not None:
            c = c ^ self.mask
        return c.reshape(-1)

    def _decode(self, x, soft=None, **kw):
        x = np.asarray(x)
        soft = x.dtype.kind == "f" if soft is None else soft
        w = x[:len(x)//self.den*self.den].reshape(-1, self.den)
        if self.mask is not None:
            w = np.where(self.mask, -w, w) if soft else w.astype(bool) ^ self.mask
        if self.to is not None:
            w = w[:, self.to]
        if not self.eqs:
            return self.dec.decode(np.ascontiguousarray(w).reshape(-1), status=True, **kw)
        n0 = self.inner.den
        u, errs = self.dec.decode(np.ascontiguousarray(w[:, :n0]).reshape(-1), status=True, **kw)
        if errs is None:
            return u, None
        got = (w[:, n0:] < 0) if soft else w[:, n0:].astype(bool)
        again = self._appended(np.asarray(self.inner.encode(np.asarray(u).astype(bool))).astype(bool).reshape(len(w), -1))[:, n0:]
        off = np.count_nonzero(again != got, axis=1)
        errs = np.asarray(errs).copy()
        ok = errs >= 0
        errs[ok] += off[ok]
        if self.t is not None:
            errs[errs > self.t] = -1 # past what the code corrects, counting the check bits: a miscorrection
        return u, errs
