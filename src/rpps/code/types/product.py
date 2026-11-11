import numpy as np

from ._code import _code, register

@register("product")
class product(_code):
    """Product code (block product turbo code, BPTC): message in a k2 x k1 matrix, every row encoded with the row
    code (n1, k1), then every column (parity columns too) with the column code (n2, k2); sent row by row

        rows, cols: the two block codes, as references to block definitions ("hamming/dmr_15_11") or definitions
        reserved: zero bits at the start of the message area (DMR: R(2..0)); the message is k1 k2 - reserved bits
        lead: zero bits sent before the matrix (DMR: R(3)), so n = lead + n1 n2
        iters: decoding rounds of all columns then all rows (hard decisions, each code's own syndrome decoder)
    After a decode `errors` is the bits corrected per word, -1 where a row or column still fails its check.
    """
    @classmethod
    def from_config(cls, log, obj):
        return cls(log, _code_obj(obj["rows"]), _code_obj(obj["cols"]), obj.get("reserved", 0), obj.get("lead", 0),
                   obj.get("iters", 4))

    def __init__(self, log, rows, cols, reserved=0, lead=0, iters=4):
        self.rows, self.cols = rows, cols # Coding objects (encoder + hard decoder)
        self.k1, self.n1 = rows._enc.num, rows._enc.den
        self.k2, self.n2 = cols._enc.num, cols._enc.den
        self.reserved, self.lead, self.iters = int(reserved), int(lead), int(iters)
        super().__init__(log, None, self.k1*self.k2 - self.reserved, self.lead + self.n1*self.n2)

    def encode(self, bits):
        b = np.asarray(bits).astype(np.uint8)
        out = []
        for m in b[:len(b)//self.num*self.num].reshape(-1, self.num):
            area = np.concatenate((np.zeros(self.reserved, np.uint8), m)).reshape(self.k2, self.k1)
            rows = self.rows._enc.encode(area.reshape(-1)).reshape(self.k2, self.n1).astype(np.uint8)
            full = self.cols._enc.encode(rows.T.reshape(-1)).reshape(self.n1, self.n2).T # Columns encoded, back to rows
            out.append(np.concatenate((np.zeros(self.lead, np.uint8), full.reshape(-1))))
        return np.concatenate(out).astype(bool) if out else np.zeros(0, bool)

    def _fix(self, coder, words, n, k):
        """Decode each word (rows of `words`, n bits) with the code's hard decoder; returns words corrected and statuses"""
        dec = coder._dec
        msg = dec.decode(words.reshape(-1)).reshape(len(words), k).astype(np.uint8)
        st = np.asarray(coder.errors)
        fixed = coder._enc.encode(msg.reshape(-1)).reshape(len(words), n).astype(np.uint8)
        ok = st >= 0
        words = words.copy()
        words[ok] = fixed[ok] # A failed word is left as received
        return words, st

    def decode(self, bits, soft=None):
        x = np.asarray(bits)
        if soft or (soft is None and x.dtype.kind == "f"):
            x = x < 0
        x = x.astype(np.uint8)
        out, errs = [], []
        for w in x[:len(x)//self.den*self.den].reshape(-1, self.den):
            M = w[self.lead:].reshape(self.n2, self.n1).copy()
            first = M.copy()
            for _ in range(self.iters):
                M_cols, st_c = self._fix(self.cols, M.T.copy(), self.n2, self.k2)
                M = M_cols.T
                M, st_r = self._fix(self.rows, M, self.n1, self.k1)
                if (st_c == 0).all() and (st_r == 0).all():
                    break
            good = (st_c >= 0).all() and (st_r >= 0).all() and self._clean(M)
            errs.append(int(np.count_nonzero(M != first)) if good else -1)
            out.append(M[:self.k2, :self.k1].reshape(-1)[self.reserved:])
        self.errors = np.array(errs, dtype=int)
        return np.concatenate(out).astype(bool) if out else np.zeros(0, bool)

    def _clean(self, M):
        """Every row and column is a codeword"""
        r = self.rows._enc.encode(M[:, :self.k1].reshape(-1)).reshape(self.n2, self.n1)
        c = self.cols._enc.encode(M[:self.k2].T.reshape(-1)).reshape(self.n1, self.n2).T
        return np.array_equal(r.astype(np.uint8), M) and np.array_equal(c.astype(np.uint8), M)

def _code_obj(ref):
    """A block code from a reference ("hamming/dmr_15_11") or a definition, decoded by syndrome table"""
    from .. import serial
    if isinstance(ref, str):
        return serial.load("block", ref, decoder="syndrome")
    return serial.init("product_part", {"base": "block", **ref}, "syndrome")
