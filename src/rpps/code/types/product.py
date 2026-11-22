import numpy as np

from ... import _effort
from ._code import _code, register

@register("product")
class product(_code):
    """Product code (block product turbo code, BPTC): message in a k2 x k1 matrix, every row encoded with the row
    code (n1, k1), then every column (parity columns too) with the column code (n2, k2); sent row by row

        rows, cols: the two block codes, as references to block definitions ("hamming/dmr_15_11") or definitions
        reserved: zero bits at the start of the message area (DMR: R(2..0)); the message is k1 k2 - reserved bits
        lead: zero bits sent before the matrix (DMR: R(3)), so n = lead + n1 n2
        offset: a fixed word XOR'd onto every codeword (a coset of the product code: parity bits sent inverted, fill), as
            a hex string of n bits, the lead first then the matrix row by row; decoding takes it off first
        iters: decoding rounds of all columns then all rows (hard decisions, each code's own syndrome decoder)
        chase: decode soft instead, as a turbo product code (Pyndiah, "Near-optimum decoding of product codes: block turbo
            codes", IEEE Trans. Commun. 1998): each row and column by Chase-II over its `chase` least reliable bits (2^chase
            test patterns through the component's algebraic decoder, the candidate nearest the input chosen). The channel
            LLRs are normalized to mean |R| = 1; the extrinsic information W of a bit is the metric difference to the best
            candidate that disagrees with the decision on it, minus the input, or beta times the decision when no candidate
            disagrees; W is normalized to mean 1 and the next half iteration decodes R + alpha W (ALPHA and BETA per half
            iteration). A row whose test patterns all fail keeps its hard decision. `iters` rounds, stopping early on
            a word whose rows and columns are all codewords
    After a decode `errors` is the bits corrected per word, -1 where a row or column still fails its check.
    """
    # Per half iteration, the last repeated. ALPHA is Pyndiah's ramp held at 0.5 (as AFF3CT): rising on to his 1.0 makes
    # words that fail diverge past 4 iterations (tests/tpc_ber.py). The first half iteration has no extrinsic yet.
    ALPHA = (0.0, 0.2, 0.3, 0.5) # Weight of the extrinsic in the next decoder input
    BETA = (0.2, 0.4, 0.6, 0.8, 1.0) # Extrinsic of a bit with no competing candidate (mean |W| is 1)
    BATCH = 32 # Words decoded together
    @classmethod
    def from_config(cls, log, obj):
        return cls(log, _code_obj(obj["rows"]), _code_obj(obj["cols"]), obj.get("reserved", 0), obj.get("lead", 0),
                   obj.get("iters", 4), obj.get("offset"), obj.get("chase"))

    def __init__(self, log, rows, cols, reserved=0, lead=0, iters=4, offset=None, chase=None):
        self.chase = None if chase is None else int(chase)
        self.soft = self.chase is not None
        self.rows, self.cols = rows, cols # Coding objects (encoder + hard decoder)
        self.k1, self.n1 = rows.encoder.num, rows.encoder.den
        self.k2, self.n2 = cols.encoder.num, cols.encoder.den
        self.reserved, self.lead, self.iters = int(reserved), int(lead), int(iters)
        super().__init__(log, None, self.k1*self.k2 - self.reserved, self.lead + self.n1*self.n2)
        self.offset = None if offset is None else _bits_of_hex(offset, self.den)

    def encode(self, bits):
        b = np.asarray(bits).astype(np.uint8)
        out = []
        for m in b[:len(b)//self.num*self.num].reshape(-1, self.num):
            area = np.concatenate((np.zeros(self.reserved, np.uint8), m)).reshape(self.k2, self.k1)
            rows = self.rows.encoder.encode(area.reshape(-1)).reshape(self.k2, self.n1).astype(np.uint8)
            full = self.cols.encoder.encode(rows.T.reshape(-1)).reshape(self.n1, self.n2).T # Columns encoded, back to rows
            word = np.concatenate((np.zeros(self.lead, np.uint8), full.reshape(-1)))
            out.append(word if self.offset is None else word ^ self.offset)
        return np.concatenate(out).astype(bool) if out else np.zeros(0, bool)

    def _fix(self, coder, words, n, k):
        """Decode each word (rows of `words`, n bits) with the code's hard decoder; returns words corrected and statuses"""
        dec = coder.decoder
        msg, st = dec.decode(words.reshape(-1), status=True)
        msg = msg.reshape(len(words), k).astype(np.uint8)
        fixed = coder.encoder.encode(msg.reshape(-1)).reshape(len(words), n).astype(np.uint8)
        ok = st >= 0
        words = words.copy()
        words[ok] = fixed[ok] # A failed word is left as received
        return words, st

    def _chase(self, coder, R, n, k):
        """Chase-II on the rows of R (rows, n), normalized LLRs: decisions D (rows, n), their signs sD (+1 for a 0), the
        metric of each decision mD (rows,) and of the best competitor per bit mC (rows, n; inf where none). A row whose
        test patterns all fail the algebraic decoder keeps its hard decision, with no competitor on any bit"""
        p = min(_effort.shift("chase", self.chase), n)
        rows = len(R)
        y = (R < 0).astype(np.uint8)
        weak = np.argsort(np.abs(R), axis=1)[:, :p] # (rows, p)
        pats = ((np.arange(1 << p)[:, None] >> np.arange(p)) & 1).astype(np.uint8) # (T, p)
        T = len(pats)
        cand = np.repeat(y[:, None, :], T, axis=1) # (rows, T, n)
        flip = np.zeros((rows, T, n), np.uint8)
        np.put_along_axis(flip, np.broadcast_to(weak[:, None, :], (rows, T, p)), np.broadcast_to(pats[None], (rows, T, p)), axis=2)
        cand ^= flip
        msg, st = coder.decoder.decode(cand.reshape(-1), status=True)
        msg = msg.reshape(rows*T, k).astype(np.uint8)
        ok = (np.asarray(st) >= 0).reshape(rows, T)
        cw = coder.encoder.encode(msg.reshape(-1)).reshape(rows, T, n).astype(np.uint8)
        metric = (np.abs(R)[:, None, :]*(cw != y[:, None, :])).sum(axis=2) # correlation discrepancy to the input
        metric = np.where(ok, metric, np.inf)
        best = np.argmin(metric, axis=1)
        idx = np.arange(rows)
        D = cw[idx, best] # (rows, n) decisions
        mD = metric[idx, best]
        failed = ~ok.any(axis=1)
        D[failed] = y[failed]
        sD = 1.0 - 2.0*D
        differs = cw != D[:, None, :] # (rows, T, n): candidate disagrees with the decision on the bit
        mC = np.where(differs, metric[:, :, None], np.inf).min(axis=1) # best competitor per bit
        mC[failed] = np.inf
        return D, sD, mD, mC

    def _soft(self, L):
        """Turbo decoding of words' LLRs (words, n2, n1): the decided matrices (words, n2, n1)"""
        L = np.asarray(L, dtype=np.float64)
        nw = len(L)
        R = L/np.maximum(np.abs(L).mean(axis=(1, 2), keepdims=True), 1e-30) # Channel input normalized: mean |R| = 1
        Wx = np.zeros_like(R)
        M = (L < 0).astype(np.uint8)
        act = np.arange(nw) # Words still decoding
        half = 0
        for _ in range(_effort.scale("iters", self.iters)):
            for axis in (0, 1): # columns, then rows
                a = self.ALPHA[min(half, len(self.ALPHA) - 1)]
                b = self.BETA[min(half, len(self.BETA) - 1)]
                inp = R[act] + a*Wx[act]
                if axis == 0:
                    D, sD, mD, mC = self._chase(self.cols, inp.transpose(0, 2, 1).reshape(-1, self.n2), self.n2, self.k2)
                    shape = (len(act), self.n1, self.n2)
                    D, sD, mC = (v.reshape(shape).transpose(0, 2, 1) for v in (D, sD, mC))
                    mD = mD.reshape(len(act), 1, self.n1)
                else:
                    D, sD, mD, mC = self._chase(self.rows, inp.reshape(-1, self.n1), self.n1, self.k1)
                    shape = (len(act), self.n2, self.n1)
                    D, sD, mC = (v.reshape(shape) for v in (D, sD, mC))
                    mD = mD.reshape(len(act), self.n2, 1)
                # Extrinsic (Pyndiah): from the best competitor, soft output minus input; without one, beta times the decision
                w = np.where(np.isfinite(mC), (mC - mD)*sD - inp, b*sD)
                Wx[act] = w/np.maximum(np.abs(w).mean(axis=(1, 2), keepdims=True), 1e-30) # Normalized: mean |W| = 1
                half += 1
            M[act] = D
            done = np.array([self._clean(m) for m in D], bool)
            act = act[~done]
            if len(act) == 0:
                break
        return M

    def _decode(self, bits, soft=None):
        x = np.asarray(bits)
        if self.chase is not None and x.dtype.kind == "f" and soft is not False:
            return self._decode_soft(x)
        if soft or (soft is None and x.dtype.kind == "f"):
            x = x < 0
        x = x.astype(np.uint8)
        out, errs = [], []
        for w in x[:len(x)//self.den*self.den].reshape(-1, self.den):
            if self.offset is not None:
                w = w ^ self.offset
            M = w[self.lead:].reshape(self.n2, self.n1).copy()
            first = M.copy()
            for _ in range(_effort.scale("iters", self.iters)):
                M_cols, st_c = self._fix(self.cols, M.T.copy(), self.n2, self.k2)
                M = M_cols.T
                M, st_r = self._fix(self.rows, M, self.n1, self.k1)
                if (st_c == 0).all() and (st_r == 0).all():
                    break
            good = (st_c >= 0).all() and (st_r >= 0).all() and self._clean(M)
            errs.append(int(np.count_nonzero(M != first)) if good else -1)
            out.append(M[:self.k2, :self.k1].reshape(-1)[self.reserved:])
        return (np.concatenate(out).astype(bool) if out else np.zeros(0, bool)), np.array(errs, dtype=int)

    def _decode_soft(self, x):
        x = np.asarray(x, dtype=np.float64)
        nw = len(x)//self.den
        w = x[:nw*self.den].reshape(nw, self.den)
        if self.offset is not None:
            w = np.where(self.offset == 1, -w, w)
        L = w[:, self.lead:].reshape(nw, self.n2, self.n1)
        out, errs = [], []
        for s0 in range(0, nw, self.BATCH):
            for Lw, M in zip(L[s0:s0 + self.BATCH], self._soft(L[s0:s0 + self.BATCH])):
                errs.append(int(np.count_nonzero(M != (Lw < 0))) if self._clean(M) else -1)
                out.append(M[:self.k2, :self.k1].reshape(-1)[self.reserved:])
        return (np.concatenate(out).astype(bool) if out else np.zeros(0, bool)), np.array(errs, dtype=int)

    def _clean(self, M):
        """Every row and column is a codeword"""
        r = self.rows.encoder.encode(M[:, :self.k1].reshape(-1)).reshape(self.n2, self.n1)
        c = self.cols.encoder.encode(M[:self.k2].T.reshape(-1)).reshape(self.n1, self.n2).T
        return np.array_equal(r.astype(np.uint8), M) and np.array_equal(c.astype(np.uint8), M)

def _bits_of_hex(text, n):
    """n bits from a hex string, most significant first (the string may hold a few padding bits at its end)"""
    raw = np.unpackbits(np.frombuffer(bytes.fromhex(text), dtype=np.uint8))
    if len(raw) < n:
        raise ValueError(f"offset: {len(raw)} bits for a word of {n}")
    return raw[:n].astype(np.uint8)

def _code_obj(ref):
    """A block code from a reference ("hamming/dmr_15_11") or a definition, decoded by syndrome table"""
    from .. import serial
    if isinstance(ref, str):
        return serial.load(f"block/{ref}", decoder="syndrome")
    return serial.init("product_part", {"base": "block", **ref}, "syndrome")
