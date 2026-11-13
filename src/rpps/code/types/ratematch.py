import numpy as np

from ._code import _code, register
from .interleave import rowcol

CONV_PERM = [1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31, 0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30]
TURBO_PERM = [0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31]

@register("ratematch")
class ratematch(_code):
    """LTE rate matching (3GPP TS 36.212 5.1.4): three coded streams -> E bits through a circular buffer

        mode "conv" (5.1.4.2): input as a convolutional encoder sends it (three bits per step); each stream sub-block
            interleaved (32 columns, CONV_PERM), the buffer is the three one after the other, read from k0 = 0
        mode "turbo" (5.1.4.1): input as the turbo code sends it (d0, d1, d2 one after the other); d0 and d1 sub-block
            interleaved with TURBO_PERM, d2 with its own permutation, buffer = v0 then v1 / v2 bit by bit, read from
            k0 = R (2 ceil(Ncb / 8R) rv + 2); ncb: soft buffer size (default the whole buffer)
        length: bits per stream (D: K for conv, K + 4 for turbo); out: E, bits sent
    The <NULL> filler is skipped; reading wraps round the buffer (repetition) or stops early (puncturing).
    encode() selects the E bits. decode() takes E LLRs and returns 3 D LLRs in the input layout, repetitions added
    (positions never sent stay 0: erasures), ready for the decoder.
    """
    soft = True

    @classmethod
    def from_config(cls, log, obj):
        return cls(log, obj["mode"], obj["length"], obj["out"], obj.get("rv", 0), obj.get("ncb"))

    def __init__(self, log, mode, length, out, rv=0, ncb=None):
        D, E = int(length), int(out)
        super().__init__(log, None, 3*D, E)
        self.mode, self.D, self.E, self.rv = mode, D, E, rv
        if mode == "conv":
            p = rowcol(D, 32, CONV_PERM)
            buf = [(i, int(k)) for i in range(3) for k in p]
            idx = [buf[t % len(buf)] for t in range(E)]
            self.index = np.array([k*3 + i for i, k in idx], dtype=np.int64) # Steps of three bits
        elif mode == "turbo":
            C = 32
            R = -(-D//C)
            Kp = R*C
            y = np.full(Kp, -1)
            y[Kp - D:] = np.arange(D)
            v01 = y.reshape(R, C)[:, TURBO_PERM].T.reshape(-1)
            v2 = y[np.array([(TURBO_PERM[k//R] + C*(k % R) + 1) % Kp for k in range(Kp)])]
            w = [(0, int(i)) for i in v01] + [t for k in range(Kp) for t in ((1, int(v01[k])), (2, int(v2[k])))]
            Ncb = len(w) if ncb is None else int(ncb)
            k0 = R*(2*(-(-Ncb//(8*R)))*rv + 2)
            sel, j = [], 0
            while len(sel) < E:
                s, k = w[(k0 + j) % Ncb]
                if k >= 0:
                    sel.append(s*D + k)
                j += 1
                if j > 4*Ncb + 4*E:
                    raise ValueError("Nothing to send in the circular buffer")
            self.index = np.array(sel, dtype=np.int64)
        else:
            raise ValueError(f"mode {mode!r}: 'conv' or 'turbo'")

    def encode(self, bits):
        x = np.asarray(bits)
        n = 3*self.D
        return np.concatenate([w[self.index] for w in x[:len(x)//n*n].reshape(-1, n)]) if len(x) >= n else x[:0]

    def decode(self, values, soft=None):
        x = np.asarray(values)
        if soft is None:
            soft = x.dtype.kind == "f"
        llr = x.astype(np.float64) if soft else 1.0 - 2.0*x.astype(np.float64)
        out = []
        for w in llr[:len(llr)//self.E*self.E].reshape(-1, self.E):
            acc = np.zeros(3*self.D)
            np.add.at(acc, self.index, w)
            out.append(acc)
        return np.concatenate(out) if out else np.zeros(0)
