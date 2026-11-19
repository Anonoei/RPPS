"""Interleavers found blind: a block interleaver's size and shape, a convolutional one's branches and delay, and columns
of a grid sent out of step

    r = rp.analysis.code.interleave.block_search(bits)                     # size N (rank), then rows x columns
    r = rp.analysis.code.interleave.forney_search(bits, [(12, 17, 8), ...]) # which (I, M, symbol) gives code structure back
    d = rp.analysis.code.interleave.column_delays(bits, rows, cols)         # each column's shift in rows (a product code's
                                                                        #  parity column sent a row late)

A block interleaver keeps whole codewords inside each block of N bits, so the stream's rank deficiency first shows at
L = N (Sicot, Houcke & Barbier 2009) as it would at a code's length: rpps.analysis.code.block.linear.find_length finds N. Read back in
the order written (each factorization N = rows x columns, either way), the code's own length shows again, short.

A convolutional (Forney) interleaver has no block: its branches delay by multiples of M symbols. Each candidate (from the
standards, or a grid) is undone, and the stream is scored by the code structure that comes back: by default the strongest
linear recurrence (rpps.analysis.sequence.recurrence), which a convolutional code's parity check is.

column_delays: bits are rows of `cols` bits, `rows` to a frame, and each column of a frame is a codeword of a column code
(a product code). A column sent d rows late is not; its checks hold again when it is read d rows back. The column code's
checks are found from all the columns (most are right), then each column's shift is the one they hold best at.
"""
import numpy as np

from .block.linear import find_length, sparse_checks
from ..sequence import recurrence
from ...code.types.interleave import rowcol

def recurrence_z(bits, max_degree=14, n=200000):
    """How much linear code structure bits have: the z of their strongest recurrence of degree up to max_degree (0: none)"""
    found = recurrence.search(np.asarray(bits, dtype=np.uint8)[:n], max_degree)
    return found[0].z if found else 0.0

class Shape:
    """A block interleaver found: size (bits), offset (first block), rows, cols (written by rows, read by columns: the
    order the bits came in was cols x rows), code (the code length seen once read back), deficiency"""
    __slots__ = ("size", "offset", "rows", "cols", "code", "deficiency")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        if self.rows is None:
            return f"blocks of {self.size} bits from bit {self.offset} (no row x column shape found)"
        return (f"blocks of {self.size} bits from bit {self.offset}: written as {self.rows} rows of {self.cols}, read by "
                f"columns; read back, a code of length {self.code} (deficiency {self.deficiency})")

    __repr__ = __str__

def block_search(bits, max_size=None, max_code=None):
    """A block interleaver's size from the rank deficiency, then the row x column shape whose undoing shows a short code"""
    b = np.asarray(bits, dtype=np.uint8)
    N, off, _ = find_length(b, max_len=max_size)
    if N is None:
        return None
    b = b[off:]
    blocks = b[:len(b)//N*N].reshape(-1, N)
    best = None
    for rows in range(2, N//2 + 1):
        if N % rows:
            continue
        cols = N//rows
        perm = rowcol(N, cols) # out[i] = in[perm[i]]: the order the interleaver read
        inv = np.argsort(perm)
        back = blocks[:, inv].reshape(-1)
        n, o, defs = find_length(back, max_len=max_code or min(N - 1, 512))
        if n is not None and n < N and (best is None or n < best.code):
            best = Shape(size=N, offset=off, rows=rows, cols=cols, code=n, deficiency=defs.get(n))
    return best or Shape(size=N, offset=off)

# (I, M, symbol bits) of standard convolutional interleavers
FORNEY = {"DVB-S/C/T (EN 300 421)": (12, 17, 8), "ATSC A/53": (52, 4, 8), "ITU J.83 B (128, 1)": (128, 1, 7),
          "ITU J.83 B (64, 2)": (64, 2, 7), "ITU J.83 B (32, 4)": (32, 4, 7), "ITU J.83 B (16, 8)": (16, 8, 7),
          "ITU J.83 B (8, 16)": (8, 16, 7), "IESS-308 (8, 4)": (8, 4, 8), "IESS-308 (4, 8)": (4, 8, 8)}

def forney_undo(bits, I, M, symbol=8):
    """A Forney deinterleaver: symbols of `symbol` bits, branch j (of I, the first symbol on branch 0) delayed by
    (I - 1 - j) M symbols (rpps.code.types.interleave.forney's decode, without a code object)"""
    b = np.asarray(bits, dtype=np.uint8)
    b = b[:len(b)//(I*symbol)*(I*symbol)]
    sym = b.reshape(-1, symbol)
    out = np.zeros_like(sym)
    for j in range(I):
        x = sym[j::I]
        d = (I - 1 - j)*M
        out[j::I] = np.concatenate((np.zeros((d, symbol), np.uint8), x))[:len(x)]
    return out.reshape(-1)

def forney_search(bits, candidates=None, score=recurrence_z, n=200000):
    """Undo each (I, M, symbol) candidate (default FORNEY) at each start (a branch and a bit within the symbol) and score
    the result. Returns [(score, (I, M, symbol), start bit)], best first"""
    b = np.asarray(bits, dtype=np.uint8)[:n]
    cands = list(FORNEY.values()) if candidates is None else list(candidates)
    out = []
    for I, M, sym in cands:
        delay = (I - 1)*I*M*sym # start-up bits of the deinterleaver
        for start in range(I*sym):
            x = b[start:]
            x = x[:len(x)//(I*sym)*(I*sym)]
            if len(x) <= delay + 4096:
                continue
            y = forney_undo(x, I, M, sym)[delay:]
            out.append((float(score(y)), (I, M, sym), start))
    return sorted(out, key=lambda t: -t[0])

def column_delays(bits, rows, cols, max_delay=2, offset=0, **kw):
    """Each column's shift in rows (positive: sent that many rows late) that makes it a codeword of the column code again.
    bits: frames of rows x cols bits, row by row, from `offset`. kw: rpps.analysis.code.block.linear.sparse_checks's.
    Returns (delays (cols,), fraction of frames the column's checks hold on at that delay (cols,), checks H)"""
    b = np.asarray(bits, dtype=np.uint8)[offset:]
    F = len(b)//(rows*cols)
    R = b[:F*rows*cols].reshape(F*rows, cols) # every row of every frame, in order
    frames = R.reshape(F, rows, cols)
    vecs = frames.transpose(0, 2, 1).reshape(-1, rows) # every column of every frame
    kw.setdefault("w_max", rows) # a short column code's checks can be heavy (Hamming (15, 11): all weight 8)
    H, _ = sparse_checks(vecs, **kw)
    if not len(H):
        raise ValueError("no checks over the columns: not a column code at this rows x cols")
    delays = np.zeros(cols, dtype=int)
    held = np.zeros(cols)
    fs = np.arange(1, F - 1) # frames whose rows +- max_delay are all held
    if len(fs) < 4:
        raise ValueError("too few frames")
    for c in range(cols):
        best = (-1.0, 0)
        for d in range(-max_delay, max_delay + 1):
            # sent d rows late: frame f's row r arrived as row f rows + r + d of the stream
            idx = fs[:, None]*rows + np.arange(rows)[None, :] + d
            vv = R[idx, c]
            h = 1.0 - ((vv.astype(np.int32) @ H.T.astype(np.int32)) % 2).mean()
            if h > best[0] + 1e-9:
                best = (h, d)
        held[c], delays[c] = best
    return delays, held, H
