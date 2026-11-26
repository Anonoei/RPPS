"""Bit manipulation; on hard bits (one a byte): the Hamming distance to a word at every position (`distance`, `find`)
and bit destuffing over many spans (`unstuff`), in C++ when rpps._native is built"""
import numpy as np

def entropy(bits):
    """Entropy of a bit stream [bits per bit]: H = -p0 log2 p0 - p1 log2 p1 (0 log 0 = 0)"""
    bits = np.asarray(bits)
    p = np.array([np.mean(bits == 0), np.mean(bits == 1)])
    p = p[p > 0]
    return float(-np.sum(p*np.log2(p)))

def reflect(v, width):
    """Reverse the order of the low `width` bits of an integer"""
    return int(format(v, f"0{width}b")[::-1], 2)

def to_base2(order):
    """Every `order`-bit pattern in counting order: (2^order, order) array of 0/1, most significant bit first"""
    return ((np.arange(2**order)[:, None] >> np.arange(order - 1, -1, -1)) & 1)

def to_base10(arr):
    """Rows of bits (most significant first) -> integers"""
    arr = np.asarray(arr)
    return np.dot(arr, 2**np.arange(arr.shape[1] - 1, -1, -1))

def hamming(a, b):
    """Hamming distance: the number of positions where bit arrays a and b differ"""
    return int(np.count_nonzero(np.asarray(a, dtype=bool) != np.asarray(b, dtype=bool)))

# ---- hard bits (one a byte, 0 or 1): the C++ kernels (rpps._native) when built, these loops their reference

def distance(bits, word):
    """The Hamming distance to `word` at every position of `bits` (len(bits) - len(word) + 1 of them, int): 0 where the
    word starts, len(word) where its inverse does. In C++ for words of up to 64 bits (a shift register and a popcount a
    bit); else, and as the reference, a window by window count"""
    from .. import _accel
    b = np.ascontiguousarray(bits, dtype=np.uint8)
    w = np.ascontiguousarray(word, dtype=np.uint8)
    if len(w) < 1 or len(b) < len(w):
        return np.zeros(0, np.int64)
    if _accel.native is not None and len(w) <= 64:
        return _accel.native.bit_distance(b & 1, w & 1).astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(b & 1, len(w))
    return np.count_nonzero(win != (w & 1)[None, :], axis=1).astype(np.int64)

def find(bits, word, errors=0):
    """Where `word` starts in `bits` with at most `errors` bits wrong"""
    return np.nonzero(distance(bits, word) <= errors)[0]

def _units(bits, unit):
    """The value of the `unit` bits (most significant first) starting at each position of `bits`"""
    b = np.asarray(bits).astype(np.int64) & 1
    if len(b) < unit:
        return np.zeros(0, np.int64)
    return np.lib.stride_tricks.sliding_window_view(b, unit) @ (1 << np.arange(unit - 1, -1, -1))

def periodic_scores(bits, span, unit, allowed, periods=None, polarity=True):
    """Periodic marker fit.

    The fraction of periods that start with an `allowed` unit, at every (polarity, bit offset, pattern position).
    span: bits a period; unit: bits a marker unit; allowed: [pattern position, unit value] booleans. Returns
    (2 or 1, span, len(allowed)); its first maximum is what rpps.sync.Align locks on"""
    allowed = np.asarray(allowed, dtype=bool)
    L = len(allowed)
    b = np.asarray(bits).astype(bool)
    n = (len(b) - span)//span if periods is None else int(periods)
    out = np.zeros((2 if polarity else 1, span, L))
    if n <= 0:
        return out
    v = _units(b[:n*span + unit - 1] if span >= unit else b, unit)
    heads = np.empty((span, n), np.int64)
    k = np.arange(n)
    heads[:] = v[np.arange(span)[:, None] + k[None, :]*span] # [offset, period]: the unit starting each period
    pos = (np.arange(L)[:, None] + k[None, :]) % L # [first position, period]: the pattern position of each period
    top = (1 << unit) - 1
    for f, h in enumerate((heads, top - heads) if polarity else (heads,)): # (inverted bits: every unit's complement)
        out[f] = allowed[pos[None, :, :], h[:, None, :]].mean(axis=2)
    return out

def unstuff(bits, starts, ends, stuff=5):
    """Bit destuffing (HDLC's: after `stuff` ones a 0 is dropped, `stuff` + 1 ones abort) of the spans [starts[i], ends[i])
    of `bits`, each span's bits then packed into bytes least significant bit first. Returns (status per span: 0 whole
    bytes, 1 aborted, 2 not a whole number of bytes; the bytes of each span, b"" where the status is not 0)"""
    from .. import _accel
    b = np.ascontiguousarray(bits, dtype=np.uint8)
    s = np.ascontiguousarray(starts, dtype=np.int64)
    e = np.ascontiguousarray(ends, dtype=np.int64)
    if _accel.native is not None:
        status, off, raw = _accel.native.bit_unstuff(b, s, e, int(stuff))
        raw = raw.tobytes()
        return status.astype(np.int64), [raw[off[i]:off[i + 1]] for i in range(len(s))]
    status, out = np.zeros(len(s), np.int64), []
    for i, (a, z) in enumerate(zip(s, e)):
        bits_out, ones, bad = [], 0, 0
        for bit in b[max(int(a), 0):min(int(z), len(b))] & 1:
            if ones == stuff:
                ones = 0
                if bit == 0:
                    continue # the stuffed zero
                bad = 1 # more than `stuff` ones inside a span: an abort
                break
            bits_out.append(bit)
            ones = ones + 1 if bit else 0
        if not bad and len(bits_out) % 8:
            bad = 2
        status[i] = bad
        out.append(np.packbits(np.array(bits_out, dtype=np.uint8), bitorder="little").tobytes() if not bad else b"")
    return status, out
