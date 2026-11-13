"""Codes from their parameters, without a definition file: `rp.code.Hamming(7, 4)`, `rp.code.RS(255, 223)`...

Each returns the same object `rp.code.load` makes from a definition, so they work the same in a pipeline:
    link = scr * rp.code.Hamming(7, 4) * rp.symbol.QAM(16)
"""
import numpy as np

from . import serial

def _make(name, obj, decoder=None):
    obj = {"base": "block", **obj}
    return serial.init(name, obj, decoder)

def Hamming(n=7, k=4, decoder=None):
    """Hamming code: n = 2**r - 1, k = n - r. A shorter k (n a bit smaller than 2**r - 1) gives a shortened code"""
    r = n - k
    if r < 2 or n > 2**r - 1 or n < k + r:
        raise ValueError(f"Hamming({n}, {k}) isn't a Hamming code or a shortened one: need n <= 2**(n-k) - 1")
    cols = [v for v in range(1, 2**r) if bin(v).count("1") >= 2][:k] # Parity checks of the data bits: not a unit vector
    p = [[(v >> (r - 1 - b)) & 1 for b in range(r)] for v in cols]
    return _make(f"hamming{n}_{k}", {"type": "linear", "code": [n, k], "p": p}, decoder)

def Repetition(count=3, decoder=None):
    """Each bit sent `count` times, decoded by majority (or by summing LLRs)"""
    return _make(f"repeat{count}", {"type": "repeat", "count": count}, decoder)

def RS(n=255, k=223, m=8, poly=None, fcr=0, prim=1, puncture=None, decoder=None):
    """Reed-Solomon over GF(2**m). n < 2**m - 1 is a shortened code. poly: the field's primitive polynomial (default for m=8: 285)
    puncture: codeword positions not sent (negative from the end), decoded as erasures"""
    poly = {3: 11, 4: 19, 5: 37, 6: 67, 7: 137, 8: 285, 9: 529, 10: 1033}[m] if poly is None else poly
    obj = {"type": "rs", "n": n, "k": k, "m": m, "poly": poly, "fcr": fcr, "prim": prim}
    if puncture:
        obj["puncture"] = list(puncture)
    return _make(f"rs{n}_{k}", obj, decoder)

def Cyclic(n, k, poly, extend=False, shorten=0, t=None, decoder=None):
    """Binary cyclic code from its generator polynomial (bit i = coefficient of x^i), message first

        Cyclic(23, 12, 0xC75)                 Golay
        Cyclic(23, 12, 0xC75, extend=True)    extended Golay (24,12)
        Cyclic(15, 11, 0b10011)               cyclic Hamming
    Decoded by syndrome table up to t errors (default: from the minimum distance); decoder="trellis" for soft MAP
    """
    obj = {"type": "cyclic", "n": n, "k": k, "poly": poly, "extend": extend, "shorten": shorten}
    if t is not None:
        obj["t"] = t
    k_out, n_out = k - shorten, n + int(extend) - shorten
    return _make(f"cyclic{n_out}_{k_out}", obj, decoder)

def Golay(n=23, decoder=None):
    """Golay (23,12) or extended Golay (24,12), g(x) = 0xC75"""
    if n not in (23, 24):
        raise ValueError(f"Golay({n}): 23 or 24")
    return _make(f"golay{n}_12", {"type": "cyclic", "n": 23, "k": 12, "poly": 0xC75, "extend": n == 24}, decoder)

def Conv(generators, constraint=None, puncture=None, terminated=False, decoder="viterbi", termination=None):
    """Rate 1/n convolutional code. generators: n taps lists (or octal/decimal ints) over the constraint length

        Conv([0o133, 0o171], 7)                          K=7 rate 1/2 (the IESS-308 generators)
        Conv([[1,0,1],[1,1,1]], puncture=[[1,1,0],[1,0,1]])
    puncture: pattern with one row per output and 1 where a bit is sent
    decoder: "viterbi" or "bcjr"
    termination: "zero" (terminated=True), "tailbiting" or "truncated"
    """
    if all(isinstance(g, (int, np.integer)) for g in generators):
        if constraint is None:
            raise ValueError("constraint is needed when generators are numbers")
        generators = [[(g >> (constraint - 1 - i)) & 1 for i in range(constraint)] for g in generators]
    generators = [list(map(int, g)) for g in generators]
    constraint = len(generators[0])
    side = {"num": 1, "den": len(generators), "constraint": constraint, "generator": generators,
            "termination": termination or ("zero" if terminated else "truncated")}
    enc = {"type": "conv", **side}
    if puncture is not None:
        enc["puncture"] = [list(map(int, row)) for row in puncture]
    return serial.init(f"conv{len(generators)}_{constraint}", {"base": "conv", "type": "split",
                       "encode": enc, "decode": {"type": decoder, **side}})
