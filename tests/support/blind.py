"""Bits and codewords the blind tools are tested on"""
import numpy as np

import rpps as rp
from rpps.code.types.rs import rs
from rpps.logger import Logger

def _feedthrough(d, taps):
    """y[n] = d[n] ^ y[n-a] ^ y[n-b]...: the scrambler rpps.code.scram.Feedthrough.descramble undoes"""
    y = np.zeros(len(d), np.uint8)
    for n in range(len(d)):
        v = d[n]
        for t in taps:
            if n >= t:
                v ^= y[n - t]
        y[n] = v
    return y

def scramble(d, taps):
    """y[n] = d[n] ^ y[n-t] for each tap (a feedthrough scrambler, register starting at zero)"""
    y = np.zeros(len(d), np.uint8)
    for n in range(len(d)):
        v = int(d[n])
        for t in taps:
            if n >= t:
                v ^= int(y[n - t])
        y[n] = v
    return y

def rs_bits(n, k, poly, fcr, prim, words=24, payload=None, seed=0, phase=37, offset=5):
    rng = np.random.default_rng(seed)
    code = rs(Logger().Child("t", 0), k, n, 8, poly, fcr, prim)
    msg = rng.integers(0, 256, (words, k)) if payload is None else np.packbits(payload[:words*k*8]).reshape(words, k)
    cw = np.asarray(code.encode(np.asarray(msg).reshape(-1), symbols=True)).astype(np.uint8)[phase:]
    return np.concatenate((rng.integers(0, 2, offset).astype(np.uint8), np.unpackbits(cw)))

def _qpsk(bits, rng, sigma=0.1):
    q = rp.symbol.load("qpsk")
    s = q.modulate(np.asarray(bits[:len(bits)//2*2], np.uint8))*1j
    return (s + sigma*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s)))).astype(np.complex64)
