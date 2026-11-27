"""Frames and receptions the parallel tests share"""
import numpy as np

import rpps as rp
from rpps.parallel import Frame, Result

def frames(n, modcods=(1,)):
    return [Frame(i, np.full(8, i, dtype=np.float32), {"ldpc": {"name": modcods[i % len(modcods)]}}, pos=8*i, pos_out=4*i, n_out=4)
            for i in range(n)]

def halve(batch):
    """A worker: the first half of each frame (the 'decode')"""
    return [Result(f.seq, f.data[:4] * 2, f.pos_out, f.n_out) for f in batch]

FRAMES = [{"modcod": 4, "pilots": 1}, {"modcod": 12, "pilots": 1}, {"modcod": 1, "pilots": 0}, {"modcod": 4, "pilots": 1}]

def receive(rx, x, o, cuts=None, noise=0.0):
    rng = np.random.default_rng(5)
    x = (x*np.exp(0.7j) + noise*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)
    st, outs, i = None, [], 0
    cuts = cuts or [len(x)]
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=o.Fs, sps=4, delay=o.delay, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        outs.append(np.asarray(rx(d).data).astype(bool))
    return np.concatenate(outs), st
