"""DVB-S2 frames and receptions more than one test file uses"""
import numpy as np

import rpps as rp
from rpps.math.interp import interpolate

KNOWN = {"timing": {"method": "known"}}
PILOT_FRAMES = [{"modcod": (4, 12)[i % 2], "pilots": 1} for i in range(4)]
MIXED_FRAMES = [{"modcod": 4, "pilots": 1}, {"modcod": 12, "pilots": 1}, {"modcod": 1, "pilots": 0}]

def seeded_bits(n, seed):
    return np.random.default_rng(seed).integers(0, 2, n).astype(bool)

def pilot_capture():
    """DVB-S2 frames after a quiet start, received with a clock 20 ppm fast, a phase offset and noise: (samples, sent
    signal, data bits)"""
    p = rp.profile.load("dvbs2/normal")
    t = rp.profile.table("dvbs2/modcod")
    dfl = [t.row({**f, "short": 0})["kbch"] - 80 for f in PILOT_FRAMES]
    data = np.random.default_rng(1).integers(0, 2, sum(dfl)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=PILOT_FRAMES)
    x = np.concatenate((np.zeros(3000, np.complex64), o.data, np.zeros(400, np.complex64)))
    x = interpolate(x, np.arange(int(len(x)/(1 + 20e-6)) - 20)*(1 + 20e-6))
    rng = np.random.default_rng(2)
    x = (x*np.exp(0.7j) + 0.05*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)
    return x, o, data

def receive(x, o, sync, cut=None, parallel=None):
    rx = rp.profile.load("dvbs2/normal", sync=sync).rx(parallel=parallel)
    st, outs = None, []
    cut = cut or len(x)
    for a in range(0, len(x), cut):
        d = rp.Data(x[a:a + cut], Fs=o.Fs, sps=4, delay=o.delay, stream=st, flush=a + cut >= len(x))
        st = d.stream
        outs.append(np.asarray(rx(d).data).astype(bool))
    if parallel is not None:
        rx.close()
    return np.concatenate(outs), st
