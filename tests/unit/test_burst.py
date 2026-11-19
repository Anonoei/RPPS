"""TDMA bursts demodulated on their shared preamble (rpps.analysis.burst)"""
import numpy as np

import rpps as rp
from rpps.analysis import burst, framing as FR

def tdma(rng, frames=3, slots=(0, 2, 3, 5), fs=2e6, baud=250e3, frame=4.8e-3, slot=0.6e-3, noise=0.08):
    q = rp.symbol.load("qpsk").points
    pre = q[rng.integers(0, 4, 26)]
    x = np.zeros(int((frames + 0.2)*frame*fs), complex)
    sent = []
    for f in range(frames):
        for s in slots:
            syms = np.concatenate((pre, q[rng.integers(0, 4, 104)]))
            w = FR.word_waveform(syms, fs/baud)
            w = w*np.exp(1j*rng.uniform(0, 2*np.pi) + 2j*np.pi*rng.uniform(-3e3, 3e3)/fs*np.arange(len(w)))
            a = int((f*frame + s*slot)*fs) + int(rng.integers(0, 6))
            x[a:a + len(w)] += w
            sent.append(syms)
    x += noise*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    return x, fs, baud, sent, q

def test_every_burst_on_its_own_carrier_and_phase():
    x, fs, baud, sent, q = tdma(np.random.default_rng(7))
    r = burst.demodulate(x, fs)
    assert r.mod == "qpsk" and abs(r.baud - baud) < 200 and len(r.preamble) >= 24
    assert abs(r.frame - 4.8e-3) < 2e-5 and len(r.slots) == 4
    assert [b.slot for b in r.bursts] == [0, 1, 2, 3]*3
    rot = set()
    for b, syms in zip(r.bursts, sent):
        assert b.ok
        dec = q[np.argmin(np.abs(b.symbols[:, None] - q[None, :]), axis=1)]
        # the found preamble starts a symbol in (its first step); one rotation for every burst
        n = min(len(dec), len(syms) - 1)
        errs = [np.mean(np.abs(dec[:n] - syms[1:n + 1]*1j**k) > 1e-6) for k in range(4)]
        assert min(errs) == 0
        rot.add(int(np.argmin(errs)))
    assert len(rot) == 1
