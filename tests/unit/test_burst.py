"""TDMA bursts demodulated on their shared preamble (rpps.analysis.spectrum.burst)"""
import numpy as np

import rpps as rp
from rpps.analysis.spectrum import burst
from rpps.analysis.framing import frames as FR

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

def test_burst_kinds_each_on_its_own_preamble():
    """A network's burst types (here two lengths), each with its own unique word: what all the bursts share is a symbol or
    two, so each kind's preamble is found from its own bursts and every burst acquired on its kind's"""
    rng = np.random.default_rng(3)
    q = rp.symbol.load("qpsk").points
    fs, baud = 2e6, 250e3
    pres = [q[rng.integers(0, 4, 40)], q[rng.integers(0, 4, 40)]]
    x = np.zeros(int(0.03*fs), complex)
    sent, at = [], 2000
    for i in range(12):
        kind = i % 2
        syms = np.concatenate((pres[kind], q[rng.integers(0, 4, 200 if kind == 0 else 450)]))
        w = FR.word_waveform(syms, fs/baud)
        w = w*np.exp(1j*rng.uniform(0, 2*np.pi) + 2j*np.pi*rng.uniform(-2e3, 2e3)/fs*np.arange(len(w)))
        x[at:at + len(w)] += w
        sent.append((kind, syms))
        at += len(w) + 600
    x += 0.06*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    r = burst.demodulate(x, fs)
    assert len(r.preambles) == 2 and all(len(p) >= 32 for p in r.preambles)
    assert len(r.bursts) == 12 and all(b.ok for b in r.bursts)
    kinds = [b.kind for b in r.bursts]
    assert len(set(kinds[0::2])) == 1 and len(set(kinds[1::2])) == 1 and kinds[0] != kinds[1] # Each type its own kind
    assert np.median([b.evm for b in r.bursts]) < 0.2

def test_burst_kinds_when_the_gate_misses_by_tens_of_symbols():
    """A long burst's gate can open tens of symbols from its start: the preamble is still found (a wider alignment)"""
    from types import SimpleNamespace
    rng = np.random.default_rng(9)
    pts = np.exp(1j*np.pi/4*np.array([1, 3, 5, 7]))
    uw = pts[rng.integers(0, 4, 64)]
    B = []
    for i in range(8):
        lead = rng.integers(0, 25) # symbols of noise before the preamble: the gate opened that early
        s = np.concatenate((0.3*(rng.standard_normal(lead) + 1j*rng.standard_normal(lead)), uw, pts[rng.integers(0, 4, 4000)]))
        s = s*np.exp(1j*rng.uniform(0, 2*np.pi)) # each burst its own phase
        B.append(SimpleNamespace(symbols=s, start=0, stop=len(s)))
    kinds, member = burst.burst_kinds(B, 1.0, pts)
    assert len(kinds) == 1 and len(kinds[0]) >= 60 and member == [0]*8
