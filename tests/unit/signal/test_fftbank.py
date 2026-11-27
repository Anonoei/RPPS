"""rp.signal.FFTBank: many channels of a capture from one FFT a hop"""
import numpy as np
import pytest

import rpps as rp

FS = 100e6

def tones(n, freqs, amps):
    t = np.arange(n)
    return sum(a*np.exp(2j*np.pi*f*t/FS) for f, a in zip(freqs, amps)).astype(np.complex64)

def run(bank, x, cuts, active=None):
    outs, i = [[] for _ in range(len(bank))], 0
    for c in cuts:
        for k, y in enumerate(bank.step(x[i:i + c], active=active)):
            outs[k].append(y)
        i += c
    for k, y in enumerate(bank.flush(active=active)):
        outs[k].append(y)
    return [np.concatenate(o) for o in outs]

@pytest.mark.parametrize("n_cut", [None, 777, 16384])
def test_each_channel_is_its_signal_at_baseband_whatever_the_cut(n_cut):
    """A tone at each channel's centre (off the FFT's bin grid) comes out as DC at its amplitude with no phase steps
    between hops; a tone 400 kHz from a 200 kHz channel is gone; one sample out per D in, however the input is cut"""
    n = 1_500_000
    mids = [12.3456e6, -31.0001e6]
    x = tones(n, [mids[0], mids[0] + 400e3, mids[1]], [1.0, 0.5, 0.25])
    bank = rp.signal.FFTBank(FS, [(mids[0], 200e3), (mids[1], 50e3)])
    cuts = [n] if n_cut is None else [n_cut]*(n//n_cut) + [n % n_cut]
    y = run(bank, x, cuts)
    for k, (amp, D) in enumerate(zip((1.0, 0.25), bank.D)):
        assert len(y[k]) == -(-n//D)
        mid = y[k][len(y[k])//8:-len(y[k])//8]
        assert abs(np.mean(mid) - amp) < 1e-3*amp and np.std(mid) < 1e-3*amp
    ref = run(rp.signal.FFTBank(FS, [(mids[0], 200e3), (mids[1], 50e3)]), x, [n])
    assert all(np.array_equal(a, b) for a, b in zip(y, ref))

def test_rates_follow_the_widths():
    bank = rp.signal.FFTBank(FS, [(0.0, 20e3), (1e6, 1e6), (-3e6, 5e6)])
    assert all(fs >= 4*w for fs, w in zip(bank.Fs_out, (20e3, 1e6, 5e6)))
    assert all(fs < 8*w for fs, w in zip(bank.Fs_out, (20e3, 1e6, 5e6))) # D the largest power of two that keeps 4x
    assert bank.n >= 16*FS/(0.2*20e3) or bank.n == 2**21 # The narrowest skirt spans 16 bins (capped)

def test_inactive_channels_cost_nothing_and_give_nothing():
    x = tones(400_000, [1e6, 2e6], [1.0, 1.0])
    y = run(rp.signal.FFTBank(FS, [(1e6, 100e3), (2e6, 100e3)]), x, [400_000], active=[1])
    assert len(y[0]) == 0 and len(y[1]) > 0

def test_a_qpsk_channel_matches_the_per_signal_front():
    """The same signal through FFTBank and through its own channel front (rp.signal.channelize + select filter)"""
    rng = np.random.default_rng(3)
    sym = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, 20000) + 1))
    w = (rp.Data(sym.astype(np.complex64), Fs=1e6, sps=1) + rp.wave.Shape(8, rp.signal.filters.load("pulse/rrc/0p35"))).data
    x = (np.asarray(w)*np.exp(2j*np.pi*1.7e6/8e6*np.arange(len(w)))).astype(np.complex64)
    sig = next(s for s in rp.analysis.survey(rp.source.Memory(x, Fs=8e6)) if abs(s.mid - 1.7e6) < 0.2e6)
    ch = sig.channel()
    old = np.asarray((rp.Data(x, Fs=8e6, flush=True) + ch.front).data)
    bank = rp.signal.FFTBank(8e6, [(sig.mid, sig.width)])
    new = run(bank, x, [len(x)])[0]
    assert bank.Fs_out[0] == ch.Fs
    d = int(round(ch.delay*ch.Fs))
    a, b = old[d + 2000:d + 100000], new[2000:100000]
    assert abs(np.vdot(a, b))/np.linalg.norm(a)/np.linalg.norm(b) > 0.99
