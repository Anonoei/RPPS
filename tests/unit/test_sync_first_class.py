"""First-class sync estimators (README.dev.md, Sync methods)

Block estimation per window at absolute stream positions, tracked across windows: timing (Oerder & Meyr), carrier
(M-th power), equalization (block least squares). Each follows a drifting clock or carrier, gives the same output
whatever the block cut, and coasts through what is not signal. Then the profiles from raw samples with only their own
steps.
"""
import numpy as np
import pytest

import rpps as rp

ch = rp.signal.channel
RRC = "rrc/0p35"

def qpsk_wave(n, seed=0, ppm=0.0, delay=1.3, cfo=0.0, noise=0.05):
    """QPSK at 4 samples per symbol through RRC, a clock `ppm` off, a delay, a carrier offset [cycles/sample] and noise"""
    rng = np.random.default_rng(seed)
    sym = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, n) + 1))
    up = np.zeros(4*n, np.complex64)
    up[::4] = sym
    tx = np.asarray((rp.Data(up, Fs=4e6, sps=4) + rp.signal.filters.load("pulse", RRC)).data)
    t = np.arange(len(tx))*(1 + ppm*1e-6) + delay
    tx = np.interp(t, np.arange(len(tx)), tx.real) + 1j*np.interp(t, np.arange(len(tx)), tx.imag)
    x = tx*np.exp(2j*np.pi*cfo*np.arange(len(tx)) + 0.4j)
    return (x + noise*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)

def run(x, make, cuts=None, Fs=4e6, sps=4):
    """Stages made by `make()`, on x cut into blocks"""
    stages = make()
    cuts = cuts or [len(x)]
    s, outs, i = None, [], 0
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=Fs, sps=sps, stream=s, flush=k == len(cuts) - 1)
        s, i = d.stream, i + n
        for st in stages:
            d = d + st
        outs.append(np.asarray(d.data))
    return np.concatenate(outs), s

def evm_qpsk(y):
    y = y/np.mean(np.abs(y))
    ref = np.exp(1j*np.pi/4*(2*np.round((np.angle(y)/(np.pi/4) - 1)/2) + 1))
    return float(np.sqrt(np.mean(np.abs(y - ref)**2)))

def timing_and_carrier():
    return [rp.signal.filters.load("pulse", RRC), rp.sync.time.OerderMeyr(window=0),
            rp.sync.phase.MPower(4, offset=np.pi, window=0)]

@pytest.mark.parametrize("ppm", [0, 50, -200, 200])
def test_timing_tracks_the_clock(ppm):
    x = qpsk_wave(40000, ppm=ppm, cfo=3e-3)
    y, s = run(x, timing_and_carrier)
    est = s.est["timing"]
    assert abs(est["ppm"] - ppm) < 10 + 0.1*abs(ppm)
    assert evm_qpsk(y[6000:]) < 0.2 # noise alone gives about 0.15 here
    assert abs(len(y) - 40000*(1 + ppm*1e-6)) < 3 # one symbol out per symbol in: no slips

def test_carrier_frequency():
    x = qpsk_wave(30000, cfo=3e-3)
    y, s = run(x, timing_and_carrier)
    assert abs(s.est["carrier"]["frequency"] - 4*3e-3) < 1e-5 # cycles per symbol

def test_any_block_cut():
    x = qpsk_wave(30000, ppm=40, cfo=2e-3)
    ref, _ = run(x, timing_and_carrier)
    for cuts in ([1000]*(len(x)//1000) + [len(x) % 1000], [4999, 7, 30000, len(x) - 35006]):
        y, _ = run(x, timing_and_carrier, cuts)
        assert len(y) == len(ref) and np.allclose(y, ref, atol=1e-4)

def test_a_gap_is_coasted_through():
    """Noise where the signal stops: the windows there are gated out, the loops coast, and the signal after is clean"""
    x = qpsk_wave(40000, ppm=30, cfo=2e-3)
    x[60000:75000] = (0.05*(np.random.default_rng(1).normal(size=15000) + 1j*np.random.default_rng(2).normal(size=15000))).astype(np.complex64)
    y, s = run(x, timing_and_carrier)
    assert s.est["timing"]["windows"] >= 15 and evm_qpsk(y[20000:]) < 0.2

def test_noise_alone_acquires_nothing():
    rng = np.random.default_rng(3)
    x = (rng.normal(size=40000) + 1j*rng.normal(size=40000)).astype(np.complex64)
    y, s = run(x, lambda: [rp.sync.phase.MPower(8, window=0)], sps=1, Fs=1e6)
    assert s.est["carrier"]["frequency"] is None and np.array_equal(y, x) # passed on, untouched
    _, s = run(np.repeat(x, 4), lambda: [rp.sync.time.OerderMeyr(window=0)])
    assert s.est["timing"]["windows"] <= 2 # (a window can pass the gate by chance)

def test_carrier_order_is_controlled_live():
    c = rp.sync.phase.MPower(4)
    c.control(order=8)
    assert c.order == 8 and c.CONTROLS == {"order": "live"}

def test_block_ls_equalizer_reaches_the_known_symbol_bound():
    rng = np.random.default_rng(0)
    q = rp.symbol.load("QPSK")
    sym = q.points[rng.integers(0, 4, 20000)]
    x = np.convolve(sym, [0.15j, 1, 0.45 - 0.2j, -0.2])[1:1 + len(sym)]*0.3*np.exp(0.2j)
    x = (x + 0.01*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)
    make = lambda: [rp.sync.equalize.BlockLS(q, 21, window=0)]
    y, s = run(x, make, Fs=1e6, sps=1)
    H = np.lib.stride_tricks.sliding_window_view(np.r_[np.zeros(10), x, np.zeros(10)], 21)[:, ::-1]
    bound = np.sqrt(np.mean(np.abs(H @ np.linalg.lstsq(H, sym, rcond=None)[0] - sym)**2))
    assert len(y) == len(x) and np.sqrt(np.mean(np.abs(y[2048:] - sym[2048:])**2)) < 1.1*bound
    y2, _ = run(x, make, [777]*(len(x)//777) + [len(x) % 777], Fs=1e6, sps=1)
    assert np.allclose(y, y2, atol=1e-5)

# ---- the profiles from raw samples: nothing but their own steps

def impaired(o, cfo_hz, pad):
    x = np.concatenate((np.zeros(pad, np.complex64), o.data, np.zeros(pad, np.complex64)))
    d = rp.Data(x, Fs=o.Fs, sps=4, flush=True)
    d = d + ch.TimingOffset(0.37) + ch.ClockDrift(20) + ch.CFO(cfo_hz) + ch.PhaseOffset(1.1) + \
        ch.AWGN(esn0_db=12, power=float(np.mean(np.abs(o.data)**2)), seed=3)
    return np.asarray(d.data)

def receive(p, x, Fs, cuts=None):
    r, st, outs, i = p.rx(), None, [], 0
    cuts = cuts or [len(x)]
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=Fs, sps=4, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        outs.append(np.asarray(r(d).data).astype(bool))
    return np.concatenate(outs), st

def test_dvbs2_from_raw_samples():
    p = rp.profile.load("dvbs2/normal")
    frames = [{"modcod": 4, "pilots": 1}, {"modcod": 12, "pilots": 1}, {"modcod": 2, "pilots": 0}, {"modcod": 4, "pilots": 1}]
    t = rp.profile.table("dvbs2/modcod")
    data = np.random.default_rng(0).integers(0, 2, sum(t.row({**f, "short": 0})["kbch"] - 80 for f in frames)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=4e6, sps=1, flush=True), fields=frames)
    got, st = receive(p, impaired(o, 0.009*o.Fs/4, 4000), o.Fs) # 0.009 cycles per symbol
    assert np.array_equal(got, data)
    assert abs(st.est["timing"]["ppm"] - 20) < 5 and abs(st.est["carrier"]["frequency"] - 0.009) < 1e-4

def test_80211b_from_raw_samples():
    p = rp.profile.load("ieee80211b/long")
    rng = np.random.default_rng(1)
    packets = [rng.integers(0, 2, 8*n).astype(bool) for n in (300, 120, 500)]
    tx, s, w = p.tx(), None, []
    for i, (b, r) in enumerate(zip(packets, [20, 10, 20])):
        d = rp.Data(b, Fs=1e6, sps=1, stream=s, flush=i == 2)
        s = d.stream
        o = tx(d, fields={"signal": r})
        w.append(o.data)
    o.data = np.concatenate(w)
    x = impaired(o, 36e3, 20000) # a HackRF 15 ppm off at 2.4 GHz
    data = np.concatenate(packets)
    got, st = receive(p, x, o.Fs)
    assert len(got) == len(data) and np.array_equal(got[:-2], data[:-2]) # (the stream's end clips its last symbol)
    assert [f["signal"] for _, f, _ in st.control.fields("plcp") if "bits" in f] == [20, 10, 20]
    got2, _ = receive(p, x, o.Fs, [9973]*(len(x)//9973) + [len(x) % 9973])
    assert np.array_equal(got2, got)
