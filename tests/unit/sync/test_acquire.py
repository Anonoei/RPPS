"""rp.sync.Acquire decides at fixed stream positions: the same detection however the stream is cut into blocks"""
import numpy as np
import pytest

import rpps as rp
from rpps import symbol

from support.signals import shaped

def stream(x, blocks, make):
    """x cut into `blocks` of one stream (the last flushing) through a fresh stage: (samples out, last detection)"""
    stage = make()
    st, outs, det, i = None, [], None, 0
    for k, b in enumerate(blocks):
        d = rp.Data(x[i:i + b], Fs=1e6, sps=4, stream=st, flush=k == len(blocks) - 1)
        st = d.stream
        o = d + stage
        outs.append(np.asarray(o.data))
        det = o.meta.det or det
        i += b
    return np.concatenate(outs), det

def cuts(n):
    return {"whole": [n], "3072": [3072]*(n//3072) + [n % 3072], "16384": [16384]*(n//16384) + [n % 16384],
            "777": [777]*(n//777) + [n % 777], "ones then the rest": [1]*50 + [n - 50]}

@pytest.mark.parametrize("mod, esn0", [("QPSK", 4.0), ("8PSK", 9.0)])
def test_acquire_any_block_cut(mod, esn0):
    """Near the decision's edge (a first look at a few thousand samples could go either way): every cut decides the
    same, from the same samples, and passes the same samples on"""
    rng = np.random.default_rng(3)
    x, _ = shaped(rng, symbol.load_or_generate(mod).points, 12000, 4, esn0_db=esn0, tau=0.4, cfo=1e-3)
    x = x.astype(np.complex64)
    ref = None
    for name, blocks in cuts(len(x)).items():
        out, det = stream(x, blocks, rp.sync.Acquire)
        got = (det.mod, det.beta, det.baud, det.tau, det.cfo, det.phase)
        if ref is None:
            ref, ref_out = got, out
            assert det.mod.lower() == mod.lower()
        else:
            assert got == ref, name
            assert np.array_equal(out, ref_out), name

def test_acquire_measures_from_where_the_signal_comes_on():
    """QPSK at 15 dB after 5000 samples of the noise alone (a carrier switched on after the recording started): the
    Es/N0 is the signal's (it read 10.8 dB with the noise counted), the timing on the same grid as without the lead,
    and every block cut decides the same"""
    rng = np.random.default_rng(5)
    x0, _ = shaped(rng, symbol.load_or_generate("QPSK").points, 12000, 4, tau=0.4)
    lead = 5000
    var = np.mean(np.abs(x0)**2)*4/10**1.5 # Es/N0 15 dB: Es = mean |x|^2 * sps
    noise = np.sqrt(var/2)*(rng.standard_normal(lead + len(x0)) + 1j*rng.standard_normal(lead + len(x0)))
    x = (np.concatenate((np.zeros(lead), x0)) + noise).astype(np.complex64)
    assert rp.sync.acquire.onset(x) == lead//256*256 + 256 or rp.sync.acquire.onset(x) == lead//256*256
    ref = None
    for name, blocks in cuts(len(x)).items():
        _, det = stream(x, blocks, rp.sync.Acquire)
        got = (det.mod, det.baud, det.tau, det.esn0)
        if ref is None:
            ref = got
            assert det.mod.lower() == "qpsk" and abs(det.esn0 - 15.0) < 0.5
            _, d0 = stream((x0 + noise[lead:]).astype(np.complex64), [len(x0)], rp.sync.Acquire)
            assert abs(((det.tau - lead - d0.tau + 2) % 4) - 2) < 0.05 # (the same instants, the lead's samples on)
        else:
            assert got == ref, name

def test_acquire_decides_at_the_stream_end():
    """A stream shorter than the next decision position is decided on all of it when it ends"""
    rng = np.random.default_rng(4)
    x, _ = shaped(rng, symbol.load_or_generate("QPSK").points, 700, 4, esn0_db=15, tau=0.2)
    x = x.astype(np.complex64)
    a = stream(x, [len(x)], rp.sync.Acquire)[1]
    b = stream(x, [1000, 1000, len(x) - 2000], rp.sync.Acquire)[1]
    assert a is not None and a.mod.lower() == "qpsk"
    assert (a.mod, a.baud, a.tau) == (b.mod, b.baud, b.tau)

@pytest.mark.slow
def test_acquire_keeps_its_decision_on_a_channelised_capture():
    """DVB-S QPSK 3/4 at 1 MBd, 1.7 MHz off in 8 MS/s, brought down by an FFT channeliser: decided per arriving block
    (the first decision, on a few thousand samples, kept), blocks of 3072 or 16384 were reported to name it BPSK at
    1-2.5 dB. Small blocks and large give the same decision: QPSK at 14 dB"""
    rng = np.random.default_rng(0)
    ts = np.zeros((900, 188), np.uint8)
    ts[:, 0] = 0x47
    ts[:, 1:] = rng.integers(0, 256, (900, 187))
    w = rp.profile.load("dvbs/qpsk_3_4", sps=8).tx()(rp.Data(np.unpackbits(ts.reshape(-1)).astype(bool), Fs=1e6, sps=1,
                                                             flush=True))
    x = np.asarray(w.data)
    y = x*np.exp(2j*np.pi*1.7e6/8e6*np.arange(len(x))) + 0.05*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    y = y.astype(np.complex64)[:2**21]
    s = max(rp.analysis.survey(rp.source.Memory(y, Fs=8e6)), key=lambda g: g.width - abs(g.mid - 1.7e6)*10)
    bank = rp.signal.FFTBank(8e6, [(s.mid, s.width)])
    z = np.concatenate([bank.step(y)[0], bank.flush()[0]])
    for blocks in ([3072]*(len(z)//3072) + [len(z) % 3072], [131072]*(len(z)//131072) + [len(z) % 131072]):
        stage = rp.sync.Acquire()
        st, det, i = None, None, 0
        for k, b in enumerate(blocks):
            d = rp.Data(z[i:i + b], Fs=bank.Fs_out[0], stream=st, flush=k == len(blocks) - 1)
            st = d.stream
            det = (d + stage).meta.det or det
            i += b
        assert det.mod == "qpsk" and det.esn0 > 12

@pytest.mark.parametrize("mod", ["qam/16qam", "psk/qpsk"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_signal_at_two_samples_a_symbol_locks_at_its_rate(mod, seed):
    """At exactly 2 samples a symbol |x|^2 puts the rate line on Fs/2, where the search couldn't see it (a noise peak was
    taken: 16QAM at 0.17-0.53 of its rate); interpolated to twice the rate it is found. 10 dB: the rate within 1e-5,
    the modulation, Es/N0 within 1 dB (10.3-10.4 over 4 seeds)"""
    from rpps.signal.filters.shaping import rrc_pulse
    rng = np.random.default_rng(seed)
    pts = np.asarray(rp.symbol.load(mod).points)
    s = pts[rng.integers(0, len(pts), 6000)]
    h = rrc_pulse(np.arange(-16, 17)/2, 0.35)
    up = np.zeros(2*len(s), complex)
    up[::2] = s
    x = np.convolve(up, h/np.sqrt(np.sum(h**2)))[16:16 + len(up)]
    x = x + np.sqrt(np.mean(np.abs(x)**2)*2/10)/np.sqrt(2)*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))
    x = x*np.exp(2j*np.pi*0.001*np.arange(len(x)))
    lk = rp.sync.Lock.find(x, 2e6)
    assert abs(lk.det.baud/1e6 - 1) < 1e-5 and lk.det.mod == mod.split("/")[1]
    assert abs(lk.quality.esn0_db - 10) < 1.0

@pytest.mark.parametrize("rate", [0.0, 2e5, -5e5, 1e6])
def test_a_drifting_carrier_is_measured_and_taken_off(rate):
    """QPSK at 1 MBd, its carrier ramping `rate` Hz/s (1 MHz/s: 16 kHz over the 16 ms searched): detect gives the rate
    (Detection.drift, over 5 seeds within 0.3%; 0.0 without one) and Es/N0 as without it (within 0.1 dB; it read 3.8
    dB low)"""
    from rpps.signal.filters.shaping import rrc_pulse
    for seed in range(3):
        rng = np.random.default_rng(seed)
        q = np.exp(1j*(np.pi/4 + np.pi/2*np.arange(4)))
        s = q[rng.integers(0, 4, 16000)]
        h = rrc_pulse(np.arange(-32, 33)/4, 0.35)
        up = np.zeros(4*len(s), complex)
        up[::4] = s
        x = np.convolve(up, h/np.sqrt(np.sum(h**2)))[32:32 + len(up)]
        t = np.arange(len(x))/4e6
        x = x*np.exp(2j*np.pi*(1e3*t + 0.5*rate*t**2))
        x = x + np.sqrt(np.mean(np.abs(x)**2)*4/10**1.5/2)*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))
        d = rp.sync.acquire.detect(x, 4e6)
        assert (d.drift == 0.0) if rate == 0 else abs(d.drift/rate - 1) < 0.01
        assert abs(d.esn0 - 15) < 0.3
