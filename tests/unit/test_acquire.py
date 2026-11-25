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
