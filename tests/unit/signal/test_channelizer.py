"""Spectrum inversion and the channelizer"""
import numpy as np

import rpps as rp
from rpps.base.types import Rep

def test_channelizer_separates_tones_and_is_block_independent(rng):
    M, Fs = 8, 8000.0
    n = np.arange(64000)
    tones = {1: 1000.0 + 50, 3: 3000.0 - 80, 6: -2000.0 + 20} # In channels 1, 3 and 6 (= -2)
    x = sum(np.exp(2j*np.pi*f*n/Fs) for f in tones.values()).astype(np.complex64)
    whole = rp.Data(x, Fs=Fs) + rp.signal.filters.Channelizer(M)
    assert whole.Fs == Fs/M and whole.data.shape[1] == M
    power = np.mean(np.abs(whole.data[100:])**2, axis=0)
    assert set(np.nonzero(power > 0.5)[0]) == {1, 3, 6}
    # Pieces of any length give the same bands
    pipe = rp.signal.filters.Channelizer(M)
    stream, outs = None, []
    for a, b in ((0, 777), (777, 5000), (5000, 5001), (5001, len(x))):
        d = rp.Data(x[a:b], Fs=Fs, stream=stream) + pipe
        stream = d.stream
        outs.append(d.data)
    assert np.allclose(np.concatenate(outs), whole.data, atol=1e-5)

# ---- signal and sync additions

def test_conjugate_inverts_the_spectrum():
    x = np.exp(2j*np.pi*0.1*np.arange(64)).astype(np.complex64)
    out = rp.Data(x, Fs=1.0) + rp.signal.Conjugate()
    assert out.REP == Rep.SAMPLES and np.allclose(np.asarray(out.data), np.conj(x))
    assert np.argmax(np.abs(np.fft.fft(np.asarray(out.data)))) == 64 - np.argmax(np.abs(np.fft.fft(x)))

def test_channelize_stage_is_designed_from_the_stream_and_streams(rng):
    x = (np.exp(2j*np.pi*0.1*np.arange(60000)) + 0.1*(rng.normal(size=60000) + 1j*rng.normal(size=60000))).astype(np.complex64)
    st = rp.signal.Channelize(freq=200e3, width=50e3)
    whole = rp.Data(x, Fs=2e6, flush=True) + st
    assert 100e3 <= whole.Fs <= 200e3 and whole.REP == Rep.SAMPLES and abs(whole.cf - 200e3) < 1
    y = np.asarray(whole.data)
    f = np.fft.fftfreq(len(y), 1/whole.Fs)[np.argmax(np.abs(np.fft.fft(y)))]
    assert abs(f - (0.1*2e6 - 200e3)) < 2*whole.Fs/len(y)
    outs, s = [], None
    for a, b in ((0, 1001), (1001, 1002), (1002, 60000)):
        d = rp.Data(x[a:b], Fs=2e6, stream=s, flush=b == 60000)
        s = d.stream
        outs.append(np.asarray((d + st).data))
    cut = np.concatenate(outs)
    n = min(len(cut), len(y))
    assert np.allclose(cut[:n], y[:n], atol=1e-4)
