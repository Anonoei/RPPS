"""OFDM symbol timing and fractional frequency from the cyclic prefix (rpps.sync.CPSync)"""
import numpy as np
import pytest

import rpps as rp
from rpps.profile import steps

N, CP = 64, 16
L = N + CP


def ofdm(S=300, start=37, eps=0.13, snr_db=20.0, ppm=0.0, seed=0):
    """QPSK on every carrier, prefixed, `start` samples of near silence first, a frequency offset of `eps` subcarrier
    spacings, a sample clock `ppm` fast (each sample taken at n (1 + ppm 1e-6) of the transmitter's time, from the OFDM
    waveform itself: a symbol is a sum of carriers over its prefix and body), noise. Returns (signal, the sent values)"""
    rng = np.random.default_rng(seed)
    X = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, (S, N)) + 1))
    k = np.fft.fftfreq(N)*N
    t = np.arange(int((start + S*L)/(1 + ppm*1e-6)))*(1 + ppm*1e-6) - start # transmitter time of each sample
    s = np.floor(t/L).astype(int)
    ok = (t >= 0) & (s < S)
    u = (t - s*L - CP)[ok] # time from the symbol body's start (the prefix is its end, cyclically)
    x = 0.01*rng.standard_normal(len(t)).astype(complex)
    x[ok] = np.sum(X[s[ok]]*np.exp(2j*np.pi*k[None, :]*u[:, None]/N), axis=1)/np.sqrt(N)
    x = x*np.exp(2j*np.pi*eps/N*np.arange(len(x)) + 0.4j)
    sd = 10**(-snr_db/20)
    x = x + sd*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))/np.sqrt(2)
    return x.astype(np.complex64), X

def receive(x, cut=None, **kw):
    stage, rx, st, outs, i = rp.sync.CPSync(N, CP, min_symbols=16, **kw), rp.wave.OFDMRx(N, CP), None, [], 0
    for b in cut or [len(x)]:
        d = rp.Data(x[i:i + b], stream=st, Fs=1e6, flush=i + b >= len(x))
        st = d.stream
        outs.append(np.asarray((d + stage + rx).data))
        i += b
    y = np.concatenate(outs)
    return y[:len(y)//N*N].reshape(-1, N), st

def evm_db(Y, X):
    """EVM against the sent values, each symbol's phase line across the carriers taken off (a common phase, and the ramp a
    timing offset inside the prefix gives: what a pilot equalizer does after)"""
    k0 = int(np.argmax([abs(np.vdot(X[j], Y[0])) for j in range(5)]))
    n = min(len(Y), len(X) - k0)
    R = Y[:n]*np.conj(X[k0:k0 + n])
    k = np.fft.fftfreq(N)*N
    out = []
    for r, y, x in zip(R, Y[:n], X[k0:k0 + n]):
        slope = np.angle(np.sum(r[1:]*np.conj(r[:-1])*(np.diff(k) == 1))) # phase step per carrier
        c = np.angle(np.sum(r*np.exp(-1j*slope*k)))
        out.append(y*np.exp(-1j*(c + slope*k)) - x)
    E = np.array(out)
    return 10*np.log10(np.mean(np.abs(E[20:])**2)), k0

def test_timing_and_frequency_found():
    x, X = ofdm()
    Y, st = receive(x)
    est = st.est["ofdm"]
    assert est["timing"] == pytest.approx(37, abs=0.5) and est["cfo"] == pytest.approx(0.13, abs=0.01)
    e, k0 = evm_db(Y, X)
    assert k0 == 0 and e < -19 # at the noise (20 dB SNR): nothing added

def test_any_block_cut():
    x, _ = ofdm(S=200)
    ref, _ = receive(x)
    y, _ = receive(x, [1, 999, 7, 6000, len(x) - 7007])
    assert y.shape == ref.shape and np.allclose(y, ref, atol=1e-5)

def test_clock_drift_followed():
    """A sample clock 300 ppm off: the symbol start walks by a sample every ~40 symbols, and the output keeps whole symbols"""
    x, X = ofdm(S=150, ppm=300.0)
    Y, st = receive(x)
    e, _ = evm_db(Y, X)
    assert e < -15 and abs(len(Y) - 150) <= 2

def test_noise_alone_is_gated():
    rng = np.random.default_rng(3)
    x = ((rng.standard_normal(40000) + 1j*rng.standard_normal(40000))/np.sqrt(2)).astype(np.complex64)
    Y, st = receive(x)
    assert len(Y) == 0 and st.est["ofdm"]["timing"] is None

def test_profile_step():
    st = steps.build({"sync": "ofdm", "n_fft": 128, "n_cp": 32}, rp.Rep.SAMPLES, steps.context())
    assert isinstance(st.rx[0][1], rp.sync.CPSync) and st.rx[0][1].N == 128
