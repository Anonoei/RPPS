"""rpps.analysis.waveform: tones, OFDM (and which system), bursts, constant envelope"""
import numpy as np

import rpps as rp
from rpps.analysis import waveform as W

RNG = np.random.default_rng(7)

def _ofdm(nfft, cp, symbols, used, rng=RNG):
    """OFDM: QPSK on the `used` subcarriers around DC (not DC), each symbol with a cyclic prefix of `cp` samples"""
    out = []
    k = np.concatenate((np.arange(-used//2, 0), np.arange(1, used//2 + 1))) % nfft
    for _ in range(symbols):
        X = np.zeros(nfft, complex)
        X[k] = (rng.choice([-1, 1], len(k)) + 1j*rng.choice([-1, 1], len(k)))/np.sqrt(2)
        s = np.fft.ifft(X)
        out.append(np.concatenate((s[-cp:], s)))
    return np.concatenate(out)

def _noise(n, snr_db, x, rng=RNG):
    p = np.mean(np.abs(x)**2)/10**(snr_db/10)
    return x + np.sqrt(p/2)*(rng.standard_normal(n) + 1j*rng.standard_normal(n))

def test_lte_like_ofdm_is_named_by_its_subcarriers_and_prefix():
    # 15 kHz subcarriers, normal prefix (144 of 2048) at 7.68 MS/s: FFT 512, prefix 36
    x = _ofdm(512, 36, 600, 300)
    o = W.ofdm(_noise(len(x), 10, x), 7.68e6)
    assert o is not None and abs(o.scs - 15e3) < 30 and abs(o.tcp - 4.69e-6) < 0.3e-6
    assert o.systems[0][0].startswith("LTE / NR 15 kHz, normal CP")

def test_wifi_ofdm_is_found_with_its_guard_interval():
    x = _ofdm(64, 16, 4000, 52) # 802.11a/g at 20 MS/s: 3.2 us symbols, 0.8 us guard
    o = W.ofdm(_noise(len(x), 15, x), 20e6)
    assert o is not None and o.systems and o.systems[0][0].startswith("Wi-Fi 802.11a/g/n/ac") and "long GI" in o.systems[0][0]

def test_a_single_carrier_whose_frame_repeats_is_not_ofdm():
    # QPSK with a 90-symbol preamble every 2000 symbols: correlates at the frame length, but is no Gaussian
    q = rp.symbol.load("qpsk").points
    pre = q[RNG.integers(0, 4, 90)]
    syms = np.concatenate([np.concatenate((pre, q[RNG.integers(0, 4, 1910)])) for _ in range(40)])
    up = np.zeros(4*len(syms), complex)
    up[::4] = syms
    x = np.convolve(up, rp.signal.filters.rrc_pulse(np.arange(-32, 33)/4, 0.35), mode="same")
    assert W.ofdm(_noise(len(x), 15, x), 4e6) is None
    w = W.classify(_noise(len(x), 15, x), 4e6)
    assert w.kind is None and w.tones is None

def test_tones_and_their_spacing():
    fs, n = 8000.0, 4096
    t = np.arange(n)/fs
    x = np.exp(2j*np.pi*-490*t) + 0.95*np.exp(2j*np.pi*130*t + 1)
    tn = W.tones(_noise(n, 30, x), fs)
    assert tn is not None and len(tn.lines) == 2
    assert abs(sorted(f for f, _ in tn.lines)[1] - sorted(f for f, _ in tn.lines)[0] - 620) < 2*fs/n
    w = W.classify(_noise(n, 30, x), fs)
    assert w.kind == "tones" and w.bursts is None # Two tones beat: not bursts
    assert W.tones(_noise(n, 0, 0*x + 1e-3), fs) is None # Noise has no lines

def test_tdd_bursts_and_their_period():
    fs = 1e6
    x = _noise(int(0.05*fs), 20, np.exp(2j*np.pi*RNG.random(int(0.05*fs))))
    t = np.arange(len(x))/fs
    x = x*((t % 5e-3) < 3.7e-3) + 1e-3*(RNG.standard_normal(len(x)))
    b = W.bursts(x, fs)
    assert b is not None and abs(b.period - 5e-3) < 0.05e-3 and abs(b.duty - 0.74) < 0.03
    assert W.bursts(_noise(len(x), 20, np.ones(len(x))), fs) is None # Steady

def test_constant_envelope():
    fs = 1e6
    t = np.arange(2**16)/fs
    fm = np.exp(1j*2*np.pi*50e3*np.cumsum(np.sin(2*np.pi*1e3*t))/fs)
    w = W.classify(_noise(len(t), 30, fm), fs)
    assert w.kind == "constant envelope" and w.envelope < 1.1

def test_a_pilot_comb_is_named_by_its_grid_not_its_repeat():
    # Symbols using every 3rd subcarrier of a 15 kHz grid repeat three times within the symbol: the lag is Tu/3 (a "45 kHz"
    # OFDM by the prefix test alone), and correlates at 2 Tu/3 too. The system is the 15 kHz grid's
    nfft, cp = 512, 36 # 7.68 MS/s
    out = []
    k = np.concatenate((np.arange(-150, 0, 3), np.arange(3, 151, 3))) % nfft
    for _ in range(600):
        X = np.zeros(nfft, complex)
        X[k] = (RNG.choice([-1, 1], len(k)) + 1j*RNG.choice([-1, 1], len(k)))/np.sqrt(2)
        s = np.fft.ifft(X)
        out.append(np.concatenate((s[-cp:], s)))
    x = np.concatenate(out)
    o = W.ofdm(_noise(len(x), 15, x), 7.68e6)
    assert o is not None and o.comb == 3 and abs(o.tu - 66.67e-6/3) < 0.3e-6
    assert o.systems and o.systems[0][0].startswith("LTE / NR 15 kHz") and "every 3rd subcarrier" in str(o)
