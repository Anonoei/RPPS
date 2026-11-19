"""GMSK and SOQPSK demodulated blind (rpps.analysis.cpm)"""
import numpy as np
import pytest

from rpps.analysis import cpm

def channel(rng, name, n, baud, Fs, ebn0, cfo=0.0, tau=0.0, ppm=0.0):
    """bits, and their signal at Fs: delayed tau symbols, carrier cfo (Hz), the clock ppm off, noise at Eb/N0 (dB)"""
    b = rng.integers(0, 2, n)
    up = 16
    y = cpm.modulate(b, name, up)
    t = np.arange(int(n/baud*Fs) - 40)/Fs*baud*(1 + ppm*1e-6)*up + tau*up
    t = t[t < len(y) - 10]
    z = cpm.interpolate(y, t, 32)*np.exp(1j*(1.1 + 2*np.pi*cfo*np.arange(len(t))/Fs))
    s = np.sqrt(Fs/baud/2/10**(ebn0/10))
    return b, z + s*(rng.standard_normal(len(z)) + 1j*rng.standard_normal(len(z)))

def errors(got, sent, soqpsk=False):
    """Bit error rate at the best of a few symbols' shift; SOQPSK up to its ambiguities (complement, alternate bits)"""
    best = 1.0
    for sh in range(-3, 4):
        a, b = got[max(0, sh):], sent[max(0, -sh):]
        n = min(len(a), len(b)) - 10
        alt = (np.arange(n) % 2).astype(sent.dtype)
        vs = [b[:n], 1 - b[:n], b[:n] ^ alt, 1 - (b[:n] ^ alt)] if soqpsk else [b[:n]]
        best = min(best, min(np.mean(a[:n] != v) for v in vs))
    return best

@pytest.mark.parametrize("name", ["msk", "gmsk_0.3", "gmsk_0.5", "soqpsk_mil", "soqpsk_tg"])
def test_sequence_detector_on_aligned_symbols(name):
    rng = np.random.default_rng(0)
    b = rng.integers(0, 2, 1500)
    y = cpm.modulate(b, name, 8)*np.exp(0.7j)
    y = y + 0.4*(rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y)))
    bits, match = cpm.mlse(y, name, 8)
    assert errors(bits[:-8], b[:-8], name.startswith("soqpsk")) < 0.003 and match > 0.6

@pytest.mark.parametrize("name", ["gmsk_0.3", "soqpsk_tg"])
def test_blind_with_carrier_timing_and_clock_off(name):
    """Rate from the x^2 lines, carrier, timing and a 40 ppm clock, all blind, at 4.565 samples a symbol"""
    rng = np.random.default_rng(1)
    Fs, baud = 48000*0.913, 9600
    b, y = channel(rng, name, 8000, baud, Fs, 8, cfo=730.0, tau=0.37, ppm=40)
    r = cpm.demodulate(y, Fs, name)
    assert abs(r.baud - baud*(1 + 40e-6)) < 0.5 and abs(r.cfo - 730) < 5
    assert errors(r.bits, b, name.startswith("soqpsk")) < 0.003

def test_identify_ranks_the_pulse():
    rng = np.random.default_rng(2)
    Fs = 48000*0.913
    _, y = channel(rng, "soqpsk_tg", 3000, 9600, Fs, 10, cfo=300.0, tau=0.2)
    assert cpm.identify(y, Fs, schemes=["gmsk_0.3", "soqpsk_mil", "soqpsk_tg", "msk"])[0][1] == "soqpsk_tg"

def test_soqpsk_complement_is_the_same_signal():
    """Why SOQPSK bits are only known up to the complement: the precoder gives the same symbols"""
    b = np.random.default_rng(3).integers(0, 2, 200)
    assert np.allclose(cpm.alphas(b, "soqpsk_tg")[2:], cpm.alphas(1 - b, "soqpsk_tg")[2:])

def test_pulses_integrate_to_a_half():
    for name in cpm.SCHEMES:
        g = cpm.frequency_pulse(name, 8)
        assert abs(g.sum()/8 - 0.5) < 1e-9
