"""The documented recipes: signals built from the symbol layer (`*`, `/`) and the waveform layer (`+`)"""
import math

import numpy as np
import pytest

import rpps as rp

HALFSINE = "halfsine/halfsine"

def oqpsk_tx(bits_or_syms, sps):
    return bits_or_syms + rp.signal.sample.up.Zeros(sps) + rp.signal.filters.load("pulse", HALFSINE) + rp.wave.Stagger()

def oqpsk_rx(wave, sps):
    """Coherent: realign the rails, matched half-sine filter, one sample per symbol at the pulse peak (scaled to unit pulses)"""
    out = wave + rp.wave.Stagger(rail="i") + rp.signal.filters.load("pulse", HALFSINE) + rp.signal.sample.down.Decimate(sps, offset=sps + sps//2)
    return out.data*2/sps

def test_oqpsk_bits_loopback():
    rng = np.random.default_rng(0)
    sps = 8
    c = rp.symbol.load("QPSK")
    msg = rng.integers(0, 256, 100).astype(np.uint8)
    bits = np.unpackbits(msg).astype(bool)
    wave = oqpsk_tx(rp.Data(msg, Fs=1e6, sps=1) * c, sps)
    assert wave.sps == sps and wave.Fs == pytest.approx(8e6)
    got = oqpsk_rx(wave, sps)
    llr = c.llr(got.astype(np.complex64), noise_var=1e-3)
    n = len(llr)
    assert n >= len(bits) - 4*c.bits_per_symbol # the last symbols are still inside the filters
    assert np.array_equal(llr < 0, bits[:n])

def test_oqpsk_error_rate_is_qpsk_theory():
    rng = np.random.default_rng(1)
    sps, N = 8, 60000
    c = rp.symbol.load("QPSK")
    bits = rng.integers(0, 2, N).astype(bool)
    wave = oqpsk_tx(rp.Data(bits, Fs=1e6, sps=1) * c, sps)
    s2 = 0.5 # complex noise variance per sample: symbol-domain N0 = 2*s2/sps, Es = 1, Eb = 1/2
    noisy = wave.data + np.sqrt(s2/2)*(rng.standard_normal(len(wave)) + 1j*rng.standard_normal(len(wave))).astype(np.complex64)
    got = oqpsk_rx(rp.Data(noisy, Fs=8e6, sps=sps), sps)
    ber = np.mean((c.llr(got.astype(np.complex64), noise_var=2*s2/sps) < 0) != bits[:2*len(got)])
    ebn0 = 1.0/2/(2*s2/sps)
    theory = 0.5*math.erfc(math.sqrt(ebn0))
    assert 0.6*theory < ber < 1.6*theory, (ber, theory)

def test_oqpsk_any_block_cut_and_constant_phase():
    rng = np.random.default_rng(2)
    sps = 8
    c = rp.symbol.load("QPSK")
    syms = (rp.Data(rng.integers(0, 256, 40).astype(np.uint8), Fs=1e6, sps=1) * c)
    up, hs, st = rp.signal.sample.up.Zeros(sps), rp.signal.filters.load("pulse", HALFSINE), rp.wave.Stagger()
    ref = (syms + up + hs + st).data
    x = syms.data
    s, outs, i = None, [], 0
    up2, hs2, st2 = rp.signal.sample.up.Zeros(sps), rp.signal.filters.load("pulse", HALFSINE), rp.wave.Stagger()
    for b in [1]*10 + [0, 33, len(x) - 43]:
        d = rp.Data(x[i:i + b], stream=s, Fs=1e6, sps=1)
        s = d.stream
        outs.append((d + up2 + hs2 + st2).data)
        i += b
    assert np.allclose(np.concatenate(outs), ref, atol=1e-5)

def test_oqpsk_carrier_phase_comes_off_before_the_rails_are_realigned():
    """A carrier phase mixes I into Q while Q is still half a symbol late, so it has to be removed from the staggered signal.
    Realigning first and rotating the symbols after leaves the rails mixed at different instants"""
    rng = np.random.default_rng(2)
    sps, phi = 8, 0.4
    c = rp.symbol.load("QPSK")
    wave = oqpsk_tx(rp.Data(rng.integers(0, 256, 40).astype(np.uint8), Fs=1e6, sps=1) * c, sps).data
    clean = oqpsk_rx(rp.Data(wave, Fs=8e6, sps=sps), sps)
    turned = wave*np.exp(1j*phi)
    early = oqpsk_rx(rp.Data(turned*np.exp(-1j*phi), Fs=8e6, sps=sps), sps) # corrected on the staggered signal
    late = oqpsk_rx(rp.Data(turned, Fs=8e6, sps=sps), sps)*np.exp(-1j*phi) # corrected after realigning
    assert np.allclose(early, clean, atol=1e-5)
    assert np.abs(late - clean).max() > 0.1

@pytest.mark.parametrize("sigma2", [0.0, 2.0])
def test_zigbee_2450_waveform_loopback(sigma2):
    """802.15.4 at 2.4 GHz: 4 bits pick a 32-chip sequence, even chips on I and odd on Q, half-sine pulses, Q staggered by half a symbol"""
    rng = np.random.default_rng(3)
    sps = 4
    code = rp.sequence.load("ieee802154_2450")
    spread = rp.symbol.Spread(code, pair=True)
    msg = rng.integers(0, 256, 80).astype(np.uint8)
    chips = rp.Data(msg, Fs=250e3, sps=1) * spread
    assert chips.Fs == pytest.approx(1e6) # 250 kb/s is 62.5k symbols of 16 I+Q chip pairs each: 1M pairs/s
    wave = oqpsk_tx(chips, sps)
    x = wave.data + np.sqrt(sigma2/2)*(rng.standard_normal(len(wave)) + 1j*rng.standard_normal(len(wave))).astype(np.complex64)
    pairs = oqpsk_rx(rp.Data(x, Fs=wave.Fs, sps=sps), sps)
    rx = rp.Data(pairs.astype(np.complex64), Fs=1e6, sps=1)
    rx = rx.derive(rx.data, rp.Category.SYMBOL, rp.dtype.SYMBOLS, rp.base.Mod, {})
    llr = rx / spread
    got = np.asarray(llr.data) < 0
    bits = np.unpackbits(msg).astype(bool)
    n = len(got)
    assert n >= len(bits) - 8
    assert np.array_equal(got, bits[:n]) # processing gain of the 32-chip sequences carries it through the noise

def test_dsss_waveform_wifi_dbpsk_with_rrc_shaping():
    """802.11b 1 Mb/s through pulse shaping and back: DBPSK, Barker-11, RRC at 4 samples per chip, matched filter, chip sampling"""
    rng = np.random.default_rng(4)
    sps = 4
    msg = rng.integers(0, 256, 30).astype(np.uint8)
    bpsk, diff, spread = rp.symbol.load("bpsk"), rp.symbol.Differential(), rp.symbol.Spread(rp.sequence.load("barker11"))
    bits = np.unpackbits(msg).astype(bool)
    chips = rp.Data(msg, Fs=1e6, sps=1) * bpsk * diff * spread
    rrc = rp.signal.filters.load("pulse", "rrc/0p35")
    tx = chips + rp.signal.sample.up.Zeros(sps) + rrc
    rx = tx + rrc # matched filter; rx.delay is the delay of both filters
    pick = rx.data[int(round(rx.delay*rx.Fs))::sps] # one sample per chip, at the pulse peaks
    n = (len(pick)//11)*11
    syms = rx.derive(pick[:n].astype(np.complex64), rp.Category.SYMBOL, rp.dtype.SYMBOLS, rp.base.Mod, {})
    got = np.asarray((syms / spread / diff / bpsk).data) < 0
    assert len(got) >= len(bits) - 16
    assert np.array_equal(got[2:len(bits) - 16], bits[2:len(bits) - 16])
