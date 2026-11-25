"""OFDM: carriers (symbol), IFFT and prefix (wave) and the pilot equalizer (sync)"""
import numpy as np

import rpps as rp

# --- OFDM: carriers (symbol) + IFFT/prefix (wave) + pilot equalizer (sync) ---
def bits_for_ofdm(rng, mod, n_ofdm, carriers):
    return rng.integers(0, 2, mod.bits_per_symbol*carriers.data_per_symbol*n_ofdm).astype(bool)

def ofdm_tx(bits, qam, carriers):
    return rp.Data(bits) * qam * carriers + rp.wave.OFDMTx(64, 16)

def ofdm_rx(wave, qam, carriers):
    return (wave + rp.wave.OFDMRx(64, 16) + rp.sync.PilotEqualizer(carriers)) / carriers / qam

def test_ofdm_round_trip(rng):
    carriers, qam = rp.symbol.Carriers(), rp.symbol.QAM(16)
    bits = bits_for_ofdm(rng, qam, 20, carriers)
    wave = ofdm_tx(bits, qam, carriers)
    assert len(wave) == 20*80 and wave.REP == rp.Rep.SAMPLES
    fft = wave + rp.wave.OFDMRx(64, 16)
    assert fft.REP == rp.Rep.SYMBOLS and len(fft) == 20*64
    assert np.array_equal(ofdm_rx(wave, qam, carriers).bits(), bits)

def test_ofdm_equalizes_mild_multipath_and_a_phase_offset(rng):
    carriers, qam = rp.symbol.Carriers(), rp.symbol.QAM(16)
    bits = bits_for_ofdm(rng, qam, 60, carriers)
    wave = ofdm_tx(bits, qam, carriers)
    rx = wave + rp.signal.channel.Multipath([1.0, 0.3j, -0.15], [0, 1, 2]) + rp.signal.channel.PhaseOffset(0.8) + rp.signal.channel.AWGN(35, power=None, seed=2)
    assert np.mean(ofdm_rx(rx, qam, carriers).bits() != bits) < 0.01

def test_ofdm_with_more_pilots_handles_a_longer_delay_spread(rng):
    # The channel changes faster across carriers the longer its echoes: the pilots have to sample it often enough
    pilots = tuple(k for k in range(-24, 25, 3) if k != 0)
    carriers, qam = rp.symbol.Carriers(pilots=pilots), rp.symbol.QAM(16)
    bits = bits_for_ofdm(rng, qam, 60, carriers)
    wave = ofdm_tx(bits, qam, carriers)
    rx = wave + rp.signal.channel.Multipath([1.0, 0.5j, -0.3], [0, 3, 9]) + rp.signal.channel.AWGN(35, power=None, seed=2)
    assert np.mean(ofdm_rx(rx, qam, carriers).bits() != bits) < 0.02

def test_ofdm_streams_in_chunks(rng):
    """Any cut of the transmit symbols and of the received samples gives the same output (the equalizer finds the pilots'
    sign from the stream position)"""
    carriers, qam = rp.symbol.Carriers(), rp.symbol.QAM(4)
    bits = bits_for_ofdm(rng, qam, 12, carriers)
    syms = (rp.Data(bits) * qam).data
    whole = (rp.Data(syms, flush=True, sps=1) * carriers + rp.wave.OFDMTx(64, 16)).data
    ctx, s, outs = None, None, []
    tx = rp.wave.OFDMTx(64, 16)
    for a, b in ((0, 100), (100, 101), (101, 300), (300, len(syms))):
        d = rp.Data(syms[a:b], ctx=ctx, sps=1, stream=s)
        s = d.stream
        out = d * carriers
        ctx = out.ctx
        outs.append((out + tx).data)
    assert np.allclose(np.concatenate(outs), whole, atol=1e-6)
    rx, eq, s, got = rp.wave.OFDMRx(64, 16), rp.sync.PilotEqualizer(carriers), None, []
    for a, b in ((0, 77), (77, 78), (78, 500), (500, len(whole))):
        d = rp.Data(whole[a:b], stream=s)
        s = d.stream
        got.append((d + rx + eq).data)
    ref = (rp.Data(whole) + rp.wave.OFDMRx(64, 16) + rp.sync.PilotEqualizer(carriers)).data
    assert np.allclose(np.concatenate(got), ref, atol=1e-5)
