"""Generated constellations (PSK/QAM/APSK/ASK/FSK): round trips, power, soft decisions and error rates"""
import numpy as np
import pytest

import rpps as rp
from conftest import awgn
from rpps.mod.utils import shapes

NAMES = ["bpsk", "qpsk", "8psk", "16psk", "16qam", "32qam", "64qam", "128qam", "256qam", "4ask", "8ask", "ook", "16apsk", "32apsk"]

def bits_for(m, n_sym, rng):
    return rng.integers(0, 2, m.constellation.bits_per_symbol*n_sym).astype(bool)

@pytest.mark.parametrize("name", NAMES)
def test_round_trip_and_unit_power(name, rng):
    m = rp.mod.load(name)
    assert len(set(m.constellation.mapping.arr.tolist())) == len(m.constellation)
    bits = bits_for(m, 2000, rng)
    syms = rp.Data(bits) * m
    assert np.array_equal((syms / m).bits(), bits)
    if name != "ook":
        assert np.mean(np.abs(syms.data)**2) == pytest.approx(1.0, rel=0.1)

@pytest.mark.parametrize("M", [4, 16, 64, 256])
def test_square_qam_is_gray(M):
    pts, lab = shapes.qam(M)
    d = np.abs(pts[:, None] - pts[None, :])
    np.fill_diagonal(d, np.inf)
    near = np.argwhere(np.abs(d - d.min()) < 1e-9)
    assert all(bin(int(lab[i]) ^ int(lab[j])).count("1") == 1 for i, j in near)

@pytest.mark.parametrize("M", [32, 128])
def test_cross_qam_is_nearly_gray(M):
    pts, lab = shapes.qam(M)
    d = np.abs(pts[:, None] - pts[None, :])
    np.fill_diagonal(d, np.inf)
    near = np.argwhere(np.abs(d - d.min()) < 1e-9)
    assert np.mean([bin(int(lab[i]) ^ int(lab[j])).count("1") for i, j in near]) < 2.0
    assert sorted(lab.tolist()) == list(range(M))

def q(x):
    from math import erfc, sqrt
    return 0.5*erfc(x/sqrt(2))

@pytest.mark.parametrize("name,esn0,theory", [
    ("qpsk", 6.0, lambda e: q(np.sqrt(e))),            # per bit, Gray: Q(sqrt(Es/N0))
    ("16qam", 12.0, lambda e: 0.75*q(np.sqrt(e/5))),   # ~ (3/4) Q(sqrt(Es/(5 N0)))
])
def test_error_rate_matches_theory(name, esn0, theory, rng):
    m = rp.mod.load(name)
    bits = bits_for(m, 60000, rng)
    syms = (rp.Data(bits) * m).data
    noisy, _ = awgn(syms, esn0, rng)
    got = np.mean((rp.Data(noisy) / m).bits() != bits)
    want = theory(10**(esn0/10))
    assert got == pytest.approx(want, rel=0.2)

def test_llr_signs_match_bits_and_grow_with_snr(rng):
    m = rp.mod.load("16qam")
    bits = bits_for(m, 4000, rng)
    syms = (rp.Data(bits) * m).data
    lo, _ = awgn(syms, 8.0, rng)
    hi, _ = awgn(syms, 20.0, rng)
    l_lo, l_hi = (rp.Data(x) / m for x in (lo, hi))
    assert l_hi.is_soft
    assert np.mean((l_hi.data < 0) == bits) > 0.999
    assert np.mean(np.abs(l_hi.data)) > np.mean(np.abs(l_lo.data))

def test_generated_names_and_errors():
    assert len(rp.mod.load("16QAM").constellation) == 16
    assert len(rp.mod.load("QAM64").constellation) == 64
    assert len(rp.mod.PSK(16).constellation) == 16
    assert len(rp.mod.APSK(rings=(8, 8)).constellation) == 16
    with pytest.raises(rp.errors.ConfigError):
        rp.mod.load("nonsense")

def test_pipeline_with_a_generated_modulation(rng):
    ecc = rp.coding.load("block", "hamming/7_4")
    link = rp.scram.load("fdt", "v35") * ecc * rp.mod.QAM(16)
    raw = bytes(rng.integers(0, 256, 66).astype(np.uint8))
    assert (rp.Data(raw) * link / link).as_bytes().hex == raw.hex()

@pytest.mark.parametrize("M", [2, 4])
def test_fsk_round_trip_and_noise(M, rng):
    m = rp.mod.FSK(M, sps=8)
    bits = bits_for(m, 500, rng)
    wave = rp.Data(bits) * m
    assert wave.sps == 8 and len(wave) == 500*8
    assert np.allclose(np.abs(wave.data), 1.0, atol=1e-5)
    assert np.array_equal((wave / m).bits(), bits)
    noisy = wave.data + 0.3*(rng.normal(size=len(wave)) + 1j*rng.normal(size=len(wave))).astype(np.complex64)
    assert np.mean((rp.Data(noisy, sps=8) / m).bits() != bits) < 0.02

def test_fsk_stream_in_chunks_matches_one_shot(rng):
    m = rp.mod.FSK(2, sps=8)
    bits = bits_for(m, 400, rng)
    wave = (rp.Data(bits, flush=True) * m).data
    ctx, outs = None, []
    for a, b in ((0, 301), (301, 302), (302, 1717), (1717, len(wave))):
        out = rp.Data(wave[a:b], sps=8, ctx=ctx) / m
        ctx = out.ctx
        outs.append(out.data)
    assert np.array_equal(np.concatenate(outs) < 0, bits)

def test_a_modulation_starts_on_its_first_map_and_can_pick_another():
    m = rp.mod.load("QPSK")
    assert np.array_equal(m.constellation.mapping.arr, m.get_maps()[0].arr)
    by_index = rp.mod.load("QPSK", 2).constellation.mapping.arr
    by_word = rp.mod.load("QPSK", "R180").constellation.mapping.arr
    assert np.array_equal(by_index, by_word) and not np.array_equal(by_index, m.get_maps()[0].arr)
    assert np.array_equal(m.set_mapping([3, 2, 1, 0]).constellation.mapping.arr, [3, 2, 1, 0])
    with pytest.raises(ValueError):
        rp.mod.load("QPSK", "Grey") # Matches every map
    bits = np.random.default_rng(0).integers(0, 2, 200).astype(bool)
    for i in range(4):
        mm = rp.mod.load("QPSK", i)
        assert np.array_equal((rp.Data(bits) * mm / mm).bits(), bits)

@pytest.mark.parametrize("name", ["bpsk", "qpsk", "8psk", "16psk", "16qam", "32qam", "64qam", "4ask", "16apsk", "dvbs2_8psk"])
def test_every_generated_map_undoes_its_channel(name, rng):
    """A signal sent on the first map and turned (and spectrally inverted) by the channel decodes on the map that says so"""
    base = rp.mod.load(name)
    maps = list(base.get_maps())
    assert len({tuple(m.arr.tolist()) for m in maps}) == len(maps) # No duplicates
    assert sum(m.inverted for m in maps) in (0, len(maps)//2) or name in ("bpsk", "4ask")
    bits = rng.integers(0, 2, base.constellation.bits_per_symbol*400).astype(bool)
    tx = (rp.Data(bits) * base).data
    for i, m in enumerate(maps):
        rx = np.conj(tx) if m.inverted else tx
        rx = (rx*np.exp(1j*m.rotation)).astype(np.complex64)
        dem = rp.mod.load(name, i)
        assert np.array_equal((rp.Data(rx) / dem).bits(), bits), m.comment

def test_the_map_counts_and_names():
    q = rp.mod.load("QPSK")
    names = [m.comment for m in q.get_maps()]
    assert len(names) == 8 and names[0] == "R000 Grey Normal" and "R270 Grey Inverted" in names
    assert len(rp.mod.load("8PSK").get_maps()) == 16 and len(rp.mod.load("16QAM").get_maps()) == 8
    assert rp.mod.load("QPSK", "R090 Inverted").constellation.mapping.inverted
    assert not rp.mod.load("QPSK", "R090").constellation.mapping.inverted # Normal unless it says inverted
