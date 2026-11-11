"""Generated constellations (PSK/QAM/APSK/ASK): round trips, power, soft decisions and error rates"""
import numpy as np
import pytest

import rpps as rp
from conftest import awgn
from rpps.symbol import shapes

NAMES = ["bpsk", "qpsk", "8psk", "16psk", "16qam", "32qam", "64qam", "128qam", "256qam", "4ask", "8ask", "ook", "16apsk", "32apsk"]

def bits_for(m, n_sym, rng):
    return rng.integers(0, 2, m.bits_per_symbol*n_sym).astype(bool)

@pytest.mark.parametrize("name", NAMES)
def test_round_trip_and_unit_power(name, rng):
    m = rp.symbol.load(name)
    assert len(set(m.mapping.arr.tolist())) == m.order
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
    m = rp.symbol.load(name)
    bits = bits_for(m, 60000, rng)
    syms = (rp.Data(bits) * m).data
    noisy, _ = awgn(syms, esn0, rng)
    got = np.mean((rp.Data(noisy, sps=1) / m).bits() != bits)
    want = theory(10**(esn0/10))
    assert got == pytest.approx(want, rel=0.2)

def test_llr_signs_match_bits_and_grow_with_snr(rng):
    m = rp.symbol.load("16qam")
    bits = bits_for(m, 4000, rng)
    syms = (rp.Data(bits) * m).data
    lo, _ = awgn(syms, 8.0, rng)
    hi, _ = awgn(syms, 20.0, rng)
    l_lo, l_hi = (rp.Data(x, sps=1) / m for x in (lo, hi))
    assert l_hi.is_soft
    assert np.mean((l_hi.data < 0) == bits) > 0.999
    assert np.mean(np.abs(l_hi.data)) > np.mean(np.abs(l_lo.data))

def test_generated_names_and_errors():
    assert rp.symbol.load("16QAM").order == 16
    assert rp.symbol.load("QAM64").order == 64
    assert rp.symbol.PSK(16).order == 16
    assert rp.symbol.APSK(rings=(8, 8)).order == 16
    with pytest.raises(rp.errors.ConfigError):
        rp.symbol.load("nonsense")

def test_pipeline_with_a_generated_modulation(rng):
    ecc = rp.code.load("block", "hamming/7_4")
    link = rp.code.scram.load("fdt", "v35") * ecc * rp.symbol.QAM(16)
    raw = bytes(rng.integers(0, 256, 66).astype(np.uint8))
    assert (rp.Data(raw) * link / link).as_bytes().hex == raw.hex()

def test_a_modulation_starts_on_its_first_map_and_can_pick_another():
    m = rp.symbol.load("QPSK")
    assert np.array_equal(m.mapping.arr, m.maps[0].arr)
    by_index = rp.symbol.load("QPSK", 2).mapping.arr
    by_word = rp.symbol.load("QPSK", "R180").mapping.arr
    assert np.array_equal(by_index, by_word) and not np.array_equal(by_index, m.maps[0].arr)
    assert np.array_equal(m.set_mapping([3, 2, 1, 0]).mapping.arr, [3, 2, 1, 0])
    with pytest.raises(ValueError):
        rp.symbol.load("QPSK", "Grey") # Matches every map
    bits = np.random.default_rng(0).integers(0, 2, 200).astype(bool)
    for i in range(4):
        mm = rp.symbol.load("QPSK", i)
        assert np.array_equal((rp.Data(bits) * mm / mm).bits(), bits)

@pytest.mark.parametrize("name", ["bpsk", "qpsk", "8psk", "16psk", "16qam", "32qam", "64qam", "4ask", "16apsk", "dvbs2_8psk"])
def test_every_generated_map_undoes_its_channel(name, rng):
    """A signal sent on the first map and turned (and spectrally inverted) by the channel decodes on the map that says so"""
    base = rp.symbol.load(name)
    maps = list(base.maps)
    assert len({tuple(m.arr.tolist()) for m in maps}) == len(maps) # No duplicates
    assert sum(m.inverted for m in maps) in (0, len(maps)//2) or name in ("bpsk", "4ask")
    bits = rng.integers(0, 2, base.bits_per_symbol*400).astype(bool)
    tx = (rp.Data(bits) * base).data
    for i, m in enumerate(maps):
        rx = np.conj(tx) if m.inverted else tx
        rx = (rx*np.exp(1j*m.rotation)).astype(np.complex64)
        dem = rp.symbol.load(name, i)
        assert np.array_equal((rp.Data(rx, sps=1) / dem).bits(), bits), m.comment

def test_the_map_counts_and_names():
    q = rp.symbol.load("QPSK")
    names = [m.comment for m in q.maps]
    assert len(names) == 8 and names[0] == "R000 Grey Normal" and "R270 Grey Inverted" in names
    assert len(rp.symbol.load("8PSK").maps) == 16 and len(rp.symbol.load("16QAM").maps) == 8
    assert rp.symbol.load("QPSK", "R090 Inverted").mapping.inverted
    assert not rp.symbol.load("QPSK", "R090").mapping.inverted # Normal unless it says inverted
