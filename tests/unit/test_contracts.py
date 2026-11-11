"""Every category's contract (ARCHITECTURE.md, Contracts and Type tags), checked over every registered implementation"""
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.base.types import Category, Rep
from rpps.errors import StreamError

def defs(domain):
    """(category/group, name) of every definition in a config domain"""
    return [(cat, n) for cat, names in _config.ls(domain).items() for n in names]

def load(domain, cat, name):
    """By its full path: bare names can be ambiguous across categories"""
    return _config.load(domain, f"{cat}/{name}", example=True)

def data_bits(n=64, seed=0):
    return rp.Data(np.random.default_rng(seed).integers(0, 2, n).astype(bool))

def symbols(n=32):
    return rp.Data(np.exp(1j*np.pi/4*(2*np.arange(n) % 8 + 1)).astype(np.complex64), sps=1)

# ---- the tags rp.Data gives

def test_data_tags():
    assert rp.Data(b"abc").REP == Rep.DATA and rp.Data(np.zeros(8, bool)).REP == Rep.DATA
    assert rp.Data(np.zeros(8, np.complex64)).REP == Rep.SAMPLES # no sps said: samples
    assert rp.Data(np.zeros(8, np.complex64), sps=1).REP == Rep.SYMBOLS
    assert rp.Data(np.zeros(8, np.float32)).REP == Rep.SAMPLES and rp.Data(np.zeros(8, np.float32)).DT == rp.dtype.REAL

# ---- stages: signal, sync, wave

def test_stages_take_samples_and_symbols_only():
    with pytest.raises(StreamError):
        data_bits() + rp.signal.sample.down.Decimate(2)
    llr = symbols() / rp.symbol.load("QPSK")
    with pytest.raises(StreamError):
        llr + rp.signal.sample.down.Decimate(2)

@pytest.mark.parametrize("pkg,cat", [(rp.sync, Category.SYNC), (rp.wave, Category.WAVE)])
def test_stage_categories(pkg, cat):
    stages = [v for v in vars(pkg).values() if isinstance(v, type) and issubclass(v, rp.Stage)]
    assert stages and all(s.CATEGORY == cat for s in stages)
    out = rp.Data(np.ones(16, np.complex64), Fs=8.0, sps=8) + rp.wave.Stagger()
    assert out.CAT == Category.WAVE and out.REP == Rep.SAMPLES

# ---- symbol: every constellation

CONST = [n for cat, n in defs("mod") if cat.startswith("constellation")] + ["16QAM", "64QAM", "16APSK", "4ASK", "OOK", "8PSK"]

@pytest.mark.parametrize("name", CONST)
def test_symbol_contract(name):
    c = rp.symbol.load(name)
    syms = data_bits(c.bits_per_symbol*16) * c
    assert syms.REP == Rep.SYMBOLS and syms.CAT == Category.SYMBOL
    llr = rp.Data(syms.data, sps=1) / c
    assert llr.REP == Rep.CHANNEL and llr.DT == rp.dtype.LLR
    with pytest.raises(StreamError):
        llr / c # bits are not symbols
    with pytest.raises(StreamError):
        rp.Data(syms.data) / c # samples (no sps) are not symbols either

def test_differential_and_spread_contracts():
    with pytest.raises(StreamError):
        data_bits() * rp.symbol.Differential()
    barker = rp.symbol.Spread(rp.spread.load("barker11"))
    with pytest.raises(StreamError):
        data_bits() * barker # one sequence spreads symbols
    zig = rp.symbol.Spread(rp.spread.load("ieee802154_2450"))
    chips = data_bits(64) * zig
    assert chips.REP == Rep.SYMBOLS
    assert (chips / zig).REP == Rep.CHANNEL

# ---- code: every channel code and scrambler

CODES = [(c, n) for c, n in defs("coding")]

@pytest.mark.parametrize("cat,name", CODES)
def test_code_contract(cat, name):
    code = load("coding", cat, name)
    assert code.CATEGORY == Category.CODE
    assert code.takes(True) == {Rep.DATA, Rep.CHANNEL} and code.takes(False) == {Rep.CHANNEL, Rep.DATA}
    with pytest.raises(StreamError):
        symbols() / code

@pytest.mark.parametrize("cat,name", defs("scram"))
def test_scrambler_contract(cat, name):
    s = load("scram", cat, name)
    out = data_bits(8*204*8) * s
    assert out.REP == Rep.CHANNEL and out.CAT == Category.CODE
    assert (out / s).REP == Rep.DATA
    with pytest.raises(StreamError):
        symbols() * s

def test_code_steps_chain_across_the_decoder():
    ecc, scr = rp.code.load("block", "hamming/7_4"), rp.code.scram.load("fdt", "v35")
    tx = rp.Data(b"Hello World!") * scr * ecc
    assert tx.REP == Rep.CHANNEL
    rx = tx / ecc / scr
    assert rx.REP == Rep.DATA and rx.as_bytes().hex == rp.Data(b"Hello World!").hex

# ---- link: every CRC and frame format

@pytest.mark.parametrize("cat,name", defs("encap"))
def test_link_contract(cat, name):
    link = load("encap", cat, name)
    assert link.CATEGORY == Category.LINK
    assert link.takes(False) == {Rep.DATA} # channel bits must be decoded first
    llr = symbols() / rp.symbol.load("QPSK")
    with pytest.raises(StreamError):
        llr / link

def test_crc_round_trip_gives_packets():
    crc = rp.link.load("crc", "crc16_xmodem")
    framed = rp.Data(b"payload!") * crc
    assert framed.REP == Rep.DATA and framed.CAT == Category.LINK
    back = framed / crc
    assert back.REP == Rep.PACKETS

# ---- an uncoded link needs an explicit code step

def test_uncoded_link():
    qpsk, crc = rp.symbol.load("QPSK"), rp.link.load("crc", "crc16_xmodem")
    syms = rp.Data(b"no fec here") * crc * qpsk
    rx = rp.Data(syms.data, sps=1) / qpsk
    with pytest.raises(StreamError):
        rx / crc
    data = rx / rp.code.Uncoded()
    assert data.REP == Rep.DATA and data.DT == rp.dtype.LLR # soft stays soft
    out = data / crc
    assert out.REP == Rep.PACKETS
    assert out.as_bytes().hex == rp.Data(b"no fec here").hex
