"""Header-driven control (rpps.profile.control)

Fields published with positions, the controls section and its tables, live / restart / none, positions carried between
representations, the re-run buffer with drop-and-report, and transmit fields from a schedule or the caller. Then the two
profiles that need it: DVB-S2 (MODCOD frame by frame, announced ahead) and 802.11b (PLCP rate, announced late).
"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from rpps.errors import ConfigError
from rpps.profile import control
from rpps._config import format_json
from support.dvbs2 import seeded_bits as bits

def tmp_profile(tmp_path, obj, name="test/p"):
    d = tmp_path / "profile" / name.split("/")[0]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name.split('/')[1]}.json").write_text(format_json(obj))

@pytest.fixture
def config_path(tmp_path, config_layer):
    config_layer(tmp_path)
    return tmp_path

# ---- tables, entries, position maps

def test_tables():
    t = rp.profile.table("dvbs2/modcod")
    assert t.row({"modcod": 12, "short": 0})["constellation"] == "psk/8psk"
    assert t.row({"modcod": 19, "short": 0})["params"] == {"rate": "3/4"} # 16APSK's radii follow the code rate
    assert t.row({"modcod": 12, "short": 0})["interleaver"] == "dvbs2_8psk_210" # 8PSK 3/5 reads its columns backwards
    assert t.row({"modcod": 4, "short": 0})["kbch"] == 32208
    assert t.row({"modcod": 4, "short": 1})["kbch"] == 7032 # A short frame (table 5b)
    assert t.row({"modcod": 11, "short": 1}) is None and t.row({"modcod": 4}) is None # No short 9/10; a key missing
    assert t.row({"modcod": 4, "short": False}) is not None # header bits as bools look the same
    assert rp.profile.table("ieee80211b/rate").row({"signal": 20})["constellation"] == "qpsk_axes"

def test_entry_values_take_fields_columns_and_calcs():
    e = control.Entry("pls", {"table": "dvbs2/modcod", "key": ["modcod", "short"],
                              "drive": {"map": {"name": "constellation"}, "pilots": {"pilots": "pilots"},
                                        "plscram": {"length": {"calc": "slots*90 + pilots*((slots - 1)//16)*36"}}}}, "t")
    v, row = e.values({"modcod": 4, "short": 0, "pilots": 1})
    assert v == {"map": {"name": "psk/qpsk"}, "pilots": {"pilots": 1}, "plscram": {"length": 360*90 + 22*36}}
    v, row = e.values({"pilots": 0}) # Only what these fields give: no row, so no column or calc that needs one
    assert v == {"pilots": {"pilots": 0}} and row is None

def test_posmap():
    m = control.PosMap()
    m.add(0, 0, 1.0)
    m.add(100, 100, 2.0)  # a change: two bits per symbol from symbol 100
    m.add(150, 200, 1.0)  # and back
    assert m.fwd(50) == 50 and m.fwd(120) == 140 and m.fwd(160) == 210
    assert m.back(140) == 120 and m.back(210) == 160 and m.back(200) == 150
    m.cut(120)
    assert m.fwd(160) == 220

# ---- the controls section, checked by the loader

def test_none_is_refused(config_path):
    steps = [{"symbol": "differential"}, {"symbol": "constellation", "id": "map", "name": "bpsk"}]
    tmp_profile(config_path, {"steps": steps, "controls": {"map": {"drive": {"differential": {"anything": "x"}}}}}, "t/none")
    with pytest.raises(ConfigError, match="cannot take"):
        rp.profile.load("t/none")
    tmp_profile(config_path, {"steps": steps, "controls": {"map": {"drive": {"nosuch": {"name": "x"}}}}}, "t/nostep")
    with pytest.raises(ConfigError, match="no such step"):
        rp.profile.load("t/nostep")
    tmp_profile(config_path, {"steps": steps + [{"symbol": "differential"}],
                              "controls": {"map": {"drive": {"differential": {"x": "y"}}}}}, "t/twice")
    with pytest.raises(ConfigError, match="several steps"):
        rp.profile.load("t/twice")

def test_steps_declare_live_and_restart():
    p = rp.profile.load("dvbs2/normal")
    ctl = {s.id: s.controls for s in p.steps}
    assert ctl["map"] == {"name": "live", "mapping": "live", "params": "live"} and ctl["ldpc"]["name"] == "live"
    assert ctl["timing"] == {} # A sync tracker's own estimates: none
    conv = rp.profile.load("dvbs/qpsk_3_4").steps[5]
    assert conv.controls["name"] == "restart" # Viterbi path metrics belong to the old trellis
    forney = rp.profile.load("dvbs/qpsk_3_4").steps[7]
    assert forney.controls["name"] == "restart" # Its delay lines hold data interleaved the old way

def test_live_operators():
    c = rp.symbol.load("bpsk")
    c.control(name="qpsk_axes")
    assert c.bits_per_symbol == 2 and c.pos_rate(True) == 2 and len(c.points) == 4
    code = rp.code.load("block/bch/dvbs2_1_2")
    code.control(name="bch/dvbs2_3_4")
    assert (code.e_num, code.e_den) == (48408, 48600)
    il = rp.code.load("interleave/dvbs2_8psk")
    il.control(name=None) # nothing: QPSK has no bit interleaver
    x = rp.base.Mod(np.linspace(-1, 1, 30).astype(np.float32), rep=Rep.CHANNEL)
    assert np.array_equal((x/il).data, x.data)

def test_restart_rebuilds_and_reports(config_path):
    """A transmit schedule restarting a convolutional code each frame: rebuilt, reported, output as from a fresh one"""
    tmp_profile(config_path, {"steps": [{"code": "conv", "id": "cc", "name": "k7_171_133/1_2"}],
                              "controls": {"cc": {"schedule": [{"name": "k7_171_133/1_2"}], "drive": {"cc": {"name": "name"}}}}},
                "t/restart")
    p = rp.profile.load("t/restart")
    tx = p.tx()
    a, b = bits(64, 1), bits(64, 2)
    s = None
    outs = []
    for x in (a, b):
        d = rp.Data(x, stream=s)
        s = d.stream
        outs.append(np.asarray(tx(d).data))
    restarts = [r for r in s.control.reports if r["what"] == "restart"]
    assert len(restarts) == 2 and restarts[0]["step"] == "cc"
    fresh = rp.code.load("conv/k7_171_133/1_2")
    assert np.array_equal(outs[1], np.asarray((rp.Data(b) * fresh).data)) # the trellis started again

# ---- operators of the frame structure

def test_pl_header_codewords():
    h = rp.symbol.load_header("dvbs2_pl")
    for fields, word in [({"modcod": 4, "short": 0, "pilots": 1}, 0x2437D66306E87850),
                         ({"modcod": 12, "short": 0, "pilots": 0}, 0x7E6D8C395CB2220A)]:
        sym = h.encode_fields(fields)
        k = (np.arange(64) + 26) % 2
        b = np.real(sym[26:]*np.exp(-1j*np.deg2rad(45 + 90*k))) < 0
        assert int("".join("1" if x else "0" for x in b), 2) == word
        rng = np.random.default_rng(0)
        noisy = sym + 0.7*(rng.normal(size=90) + 1j*rng.normal(size=90))
        got, errors = h.decode_fields(noisy)
        assert got == {"s2x": 0, **fields} and errors > 0 # soft decoding over every word still finds it

def test_pl_header_reads_dvbs2x_modcods():
    """EN 302 307-2 5.5.2: b0 = 1 is a DVB-S2X MODCOD, coded with one more row (Figure 20) and sent a quarter turn on
    from the SOF. The word built here from the standard's own construction is what the header sends and reads"""
    h = rp.symbol.load_header("dvbs2_pl")
    G = ["10010000101011000010110111011101", "01010101010101010101010101010101", "00110011001100110011001100110011",
         "00001111000011110000111100001111", "00000000111111110000000011111111", "00000000000000001111111111111111", "1"*32]
    scr = "0111000110011101100000111100100101010011010000100010110111111010"
    for pls in (168, 182, 183): # 16APSK 13/18 normal; 32APSK 7/9 short; the same with pilots
        b = [(pls >> (7 - i)) & 1 for i in range(8)]
        y = np.zeros(32, int)
        for i in range(7):
            if b[i]:
                y ^= np.array([int(c) for c in G[i]])
        word = np.ravel(np.column_stack([y, y ^ b[7]])) ^ np.array([int(c) for c in scr])
        fields = {"s2x": 1, "modcod": (pls >> 2) & 31, "short": (pls >> 1) & 1, "pilots": pls & 1}
        sym = h.encode_fields(fields)
        k = (np.arange(64) + 26) % 2
        sent = np.real(sym[26:]*np.exp(-1j*np.deg2rad(45 + 90*k + 90))) < 0 # (the quarter turn)
        assert np.array_equal(sent.astype(int), word)
        rng = np.random.default_rng(pls)
        got, errors = h.decode_fields(sym + 0.7*(rng.normal(size=90) + 1j*rng.normal(size=90)))
        assert got == fields
    t = rp.profile.table("dvbs2/modcod")
    assert t.row({"modcod": 10, "short": 0}) is not None and t.row({"s2x": 1, "modcod": 10, "short": 0}) is None

def test_pl_scrambling_sequence():
    ps = rp.symbol.PhaseScramble(rp.sequence.load("dvbs2_pl"), [131072, 0])
    assert ps.R(20).tolist() == [0, 1, 1, 1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3, 1, 3, 1, 2] # EN 302 307 R_0(i)
    x = np.exp(2j*np.pi*np.random.default_rng(1).random(500)).astype(np.complex64)
    s = rp.Data(x, sps=1)
    assert np.allclose(((s * ps) / rp.symbol.PhaseScramble(ps.sequence, [131072, 0])).data, x, atol=1e-5)

def test_pilots_round_trip():
    p = rp.symbol.Pilots(slots=40, pilots=True)
    total, starts = p.layout()
    assert starts == [16*90, 32*90 + 36] and total == 40*90 + 72
    x = np.exp(1j*np.arange(40*90)).astype(np.complex64)
    y = rp.Data(x, sps=1) * p
    assert len(y) == total and np.allclose(y.data[16*90:16*90 + 36], (1 + 1j)/np.sqrt(2))
    assert np.allclose((y / rp.symbol.Pilots(slots=40)).data, x)

def test_link_header_round_trip():
    h = rp.link.load("header/ieee80211b_plcp")
    h.control(signal=20)
    out = rp.Data(bits(200, 3)) * h
    assert len(out) == 128 + 16 + 48 + 200
    fields, ok = h.parse(np.asarray(out.data)[144:192])
    assert ok and fields == {"signal": 20, "service": 0, "length": 100}
    bad = np.asarray(out.data)[144:192].copy()
    bad[5] ^= True
    assert not h.parse(bad)[1]

def test_delimiter_finds_frames_and_publishes():
    word = 0xF3A0
    d = rp.sync.Delimiter(word, 16, "lsb", name="sfd")
    w = ((word >> np.arange(16)) & 1).astype(bool)
    stream = np.concatenate((bits(50, 1), w, bits(30, 2), w, bits(20, 3)))
    out = rp.Data(stream, flush=True) + d
    log = out.stream.control.published
    assert [(f["frame"], at) for _, f, at, *_ in log] == [(0, 66), (1, 112)]
    assert np.array_equal(out.data, np.concatenate((stream[66:96], stream[112:])))
