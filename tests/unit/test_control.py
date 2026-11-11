"""Header-driven control (ARCHITECTURE.md, Header-driven control; Migration step 7)

Fields published with positions, the controls section and its tables, live / restart / none, positions carried between
representations, the re-run buffer with drop-and-report, and transmit fields from a schedule or the caller. Then the two
profiles that need it: DVB-S2 (MODCOD frame by frame, announced ahead) and 802.11b (PLCP rate, announced late).
"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from rpps.errors import ConfigError
from rpps.profile import control, runner as runner_mod
from rpps._config import format_json

KNOWN = {"timing": {"method": "known"}}

def bits(n, seed):
    return np.random.default_rng(seed).integers(0, 2, n).astype(bool)

def tmp_profile(tmp_path, obj, name="test/p"):
    d = tmp_path / "profile" / name.split("/")[0]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name.split('/')[1]}.json").write_text(format_json({"base": "profile", **obj}))
    rp.add_config_path(tmp_path)

@pytest.fixture
def config_path(tmp_path):
    yield tmp_path
    if tmp_path in rp._config._EXTRA:
        rp._config._EXTRA.remove(tmp_path)

# ---- tables, entries, position maps

def test_tables():
    t = rp.profile.table("dvbs2/modcod")
    assert t.row({"modcod": 12, "short": 0})["constellation"] == "dvbs2_8psk"
    assert t.row({"modcod": 12, "short": 0})["interleaver"] == "dvbs2_8psk_210" # 8PSK 3/5 reads its columns backwards
    assert t.row({"modcod": 4, "short": 0})["kbch"] == 32208
    assert t.row({"modcod": 4, "short": 1}) is None and t.row({"modcod": 4}) is None # no short frames; a key missing
    assert t.row({"modcod": 4, "short": False}) is not None # header bits as bools look the same
    assert rp.profile.table("ieee80211b/rate").row({"signal": 20})["constellation"] == "qpsk_axes"

def test_entry_values_take_fields_columns_and_calcs():
    e = control.Entry("pls", {"table": "dvbs2/modcod", "key": ["modcod", "short"],
                              "drive": {"map": {"name": "constellation"}, "pilots": {"pilots": "pilots"},
                                        "plscram": {"length": {"calc": "slots*90 + pilots*((slots - 1)//16)*36"}}}}, "t")
    v, row = e.values({"modcod": 4, "short": 0, "pilots": 1})
    assert v == {"map": {"name": "dvbs2_qpsk"}, "pilots": {"pilots": 1}, "plscram": {"length": 360*90 + 22*36}}
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
    assert ctl["map"] == {"name": "live", "mapping": "live"} and ctl["ldpc"]["name"] == "live"
    assert ctl["timing"] == {} # A sync tracker's own estimates: none
    conv = rp.profile.load("dvbs/qpsk_3_4").steps[5]
    assert conv.controls["name"] == "restart" # Viterbi path metrics belong to the old trellis
    forney = rp.profile.load("dvbs/qpsk_3_4").steps[7]
    assert forney.controls["name"] == "restart" # Its delay lines hold data interleaved the old way

def test_live_operators():
    c = rp.symbol.load("bpsk")
    c.control(name="qpsk_axes")
    assert c.bits_per_symbol == 2 and c.pos_rate(True) == 2 and len(c.points) == 4
    code = rp.code.load("block", "bch/dvbs2_1_2")
    code.control(name="bch/dvbs2_3_4")
    assert (code.e_num, code.e_den) == (48408, 48600)
    il = rp.code.load("interleave", "dvbs2_8psk")
    il.control(name=None) # nothing: QPSK has no bit interleaver
    x = rp.base.Mod(np.linspace(-1, 1, 30).astype(np.float32), rep=Rep.CHANNEL)
    assert np.array_equal((x/il).data, x.data)

def test_restart_rebuilds_and_reports(config_path):
    """A transmit schedule restarting a convolutional code each frame: rebuilt, reported, output as from a fresh one"""
    tmp_profile(config_path, {"steps": [{"code": "fec", "id": "cc", "type": "conv", "name": "k7_171_133/1_2"}],
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
    fresh = rp.code.load("conv", "k7_171_133/1_2")
    assert np.array_equal(outs[1], np.asarray((rp.Data(b) * fresh).data)) # the trellis started again

# ---- operators of the frame structure

def test_pl_header_codewords():
    h = rp.symbol.load("dvbs2_pl")
    for fields, word in [({"modcod": 4, "short": 0, "pilots": 1}, 0x2437D66306E87850),
                         ({"modcod": 12, "short": 0, "pilots": 0}, 0x7E6D8C395CB2220A)]:
        sym = h.encode_fields(fields)
        k = (np.arange(64) + 26) % 2
        b = np.real(sym[26:]*np.exp(-1j*np.deg2rad(45 + 90*k))) < 0
        assert int("".join("1" if x else "0" for x in b), 2) == word
        rng = np.random.default_rng(0)
        noisy = sym + 0.7*(rng.normal(size=90) + 1j*rng.normal(size=90))
        got, errors = h.decode_fields(noisy)
        assert got == fields and errors > 0 # soft decoding over the 128 words still finds it

def test_pl_scrambling_sequence():
    ps = rp.symbol.PhaseScramble(rp.code.scram.load("adt", "dvbs2_pl"), [131072, 0])
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
    h = rp.link.load("header", "ieee80211b_plcp")
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

# ---- 802.11b: the PLCP rate, announced late

def b_transmit(packets, rates):
    p = rp.profile.load("ieee80211b/long", sync=KNOWN)
    tx, s, waves = p.tx(), None, []
    for i, (psdu, r) in enumerate(zip(packets, rates)):
        d = rp.Data(psdu, Fs=1e6, sps=1, stream=s, flush=i == len(packets) - 1)
        s = d.stream
        o = tx(d, fields={"signal": r})
        waves.append(o.data)
    return p, np.concatenate(waves + [np.zeros(400, np.complex64)]), o

def b_receive(p, x, o, cuts=None, noise=0.0):
    if noise:
        x = (x + noise*(np.random.default_rng(9).normal(size=len(x)) + 1j*np.random.default_rng(10).normal(size=len(x)))).astype(np.complex64)
    rx, st, outs, i = p.rx(), None, [], 0
    cuts = cuts or [len(x)]
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=o.Fs, sps=4, delay=o.delay, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        outs.append(np.asarray(rx(d).data).astype(bool))
    return np.concatenate(outs), st

@pytest.fixture(scope="module")
def b_signal():
    packets = [bits(8*n, i) for i, n in enumerate([40, 25, 60, 33])]
    rates = [20, 10, 20, 20]
    p, x, o = b_transmit(packets, rates)
    return p, x, o, packets, rates

def test_80211b_rate_switches_mid_packet(b_signal):
    p, x, o, packets, rates = b_signal
    got, st = b_receive(p, x, o)
    ref = np.concatenate(packets)
    assert len(got) == len(ref)
    assert np.array_equal(got[:-2], ref[:-2]) # (the end of the stream cuts the last symbol's pulse tail)
    ctl = st.control
    heads = [f for _, f, _ in ctl.fields("plcp") if "signal" in f]
    assert [f["signal"] for f in heads] == rates and [f["bits"] for f in heads] == [len(b) for b in packets]
    assert ctl.rerun == len(packets) and not ctl.reports # each header re-ran its payload, nothing dropped
    assert st.est["ieee80211b_plcp"]["frames"] == 4 and st.est["ieee80211b_plcp"]["crc_failed"] == 0

def test_80211b_any_block_cut(b_signal):
    p, x, o, packets, rates = b_signal
    ref, _ = b_receive(p, x, o)
    for cuts in ([1000]*(len(x)//1000) + [len(x) % 1000], [37]*(len(x)//37) + [len(x) % 37], [5000, 3, 20000, len(x) - 25003]):
        got, st = b_receive(p, x, o, cuts)
        assert np.array_equal(got, ref), cuts[:3]
        assert not st.control.reports

def test_80211b_through_noise(b_signal):
    p, x, o, packets, rates = b_signal
    got, st = b_receive(p, x, o, noise=0.3)
    ref = np.concatenate(packets)
    assert len(got) == len(ref) and np.mean(got != ref) < 0.01 # each channel error is three after the descrambler

def test_too_late_is_dropped_and_reported(b_signal, monkeypatch):
    """A buffer too short for the header's reach: the change is applied from the oldest position still held, and the drop
    reported (which change, from where, how much was lost)"""
    p, x, o, packets, rates = b_signal
    monkeypatch.setattr(runner_mod, "MIN_SEGMENT", 4)
    p = rp.profile.load("ieee80211b/long", sync=KNOWN)
    p.controls["plcp"].reach = 1
    got, st = b_receive(p, x[:len(x)//3], o)
    drops = [r for r in st.control.reports if r["what"] == "dropped"]
    assert drops and drops[0]["step"] == "plcp" and drops[0]["lost"] > 0 and drops[0]["applied"] > drops[0]["at"] - 1000
    assert st.est["ieee80211b_plcp"]["frames"] >= 1 # and it went on

def test_80211b_transmit_fields_from_the_schedule():
    p = rp.profile.load("ieee80211b/long", signal=10)
    o = p.tx()(rp.Data(bits(80, 1), Fs=1e6, sps=1))
    assert [f["signal"] for _, f, _ in o.stream.control.fields("plcp") if "bits" in f] == [10] # the parameter's
    o = p.tx()(rp.Data(bits(80, 1), Fs=1e6, sps=1), fields={"signal": 20})
    assert [f["signal"] for _, f, _ in o.stream.control.fields("plcp") if "bits" in f] == [20] # the caller's

# ---- DVB-S2: the MODCOD frame by frame, announced ahead

FRAMES = [{"modcod": 4, "pilots": 1}, {"modcod": 12, "pilots": 1}, {"modcod": 1, "pilots": 0}]

@pytest.fixture(scope="module")
def s2_signal():
    p = rp.profile.load("dvbs2/normal", sync=KNOWN)
    t = rp.profile.table("dvbs2/modcod")
    dfl = [t.row({**f, "short": 0})["kbch"] - 80 for f in FRAMES]
    data = bits(sum(dfl), 4)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=FRAMES)
    x = np.concatenate((o.data, np.zeros(200, np.complex64)))
    return p, x, o, data, dfl

def s2_receive(p, x, o, cuts=None, noise=0.0, phase=0.7):
    rng = np.random.default_rng(5)
    x = (x*np.exp(1j*phase) + noise*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)
    rx, st, outs, i = p.rx(), None, [], 0
    cuts = cuts or [len(x)]
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=o.Fs, sps=4, delay=o.delay, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        outs.append(np.asarray(rx(d).data).astype(bool))
    return np.concatenate(outs), st

def test_dvbs2_transmit_frames(s2_signal):
    p, x, o, data, dfl = s2_signal
    heads = [(f["modcod"], f["pilots"], at) for _, f, at in o.stream.control.fields("pls")]
    assert heads == [(4, 1, 0), (12, 1, dfl[0]), (1, 0, dfl[0] + dfl[1])] # each frame's fields, where its data starts
    symbols = [90 + 360*90 + 22*36, 90 + 240*90 + 14*36, 90 + 360*90]
    assert len(o) == 4*sum(symbols)

def test_dvbs2_vcm_loopback(s2_signal):
    p, x, o, data, dfl = s2_signal
    got, st = s2_receive(p, x, o, noise=0.1) # about 11 dB Es/N0 and a carrier phase
    assert np.array_equal(got, data)
    assert [(f["modcod"], f["pilots"]) for _, f, _ in st.control.fields("pls")] == [(4, 1), (12, 1), (1, 0)]
    bb = [f for _, f, _ in st.control.fields("bbheader") if "dfl" in f]
    assert [f["dfl"] for f in bb] == dfl and all(f["ts_gs"] == 1 for f in bb)
    assert st.est["dvbs2_bb"]["crc_failed"] == 0 and st.control.rerun == 0 # announced ahead: nothing re-run

def test_dvbs2_any_block_cut(s2_signal):
    p, x, o, data, dfl = s2_signal
    got, _ = s2_receive(p, x, o, cuts=[30011]*(len(x)//30011) + [len(x) % 30011])
    assert np.array_equal(got, data)

def test_dvbs2_frames_found_at_low_snr(s2_signal):
    """Frame sync and the PLS hold well below what the data codes need: here about -3 dB Es/N0"""
    p, x, o, data, dfl = s2_signal
    got, st = s2_receive(p, x, o, noise=0.5)
    assert [(f["modcod"], f["pilots"]) for _, f, _ in st.control.fields("pls")] == [(4, 1), (12, 1), (1, 0)]
    assert len(got) == 0 or np.array_equal(got, data[:len(got)]) # nothing decodes wrong: a failed frame is dropped

def test_dvbs2_frame_cut_short(s2_signal):
    """A timing slip takes two symbols out of the first frame (as on air): it is lost and reported, and the next frames,
    which start at their own SOF, still decode: a codeword never carries over from one frame into the next"""
    p, x, o, data, dfl = s2_signal
    cut = 4*5000 # two symbols, inside the first frame's payload
    got, st = s2_receive(p, np.concatenate((x[:cut], x[cut + 8:])), o)
    assert np.array_equal(got, data[dfl[0]:])
    assert any(r["what"] == "cut short" for r in st.control.reports)

def test_dvbs2_from_timed_symbols(s2_signal):
    """A source of timed symbols with the carrier on: the profile from the frame sync"""
    p, x, o, data, dfl = s2_signal
    rx = p.rx(start=Rep.SYMBOLS)
    assert type(rx.ops[0][1]).__name__ == "FrameSync"
    syms = np.asarray((rp.Data(x, Fs=o.Fs, sps=4, delay=o.delay, flush=True) + p.rx().ops[0][1] + p.rx().ops[1][1]).data)
    out = rx(rp.Data((syms*np.exp(0.3j)).astype(np.complex64), sps=1, flush=True))
    assert np.array_equal(np.asarray(out.data).astype(bool), data)
