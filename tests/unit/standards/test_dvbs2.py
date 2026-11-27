"""DVB-S2 through its profile: MODCOD frame by frame, and multiple input streams"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from support.dvbs2 import KNOWN, MIXED_FRAMES as FRAMES, seeded_bits as bits

# ---- DVB-S2: the MODCOD frame by frame, announced ahead

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
    ntaps = next(s.ntaps for s in o.stream.states.values() if hasattr(s, "ntaps")) # the shaping filter's (its stream state)
    assert len(o) == 4*sum(symbols) + 4*(-(-(ntaps - 1)//4)) # and the last pulses' tail, whole symbols of it

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
    assert [type(op).__name__ for _, op in rx][:3] == ["Inversion", "MPower", "FrameSync"] # no wave, no timing
    syms = np.asarray((rp.Data(x, Fs=o.Fs, sps=4, delay=o.delay, flush=True) + p.rx().ops[0][1] + p.rx().ops[1][1]).data)
    out = rx(rp.Data((syms*np.exp(0.3j)).astype(np.complex64), sps=1, flush=True))
    assert np.array_equal(np.asarray(out.data).astype(bool), data)

@pytest.mark.parametrize("modcod", [18, 21, 23, 24, 26, 28]) # 16APSK 2/3, 5/6, 9/10; 32APSK 3/4, 5/6, 9/10
def test_dvbs2_apsk_loopback(modcod):
    """16APSK and 32APSK (EN 302 307-1 5.4.3-5.4.4): each rate's own ring ratios, the bit interleaver and the mapping"""
    p = rp.profile.load("dvbs2/normal", sync=KNOWN)
    t = rp.profile.table("dvbs2/modcod")
    frames = [{"modcod": modcod, "pilots": 1}, {"modcod": 4, "pilots": 1}] # Then QPSK: the receiver switches back
    dfl = [t.row({**f, "short": 0})["kbch"] - 80 for f in frames]
    data = bits(sum(dfl), modcod)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=frames)
    x = np.concatenate((o.data, np.zeros(200, np.complex64)))
    got, st = s2_receive(p, x, o, noise=0.02) # about 25 dB Es/N0: 32APSK 9/10 needs about 16
    assert np.array_equal(got, data)
    assert [f["modcod"] for _, f, _ in st.control.fields("pls")] == [modcod, 4]
    assert st.est["dvbs2_bb"]["crc_failed"] == 0

@pytest.mark.parametrize("modcod", [1, 4, 12, 14, 18, 22, 24, 27]) # Short: QPSK 1/4, 1/2; 8PSK 3/5, 3/4; 16APSK 2/3, 8/9; 32APSK 3/4, 8/9
def test_dvbs2_short_frame_loopback(modcod):
    """Short FECFRAMEs (16200 bits, EN 302 307-1 annex C and tables 5b, 6b, 7b), between normal ones"""
    p = rp.profile.load("dvbs2/normal", sync=KNOWN)
    t = rp.profile.table("dvbs2/modcod")
    frames = [{"modcod": modcod, "pilots": 1, "short": 1}, {"modcod": 4, "pilots": 0}, {"modcod": modcod, "pilots": 0, "short": 1}]
    dfl = [t.row({"short": 0, **f})["kbch"] - 80 for f in frames]
    data = bits(sum(dfl), 100 + modcod)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=frames)
    x = np.concatenate((o.data, np.zeros(200, np.complex64)))
    got, st = s2_receive(p, x, o, noise=0.02)
    assert np.array_equal(got, data)
    assert [(f["modcod"], f["short"]) for _, f, _ in st.control.fields("pls")] == [(modcod, 1), (4, 0), (modcod, 1)]
    assert st.est["dvbs2_bb"]["crc_failed"] == 0

# ---- DVB-S2 multiple input streams: the trunk's PLS drives every branch's BBHEADER

MIS_FRAMES = [{"modcod": 4, "pilots": 1, "sis": 0, "isi": 0}, {"modcod": 12, "pilots": 1, "sis": 0, "isi": 1},
              {"modcod": 1, "pilots": 0, "sis": 0, "isi": 0}, {"modcod": 4, "pilots": 1, "sis": 0, "isi": 1}]

@pytest.mark.parametrize("block", [None, 30011])
def test_dvbs2_multistream_branches(block):
    """Frames of two input streams (ISI 0, 1), sent by dvbs2/normal: dvbs2/mis gives each stream its own data fields. The
    BBFRAME length each branch's BBHEADER needs comes from the PLS in the trunk, frame by frame"""
    p = rp.profile.load("dvbs2/normal", sync=KNOWN)
    t = rp.profile.table("dvbs2/modcod")
    dfl = [t.row({**f, "short": 0})["kbch"] - 80 for f in MIS_FRAMES]
    data = bits(sum(dfl), 9)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=MIS_FRAMES)
    x = (np.concatenate((o.data, np.zeros(200, np.complex64)))*np.exp(0.7j)).astype(np.complex64)
    parts = np.split(data, np.cumsum(dfl)[:-1])
    rx, st, got, i = rp.profile.load("dvbs2/mis", sync=KNOWN).rx(), None, {"a": [], "b": []}, 0
    cuts = [len(x)] if block is None else [block]*(len(x)//block) + [len(x) % block]
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=o.Fs, sps=4, delay=o.delay, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        out = rx(d)
        for s in got:
            got[s].append(np.asarray(out[s].data).astype(bool))
    assert np.array_equal(np.concatenate(got["a"]), np.concatenate((parts[0], parts[2])))
    assert np.array_equal(np.concatenate(got["b"]), np.concatenate((parts[1], parts[3])))
    for s in got:
        e = out[s].stream.est["dvbs2_bb"]
        assert (e["frames"], e["skipped"], e["crc_failed"]) == (2, 2, 0)
