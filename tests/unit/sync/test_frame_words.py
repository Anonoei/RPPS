"""Frame sync among symbols by named words: alternatives, mid-burst words, real levels, Zadoff-Chu preambles, what a match
publishes; the symbol marker and puncture steps (rpps.sync.FrameSync, rpps.symbol.Marker, rpps.symbol.Puncture)"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.control import of

LEV = {1: 3.0, 0: 1.0, 2: -1.0, 3: -3.0}
DMR = {"bs_voice": "755FD7DF75F7", "bs_data": "DFF57D75DF5D", "ms_voice": "7F7D5DD57DFD", "ms_data": "D5D7F77FD757"}

def levels(h):
    b = format(int(h, 16), "048b")
    return np.array([LEV[int(b[i:i + 2], 2)] for i in range(0, 48, 2)])

def _run(stage, y, cuts, rng):
    idx = np.r_[0, np.sort(rng.choice(np.arange(1, len(y)), cuts, replace=False)), len(y)] if cuts else np.r_[0, len(y)]
    outs, pubs, stream = [], [], None
    for a, b in zip(idx[:-1], idx[1:]):
        kw = {} if stream is None else {"stream": stream}
        r = rp.Data(y[a:b], sps=1, flush=(b == len(y)), **kw) + stage
        stream = r.meta.stream
        outs.append(np.asarray(r.data))
        pubs += [p[1] for p in of(stream).published]
        of(stream).published.clear()
    return np.concatenate(outs), pubs

def _bursts(rng, gain, n=24):
    alts = {k: levels(v) for k, v in DMR.items()}
    names, seq, truth = list(alts), [rng.choice([-3, -1, 1, 3], 40)], []
    for _ in range(n):
        nm = names[rng.integers(len(names))]
        burst = np.concatenate((rng.choice([-3, -1, 1, 3], 54), alts[nm], rng.choice([-3, -1, 1, 3], 54)))
        seq += [rng.choice([-3, -1, 1, 3], 12), burst]
        truth.append((nm, burst))
    seq.append(rng.choice([-3, -1, 1, 3], 20))
    x = np.concatenate(seq)
    return alts, (gain*x + rng.normal(0, 0.1, len(x))).astype(np.float32), truth

@pytest.mark.parametrize("cuts", [0, 41])
def test_alternatives_mid_burst_real_levels(rng, cuts):
    """DMR-like: four words, voice and data each other inverted, the sync 54 symbols into a 132-symbol burst; real levels
    stay real, scaled back to the words' +-3; any cut gives the same"""
    alts, y, truth = _bursts(rng, 0.42)
    fs = rp.sync.FrameSync(alts, before=54, length=132, scale=True, polarity=False, threshold=0.8)
    out, pubs = _run(fs, y, cuts, np.random.default_rng(cuts))
    assert out.dtype == np.float32 and len(out) == 132*len(truth)
    assert [p["word"] for p in pubs] == [t[0] for t in truth]
    frames = out.reshape(len(truth), 132)
    sliced = np.array([-3, -1, 1, 3])[np.argmin(np.abs(frames[..., None] - np.array([-3, -1, 1, 3])), axis=-1)]
    assert np.array_equal(sliced, np.array([t[1] for t in truth]))
    assert all(abs(p["scale"] - 0.42) < 0.02 for p in pubs)

def test_polarity_true_confuses_inverted_words(rng):
    """With an unknown sense an inverted voice word is a data word: polarity=False is what tells them apart"""
    alts, y, truth = _bursts(rng, -0.42)
    out, pubs = _run(rp.sync.FrameSync(alts, before=54, length=132, threshold=0.8), y, 0, rng)
    inverse = {"bs_voice": "bs_data", "bs_data": "bs_voice", "ms_voice": "ms_data", "ms_data": "ms_voice"}
    assert all(p["word"] in (t[0], inverse[t[0]]) for p, t in zip(pubs, truth))

def test_esn0_is_the_frames(rng):
    """The published Es/N0 is the frame's symbols' power over the noise the preamble's residual measured (within 1 dB of
    the truth, not the preamble's own power: its +-3 levels are 2 dB over the frame's average), over seeds (measured:
    within -0.3 .. +0.9 dB)"""
    errs = []
    for seed in range(8):
        r = np.random.default_rng(seed)
        alts, y, truth = _bursts(r, 0.5)
        _, pubs = _run(rp.sync.FrameSync(alts, before=54, length=132, polarity=False, threshold=0.8), y, 0, r)
        power = (108*5 + 24*9)/132 # A frame's symbols: 108 of the four levels (5 on average) and the word's 24 at +-3
        errs.append(np.median([p["esn0"] for p in pubs]) - 10*np.log10(0.25*power/0.01))
    assert max(abs(e) for e in errs) < 1.0

def _zc_stream(rng, cfo, esn0=8.0, frames=16, payload=400):
    alts = {f"{u}_{s}": (rp.sequence.zadoff_chu(u, 63, s), {"root": u, "shift": s}) for u in (25, 29, 34) for s in (0, 21, 42)}
    qpsk = np.exp(1j*(np.pi/4 + np.pi/2*np.arange(4)))
    seq, truth = [qpsk[rng.integers(4, size=50)]], []
    for _ in range(frames):
        nm = list(alts)[rng.integers(len(alts))]
        seq += [alts[nm][0], qpsk[rng.integers(4, size=payload)]]
        truth.append(nm)
    x = np.concatenate(seq)
    y = x*np.exp(2j*np.pi*cfo*np.arange(len(x)) + 0.7j)
    y = y + (rng.normal(size=len(y)) + 1j*rng.normal(size=len(y)))*np.sqrt(10**(-esn0/10)/2)
    return alts, y.astype(np.complex64), truth

@pytest.mark.parametrize("cfo, segments", [(0.0, 1), (0.004, 1), (0.008, 4)]) # (each clean over 20 seeds)
def test_zadoff_chu_roots_shifts_and_carrier(rng, cfo, segments):
    alts, y, truth = _zc_stream(rng, cfo)
    _, pubs = _run(rp.sync.FrameSync(alts, segments=segments, threshold=0.5 if segments == 1 else 0.6), y, 0, rng)
    assert [p["word"] for p in pubs] == truth
    assert all((p["root"], p["shift"]) == (alts[t][1]["root"], alts[t][1]["shift"]) for p, t in zip(pubs, truth))
    assert abs(np.median([p["cfo"] for p in pubs]) - cfo) < 0.0005
    assert abs(np.median([p["esn0"] for p in pubs]) - 8.0) < 1.0

def test_segments_hold_a_word_a_carrier_turns(rng):
    """0.76 of a turn across a 63-symbol word: the coherent sum names few frames right, four segments nearly all
    (measured over 10 seeds: 2-9 of 16 against 14-16, at least 7 more)"""
    alts, y, truth = _zc_stream(rng, 0.012, esn0=10)
    _, whole = _run(rp.sync.FrameSync(alts, threshold=0.5), y, 0, rng)
    _, parts = _run(rp.sync.FrameSync(alts, segments=4, threshold=0.6), y, 0, rng)
    right = lambda pubs: sum(p.get("word") == t for p, t in zip(pubs, truth)) if len(pubs) == len(truth) else 0
    assert right(parts) >= len(truth) - 3 and right(parts) - right(whole) >= 4

def test_classify_names_the_constellation(rng):
    pts = {"qpsk": rp.symbol.load("psk/qpsk").points, "8psk": rp.symbol.load("psk/8psk").points}
    pre = rp.sequence.zadoff_chu(25, 63)
    seq, truth = [], []
    for k in range(8):
        name = "qpsk" if k % 2 else "8psk"
        p = pts[name]/np.sqrt(np.mean(np.abs(pts[name])**2))
        seq += [pre, p[rng.integers(len(p), size=300)]]
        truth.append(f"psk/{name}")
    y = np.concatenate(seq)
    y = (y + (rng.normal(size=len(y)) + 1j*rng.normal(size=len(y)))*np.sqrt(10**(-15/10)/2)).astype(np.complex64)
    fs = rp.sync.FrameSync(pre, length=363, classify=["psk/qpsk", "psk/8psk"], threshold=0.5)
    _, pubs = _run(fs, y, 0, rng)
    assert [p["modulation"] for p in pubs] == truth

def test_classify_tells_a_turned_map_by_the_words_phase(rng):
    """Behind a Zadoff-Chu word the phase is known: 8PSK on DVB-S2's turn (k*45 degrees) is told from 8PSK (22.5 + k*45),
    published as its map, and that map decodes the frame"""
    pre = rp.sequence.zadoff_chu(25, 63)
    plain, dvb = rp.symbol.load("psk/8psk"), rp.symbol.load("psk/8psk", "dvb-s2")
    seq, truth, sent = [], [], []
    for k in range(6):
        c = dvb if k % 2 else plain
        bits = rng.integers(0, 2, 900).astype(bool)
        seq += [pre, np.asarray((rp.Data(bits) * c).data)]
        truth.append(("psk/8psk", "DVB-S2" if k % 2 else None))
        sent.append(bits)
    y = np.concatenate(seq)*np.exp(0.7j) # (an unknown carrier phase: the word takes it off)
    y = (y + (rng.normal(size=len(y)) + 1j*rng.normal(size=len(y)))*np.sqrt(10**(-20/10)/2)).astype(np.complex64)
    fs = rp.sync.FrameSync(pre, length=363, classify=["psk/qpsk", "psk/8psk"], threshold=0.5)
    r = rp.Data(y, sps=1, flush=True) + fs
    pubs = [p[1] for p in of(r.meta.stream).published if "modulation" in p[1]]
    assert [(p["modulation"], p["mapping"]) for p in pubs] == truth
    frames = np.asarray(r.data).reshape(-1, 363)[:, 63:]
    for f, p, bits in zip(frames, pubs, sent):
        c = rp.symbol.load(p["modulation"], p["mapping"] or 0)
        assert np.array_equal(c.llr(f) < 0, bits)

def test_classify_keeps_the_last_when_a_frame_is_short(rng):
    pre = rp.sequence.zadoff_chu(25, 63)
    y = np.concatenate([pre, np.exp(1j*np.pi/4)*np.ones(20), pre, np.ones(20)]).astype(np.complex64)
    fs = rp.sync.FrameSync(pre, length=83, classify=["psk/qpsk", "psk/8psk"], threshold=0.5)
    r = rp.Data(y, sps=1, flush=True) + fs
    assert all("modulation" not in p[1] for p in of(r.meta.stream).published)
    assert any(x["what"] == "unclassified" for x in of(r.meta.stream).reports)

def test_zc_sequence_definition():
    s = rp.sequence.init("zc_test", {"base": "sequence", "type": "zc", "root": 25, "length": 63, "shift": 3})
    assert np.allclose(s.symbols, rp.sequence.zadoff_chu(25, 63, 3)) and s.fields == {"root": 25, "shift": 3}

def _profile(steps, controls=None):
    return rp._config.load_obj("profile", "test/frame_words", {"base": "profile", "steps": steps, "controls": controls or {}})

def test_symbol_marker_and_puncture_round_trip(rng):
    """tx: a Zadoff-Chu marker before frames of 70 symbols, a known symbol at 35 of every 36 (72 a frame); rx: the frame sync finds the
    marker, the frame drives the puncture and marker restarts, out come the symbols"""
    steps = [{"sync": "frame", "id": "fs", "zc": {"roots": [25], "length": 63}, "length": 63 + 72, "threshold": 0.5},
             {"symbol": "marker", "id": "pre", "zc": {"length": 63, "root": 25}},
             {"symbol": "puncture", "id": "status", "every": 36, "at": [35], "value": [1, 0]},
             {"symbol": "constellation", "id": "map", "name": "psk/qpsk"}]
    controls = {"fs": {"drive": {"pre": {"frame": "frame"}, "status": {"frame": "frame"}}},
                "pre": {"on": "tx", "frame": 140, "drive": {"pre": {"frame": True}, "status": {"frame": True}}}} # (a constant: a restart)
    p = _profile(steps, controls)
    data = rng.integers(0, 2, 140*5).astype(bool)
    sym = np.asarray(p.tx()(rp.Data(data, sps=1, flush=True)).data)
    assert len(sym) == 5*(63 + 72) and np.allclose(sym[63 + 35], 1) and np.allclose(sym[63 + 71], 1) # (70 data symbols, statuses at 35, 71)
    y = np.concatenate((np.zeros(30, np.complex64), sym*np.exp(0.8j)))
    out = rp.Data(y, sps=1, flush=True) + p.rx()
    assert np.array_equal(np.asarray(out.bits())[:len(data)], data)

def test_frame_step_refuses_both_kinds():
    with pytest.raises(rp.errors.ConfigError, match="not both"):
        _profile([{"sync": "frame", "sequence": "p25_fs", "zc": {"roots": [25], "length": 63}},
                  {"symbol": "constellation", "name": "ask/4level"}]).rx()

@pytest.mark.parametrize("cuts", [[], [7, 50], [1, 2, 3, 100]])
def test_puncture_any_cut(rng, cuts):
    p = _profile([{"symbol": "puncture", "id": "st", "every": 36, "at": [35], "value": [5, 0]}])
    x = rng.choice([-3, -1, 1, 3], 35*4 + 10).astype(np.complex64)
    whole = np.asarray(p.tx()(rp.Data(x, sps=1, flush=True)).data)
    assert len(whole) == 154 and np.allclose(whole[[35, 71, 107, 143]], 5)
    tx, rx, parts, back = p.tx(), p.rx(), [], []
    idx = [0] + cuts + [len(x)]
    for a, b in zip(idx[:-1], idx[1:]):
        parts.append(np.asarray(tx(rp.Data(x[a:b], sps=1, flush=(b == len(x)))).data))
    assert np.array_equal(np.concatenate(parts), whole)
    idx = [0] + cuts + [len(whole)]
    for a, b in zip(idx[:-1], idx[1:]):
        back.append(np.asarray((rp.Data(whole[a:b], sps=1, flush=(b == len(whole))) + rx).data))
    assert np.array_equal(np.concatenate(back), x)

def test_auto_segments_hold_a_long_word_a_burst_turns(rng):
    """A 64-symbol unique word under 0.01 cycles a symbol (5 kHz at 512 kBd: 0.64 of a turn across it): one coherent
    correlation keeps about 0.42 of its match and loses it, "auto" (4 parts) finds every one"""
    uw = {f"t{i}": (1 - 2*rng.integers(0, 2, 64)).astype(np.complex128) for i in range(3)}
    qpsk = np.exp(1j*(np.pi/4 + np.pi/2*np.arange(4)))
    seq, truth = [qpsk[rng.integers(4, size=40)]], []
    for _ in range(12):
        name = list(uw)[rng.integers(3)]
        seq += [uw[name], qpsk[rng.integers(4, size=500)]]
        truth.append(name)
    x = np.concatenate(seq)
    y = (x*np.exp(2j*np.pi*0.01*np.arange(len(x))) + 0.1*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)
    whole = rp.sync.FrameSync(uw, threshold=0.6)
    auto = rp.sync.FrameSync(uw, threshold=0.6, segments="auto")
    assert auto.segments == 4
    _, a = _run(auto, y, 0, rng)
    _, w = _run(whole, y, 0, rng)
    assert [p["word"] for p in a] == truth and len(w) < len(truth)//2

QPSK = np.exp(1j*(np.pi/4 + np.pi/2*np.arange(4)))

def _worded(rng, cfos, esn0, payload=1500):
    """Frames of a 26-symbol preamble, one of 16 words of 64 and QPSK data, each at its own carrier frequency (cycles a
    symbol, the phase carried on): (preamble, words, received, the frames sent)"""
    pre, words = QPSK[rng.integers(4, size=26)], QPSK[rng.integers(4, size=(16, 64))]
    seq, ph, frames = [QPSK[rng.integers(4, size=100)]], 0.0, []
    for f in cfos:
        x = np.concatenate((pre, words[rng.integers(16)], QPSK[rng.integers(4, size=payload)]))
        seq.append(x*np.exp(1j*(ph + 2*np.pi*f*np.arange(len(x)))))
        ph += 2*np.pi*f*len(x)
        frames.append(x)
    y = np.concatenate(seq)
    y = y + (rng.normal(size=len(y)) + 1j*rng.normal(size=len(y)))*np.sqrt(10**(-esn0/10)/2)
    return pre, words, y.astype(np.complex64), frames

def _phase_right(out, frames):
    """Frames out whose data sits on its points (the residual phase's spread under a radian)"""
    L = len(frames[0])
    return sum(float(np.std(np.angle(out[k*L + 90:(k + 1)*L]*np.conj(x[90:])))) < 1.0 for k, x in enumerate(frames[:-1]))

def test_a_tracked_frequency_holds_frames_a_preamble_alone_cannot(rng):
    """1e-3 cycles a symbol at 6 dB, 1.6 turns a frame: the line from one preamble's phase to the next wraps, the
    frequency tracked over the frames' known symbols doesn't (over 20 seeds: 9-11 of 11 frames against none, the
    median frequency within 1.5e-4)"""
    pre, words, y, frames = _worded(rng, [1e-3]*12, 6.0)
    tracked, pubs = _run(rp.sync.FrameSync(pre, words, cfo="track"), y, 0, rng)
    plain, _ = _run(rp.sync.FrameSync(pre, words), y, 0, rng)
    assert _phase_right(tracked, frames) >= 9 and _phase_right(plain, frames) <= 2
    assert abs(np.median([p["cfo"] for p in pubs[3:]]) - 1e-3) < 2.5e-4 # (its spread at 6 dB: some 7e-5)
    again, pubs2 = _run(rp.sync.FrameSync(pre, words, cfo="track"), y, 37, rng) # (however the stream is cut)
    assert np.allclose(again, tracked, atol=1e-5) and [p["cfo"] for p in pubs2] == pytest.approx([p["cfo"] for p in pubs])

def test_the_tracked_frequency_starts_again_when_the_offset_changes(rng):
    """A carrier step before acquiring mid-stream: 2e-3 then 0 cycles a symbol at 8 dB. The frames before are no
    evidence once a frame's own estimate is far off them: the next frames read near 0 (20 seeds: under 3.5e-4 from the
    second after; within 3e-4 of 2e-3 before), not the slow decay of an average"""
    pre, words, y, _ = _worded(rng, [2e-3]*6 + [0.0]*6, 8.0)
    _, pubs = _run(rp.sync.FrameSync(pre, words, cfo="track"), y, 0, rng)
    c = np.array([p["cfo"] for p in pubs])
    assert np.all(np.abs(c[1:6] - 2e-3) < 4.5e-4) and np.all(np.abs(c[7:]) < 5e-4)

def test_cfo_is_true_false_or_track():
    with pytest.raises(ValueError, match="track"):
        rp.sync.FrameSync(QPSK, cfo="tracked")
