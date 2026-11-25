"""Line codes, sync markers and words, delimiting with a fixed length and polarity, a header's frame check sequence,
streaming HDLC, spectral inversion, the peak AGC, the lazy channelizer, quasi-cyclic LDPC"""
import numpy as np
import pytest

import rpps as rp
from rpps.base import Mod
from rpps.base.types import Rep
from rpps.errors import ConfigError

def hard(d):
    x = np.asarray(d.data)
    return x < 0 if x.dtype.kind == "f" else x.astype(bool)

def as_data_bits(b, rep=Rep.DATA, **kw):
    d = rp.Data(np.asarray(b, dtype=bool), **kw)
    return d.derive(np.asarray(b, dtype=bool), rp.Category.CODE, rp.dtype.BITS, rep=rep)

def as_llr(d, scale=3.0):
    """Sure LLRs with the hard decisions of d"""
    return d.derive((1 - 2*np.asarray(d.data).astype(np.float32))*scale, rp.Category.SYMBOL, rp.dtype.LLR, Mod, rep=Rep.CHANNEL)

# ---- line codes

@pytest.mark.parametrize("name", rp.code.line.NAMES)
def test_line_code_round_trip_hard_and_soft(name, rng):
    b = rng.integers(0, 2, 300).astype(bool)
    L = rp.code.Line(name)
    y = rp.Data(b) * L
    assert y.REP == Rep.CHANNEL and len(y) == (2 if L.pairs else 1)*len(b)
    assert np.array_equal(hard(y / L), b)
    s = as_llr(y) / L
    assert s.DT == rp.dtype.LLR and s.REP == Rep.DATA and np.array_equal(hard(s), b)

@pytest.mark.parametrize("name", rp.code.line.NAMES)
def test_line_code_any_block_cut(name, rng):
    b = rng.integers(0, 2, 200).astype(bool)
    L = rp.code.Line(name)
    whole = hard(rp.Data(b) * L)
    for cuts in ([7, 1, 192], [3]*66 + [2]):
        outs, ctx, i = [], None, 0
        for c in cuts:
            d = rp.Data(b[i:i + c]) if ctx is None else rp.Data(b[i:i + c], ctx=ctx)
            o = d * L
            ctx, i = o.ctx, i + c
            outs.append(hard(o))
        assert np.array_equal(np.concatenate(outs), whole)
    coded = whole
    outs, ctx, i = [], None, 0
    for c in [5, 1, len(coded) - 6]:
        d = as_data_bits(coded[i:i + c], Rep.CHANNEL) if ctx is None else as_data_bits(coded[i:i + c], Rep.CHANNEL, ctx=ctx)
        o = d / L
        ctx, i = o.ctx, i + c
        outs.append(hard(o))
    assert np.array_equal(np.concatenate(outs), b)

def test_line_conventions():
    L = rp.code.Line("nrzi")
    assert np.array_equal(hard(rp.Data(np.array([1, 1, 0, 1, 0, 0], bool)) * L), [0, 0, 1, 1, 0, 1]) # 0: a transition
    L = rp.code.Line("nrzm")
    assert np.array_equal(hard(rp.Data(np.array([1, 1, 0, 1, 0, 0], bool)) * L), [1, 0, 0, 1, 1, 1]) # 1: a transition
    assert np.array_equal(hard(rp.Data(np.array([0, 1], bool)) * rp.code.Line("manchester")), [1, 0, 0, 1]) # IEEE: 0 -> 10
    assert np.array_equal(hard(rp.Data(np.array([0, 1], bool)) * rp.code.Line("thomas")), [0, 1, 1, 0])
    assert np.array_equal(hard(rp.Data(np.array([0, 1], bool)) * rp.code.Line("invert")), [1, 0])
    with pytest.raises(ValueError):
        rp.code.Line("ami")
    m = rp.code.Line("manchester")
    out = as_data_bits([1, 1, 0, 0, 1, 0], Rep.CHANNEL) / m
    assert list(out.failed) == [True, True, False] # 11 and 00 are not Manchester pairs: those bits failed
    soft = as_llr(as_data_bits([1, 1, 0, 0, 1, 0], Rep.CHANNEL)) / m
    assert list(soft.failed) == [True, True, False]

# ---- words and markers

def test_word_sequence():
    w = rp.sequence.load("ccsds_asm")
    assert w.n == 32 and w.value == 0x1ACFFC1D and list(w.bits[:8].astype(int)) == [0, 0, 0, 1, 1, 0, 1, 0]
    ble = rp.sequence.load("ble_adv_aa")
    assert ble.order == "lsb" and ble.n == 40 and list(ble.bits[:8].astype(int)) == [0, 1, 0, 1, 0, 1, 0, 1]
    assert rp.sequence.load("mode_s_preamble").value == 0xA140
    with pytest.raises(ConfigError):
        rp.sequence.word.word_bits("0x1FF", 8)

def test_marker_fixed_length_round_trip_and_errors(rng):
    frames = rng.integers(0, 2, 3*64).astype(bool)
    M = rp.code.Marker("0x1ACFFC1D", 32, length=64)
    coded = rp.Data(frames, flush=True) * M
    assert len(coded) == 3*96 and coded.REP == Rep.CHANNEL
    assert M.pos_rate(False) == 96/64 and M.pos_rate(True) == 64/96
    bits = np.asarray(coded.data).copy()
    bits[96 + 3] ^= True # one marker bit wrong in the second frame
    back = as_data_bits(bits, Rep.CHANNEL, flush=True) / M
    assert np.array_equal(hard(back), frames) and list(back.errors) == [0, 1, 0]
    soft = as_llr(coded) / M
    assert soft.DT == rp.dtype.LLR and np.array_equal(hard(soft), frames)

def test_delimiter_keeps_the_word_fixes_polarity_and_the_marker_strips_it(rng):
    w = rp.sequence.load("ccsds_asm")
    frames = rng.integers(0, 2, 4*64).astype(bool)
    M = rp.code.Marker(w.bits, length=64)
    coded = np.asarray((rp.Data(frames, flush=True) * M).data)
    stream = np.concatenate((rng.integers(0, 2, 20).astype(bool), ~coded, rng.integers(0, 2, 9).astype(bool))) # inverted
    D = rp.sync.Delimiter(w, length=64, keep=True, polarity=True)
    for cuts in ([len(stream)], [13, 100, 7, len(stream) - 120]):
        outs, st, i = [], None, 0
        for k, c in enumerate(cuts):
            d = as_data_bits(stream[i:i + c], Rep.CHANNEL, stream=st, flush=k == len(cuts) - 1)
            st, i = d.stream, i + c
            outs.append(hard(d + D))
        aligned = np.concatenate(outs)
        assert np.array_equal(aligned, coded) and st.est["delimiter"]["polarity"] == -1 and st.est["delimiter"]["frames"] == 4
    back = as_data_bits(aligned, Rep.CHANNEL, flush=True) / M
    assert np.array_equal(hard(back), frames) and not any(back.errors)

def test_delimiter_without_keep_gives_the_bits_after_the_word():
    bits = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1], bool) # the word 0100 at index 2, then 11; no other exact match
    out = as_data_bits(bits, Rep.CHANNEL, flush=True) + rp.sync.Delimiter(0b0100, 4, length=2)
    assert list(hard(out).astype(int)) == [1, 1]
    out = as_data_bits(bits, Rep.CHANNEL, flush=True) + rp.sync.Delimiter(0b0100, 4, length=2, errors=1)
    assert list(hard(out).astype(int)) == [1, 1, 1, 1] # with one error allowed, 0101 at index 8 too (then 11)

@pytest.mark.parametrize("order", ["msb", "lsb"])
def test_a_word_of_any_width(order, rng):
    """A 100-byte unique word: as a hex string, its bits and value round trip (no 64-bit limit)"""
    raw = rng.integers(0, 256, 100, dtype=np.uint8)
    w = rp.sequence.word.Word("uw", "0x" + raw.tobytes().hex(), 800, order)
    msb = np.unpackbits(raw).astype(bool)
    assert w.n == 800 and w.value == int(raw.tobytes().hex(), 16)
    assert np.array_equal(w.bits, msb[::-1] if order == "lsb" else msb)
    with pytest.raises(ConfigError):
        rp.sequence.word.word_bits("0x" + "ff"*101, 800)

def test_a_long_word_is_delimited_with_errors_across_any_block_cut(rng):
    """800 bits with 40 errors in every copy: matched by FFT correlation, exactly where the window-by-window count would,
    the inverted frames put right, whatever the block cut and as LLRs too"""
    w = rng.integers(0, 2, 800).astype(bool)
    frames = rng.integers(0, 2, 5*3000).astype(bool)
    M = rp.code.Marker(w, length=3000)
    coded = np.asarray((rp.Data(frames, flush=True) * M).data).copy()
    for k in range(5): # 40 bit errors in each word, the third frame inverted
        coded[k*3800 + rng.choice(800, 40, replace=False)] ^= True
    coded[2*3800:3*3800] ^= True
    stream = np.concatenate((rng.integers(0, 2, 333).astype(bool), coded))
    allowed = rp.sync.word_errors(800, 1e-9, 3800, polarity=True)
    assert 250 < allowed < 320
    D = rp.sync.Delimiter(w, length=3000, keep=True, polarity=True, errors=allowed)
    hard_bits = stream
    starts = [s for s, _ in D._starts(hard_bits, 0)]
    win = np.lib.stride_tricks.sliding_window_view(hard_bits, 800)
    bad = np.count_nonzero(win != w[None, :], axis=1)
    assert starts == sorted(set(np.nonzero(bad <= allowed)[0]) | set(np.nonzero(800 - bad <= allowed)[0]))
    assert starts == [333 + k*3800 for k in range(5)]
    for cuts, soft in (([len(stream)], False), ([1000, 7, 5000, len(stream)], True)):
        outs, st, i = [], None, 0
        for k, c in enumerate(cuts):
            d = as_data_bits(stream[i:i + c], Rep.CHANNEL, stream=st, flush=i + c >= len(stream))
            st, i = d.stream, i + c
            outs.append(hard((as_llr(d) if soft else d) + D))
            if i >= len(stream):
                break
        back = as_data_bits(np.concatenate(outs), Rep.CHANNEL, flush=True) / M
        assert np.array_equal(hard(back), frames) and list(back.errors) == [40]*5

def test_word_errors_bounds_random_matches(rng):
    """The count word_errors allows keeps random windows under the false-alarm rate, and one more would not"""
    e = rp.sync.word_errors(40, 1e-2)
    within = rng.integers(0, 2, (200000, 40)).sum(axis=1) # (errors against the all-zeros word)
    assert np.mean(within <= e) < 1e-2 < np.mean(within <= e + 1)
    assert rp.sync.word_errors(32, 1e-6, 8160) == -1 and rp.sync.word_errors(32, 1e-3, 8160) == 1

@pytest.mark.parametrize("turn", range(4))
def test_frame_sync_finds_a_bit_word_under_any_rotation(turn, rng):
    """A profile's frame step made from a 100-byte word, through its own QPSK mapping: found coherently, so a carrier
    left 90, 180 or 270 degrees off is put right by the word, and the marker after the mapping strips it"""
    uw = "0x" + rng.integers(0, 256, 100, dtype=np.uint8).tobytes().hex()
    p = rp._config.load_obj("profile", "test/uw", {"base": "profile", "steps": [
        {"sync": "frame", "id": "uw", "name": "uw", "word": uw, "bits": 800, "threshold": 0.3},
        {"symbol": "constellation", "name": "qpsk"},
        {"code": "marker", "word": uw, "bits": 800, "length": 4000}]})
    pay = rng.integers(0, 2, 4*4000).astype(bool)
    sy = np.asarray(p.tx()(rp.Data(pay, flush=True)).data)
    lead = (rng.normal(size=777) + 1j*rng.normal(size=777))/np.sqrt(2)
    y = np.concatenate((lead, sy))*np.exp(1j*(np.pi/2*turn + 0.3))
    y = y + 0.2*(rng.normal(size=len(y)) + 1j*rng.normal(size=len(y)))
    outs = list(rp.source.Memory(y.astype(np.complex64), sps=1)(1234) + p.rx())
    got = np.concatenate([hard(o) for o in outs])
    assert len(got) == len(pay) and np.mean(got != pay) < 1e-3

def test_frame_step_takes_a_header_or_a_word():
    for steps in ([{"sync": "frame"}], [{"sync": "frame", "header": "dvbs2_pl", "word": "0x1", "bits": 1}]):
        with pytest.raises(ConfigError, match="one of the two"):
            rp._config.load_obj("profile", "test/bad", {"base": "profile", "steps": steps}).rx()

def test_delimiter_frames_by_control_strip(rng):
    """Frames of no fixed length: a profile's frame control tells the marker where each starts"""
    w = rp.sequence.load("mode_s_preamble")
    M = rp.code.Marker(w.bits)
    assert M.pos_rate(True) == 1.0
    a, b = rng.integers(0, 2, 40).astype(bool), rng.integers(0, 2, 24).astype(bool)
    ctx = None
    outs = []
    for part in (a, b):
        M.control(frame=1)
        d = rp.Data(part) if ctx is None else rp.Data(part, ctx=ctx)
        o = d * M
        ctx = o.ctx
        outs.append(hard(o))
    assert len(outs[0]) == 56 and len(outs[1]) == 40 and np.array_equal(outs[0][:16], w.bits) and np.array_equal(outs[1][16:], b)
    M2 = rp.code.Marker(w.bits)
    M2.control(frame=0)
    back = as_data_bits(outs[0], Rep.CHANNEL) / M2
    assert np.array_equal(hard(back), a) and list(back.errors) == [0]

# ---- link: a frame check sequence after the payload, and streaming HDLC

def test_header_fcs_round_trip_and_rejects_a_damaged_frame(rng):
    msg = rng.integers(0, 2, 8*20).astype(bool)
    tx = rp.Data(msg) * rp.link.load("header/ble_adv")
    assert len(tx) == 16 + 160 + 24
    out = as_data_bits(np.asarray(tx.data)) / rp.link.load("header/ble_adv")
    assert np.array_equal(hard(out), msg) and out.stream.est["ble_adv"]["last"]["length"] == 20
    bad = np.asarray(tx.data).copy()
    bad[16 + 50] ^= True
    out = as_data_bits(bad) / rp.link.load("header/ble_adv")
    assert len(out) == 0 and out.stream.est["ble_adv"]["fcs_failed"] == 1 and out.stream.est["ble_adv"]["frames"] == 0

def test_mode_s_header_matches_a_real_frame():
    """A real DF17 extended squitter (ICAO address 4840D6): the header reads its DF, checks its parity and gives the 83 bits between"""
    frame = bytes.fromhex("8D4840D6202CC371C32CE0576098")
    bits = np.unpackbits(np.frombuffer(frame, np.uint8)).astype(bool)
    out = as_data_bits(bits) / rp.link.load("header/mode_s")
    assert out.stream.est["mode_s"]["last"] == {"df": 17} and len(out) == 83
    assert np.array_equal(hard(out), bits[5:88])

@pytest.mark.parametrize("name", ["hdlc", "hdlc_nrzi"])
def test_hdlc_streams_frames_across_any_block_cut(name, rng):
    fr = rp.link.load(f"frame/{name}")
    pk = [bytes(rng.integers(0, 256, n).astype(np.uint8)) for n in (10, 33, 100, 12)]
    bits, ctx = [], None
    for p in pk:
        d = rp.Data(p) if ctx is None else rp.Data(p, ctx=ctx)
        o = d * fr
        ctx = o.ctx
        bits.append(np.asarray(o.data))
    stream = np.concatenate([rng.integers(0, 2, 37).astype(bool)] + bits + [rng.integers(0, 2, 51).astype(bool)])
    ref = np.unpackbits(np.frombuffer(b"".join(pk), np.uint8)).astype(bool)
    for cuts in ([len(stream)], [5]*(len(stream)//5) + [len(stream) % 5], [1000, 3, len(stream) - 1003]):
        outs, ctx, i = [], None, 0
        for k, c in enumerate(cuts):
            seg = stream[i:i + c]
            d = as_data_bits(seg, flush=k == len(cuts) - 1) if ctx is None else as_data_bits(seg, ctx=ctx, flush=k == len(cuts) - 1)
            o = d / fr
            ctx, i = o.ctx, i + c
            outs.append(hard(o))
        assert np.array_equal(np.concatenate(outs), ref), cuts[:3]

# ---- signal and sync additions

def test_conjugate_inverts_the_spectrum():
    x = np.exp(2j*np.pi*0.1*np.arange(64)).astype(np.complex64)
    out = rp.Data(x, Fs=1.0) + rp.signal.Conjugate()
    assert out.REP == Rep.SAMPLES and np.allclose(np.asarray(out.data), np.conj(x))
    assert np.argmax(np.abs(np.fft.fft(np.asarray(out.data)))) == 64 - np.argmax(np.abs(np.fft.fft(x)))

def test_peak_agc_scales_each_burst_by_its_own_peak():
    x = np.concatenate((np.zeros(10), 0.3*np.array([1, 0, 1, 1, 0]*4), np.zeros(200), 2.0*np.array([1, 1, 0, 1, 0]*4))).astype(np.float32)
    out = np.asarray((rp.Data(x, Fs=1.0) + rp.sync.PeakAGC(target=1.0, decay=0.999)).data)
    assert np.allclose(out[10:30][x[10:30] > 0], 1.0, atol=1e-5) and np.allclose(out[-20:][x[-20:] > 0], 1.0, atol=0.03)
    assert np.allclose(out[x == 0], 0.0)

def test_channelize_stage_is_designed_from_the_stream_and_streams(rng):
    x = (np.exp(2j*np.pi*0.1*np.arange(60000)) + 0.1*(rng.normal(size=60000) + 1j*rng.normal(size=60000))).astype(np.complex64)
    st = rp.signal.Channelize(freq=200e3, width=50e3)
    whole = rp.Data(x, Fs=2e6, flush=True) + st
    assert 100e3 <= whole.Fs <= 200e3 and whole.REP == Rep.SAMPLES and abs(whole.cf - 200e3) < 1
    y = np.asarray(whole.data)
    f = np.fft.fftfreq(len(y), 1/whole.Fs)[np.argmax(np.abs(np.fft.fft(y)))]
    assert abs(f - (0.1*2e6 - 200e3)) < 2*whole.Fs/len(y)
    outs, s = [], None
    for a, b in ((0, 1001), (1001, 1002), (1002, 60000)):
        d = rp.Data(x[a:b], Fs=2e6, stream=s, flush=b == 60000)
        s = d.stream
        outs.append(np.asarray((d + st).data))
    cut = np.concatenate(outs)
    n = min(len(cut), len(y))
    assert np.allclose(cut[:n], y[:n], atol=1e-4)

def test_fm_by_deviation_round_trips_audio():
    t = np.arange(4000)/48e3
    audio = (0.7*np.sin(2*np.pi*440*t)).astype(np.float32)
    wave = rp.Data(audio, Fs=48e3) + rp.wave.FM(deviation=5e3)
    back = rp.Data(np.asarray(wave.data), Fs=48e3) + rp.wave.FMDemod(deviation=5e3)
    assert back.DT == rp.dtype.REAL and np.allclose(np.asarray(back.data)[1:], audio[1:], atol=1e-3)

def test_am_envelope_can_keep_its_mean():
    x = np.array([0, 1, 1, 0, 1, 0, 0, 0], np.float32)*0.5 + 0j
    env = np.asarray((rp.Data(x.astype(np.complex64), Fs=1.0) + rp.wave.AMDemod(alpha=None)).data)
    assert np.allclose(env, np.abs(x))

# ---- a quasi-cyclic LDPC code from its base matrix

@pytest.mark.parametrize("rate,k", [("1_2", 324), ("2_3", 432), ("3_4", 486), ("5_6", 540)])
def test_ieee80211n_ldpc_from_shifts(rate, k, rng):
    c = rp.code.load(f"block/ldpc/ieee80211n_648_{rate}")
    assert c.encoder.num == k and c.encoder.den == 648
    msg = rng.integers(0, 2, 2*k).astype(bool)
    enc = rp.Data(msg) * c
    assert len(enc) == 2*648
    assert not c.decoder.syndrome(np.asarray(enc.data).reshape(2, 648)).any() # every check holds
    llr = (1 - 2*np.asarray(enc.data).astype(np.float32))*2.0 + rng.normal(size=len(enc)).astype(np.float32)*0.8
    dec = enc.derive(llr, rp.Category.SYMBOL, rp.dtype.LLR, Mod, rep=Rep.CHANNEL) / c
    assert np.array_equal(hard(dec), msg) and (dec.errors >= 0).all()

def test_qc_checks_lift_shifts():
    from rpps.code.types.ldpc import qc_checks
    n, checks = qc_checks([[0, -1, 1], [2, 0, -1]], 3)
    assert n == 9 and len(checks) == 6
    assert checks[0].tolist() == [0, 7] and checks[1].tolist() == [1, 8] and checks[2].tolist() == [2, 6] # shift 1: row r -> column (r + 1) % 3
    assert checks[3].tolist() == [2, 3] # shift 2 of block 0: row 0 -> column 2

@pytest.mark.parametrize("name", ["biphase_m", "biphase_s", "miller"])
def test_irig_line_codes(name, rng):
    import rpps as rp
    x = rng.integers(0, 2, 400).astype(bool)
    line = rp.code.Line(name)
    ctx, parts = None, []
    for a, b in ((0, 3), (3, 3), (3, 250), (250, 400)): # any block cut
        o = (rp.Data(x[a:b]) if ctx is None else rp.Data(x[a:b], ctx=ctx)) * line
        ctx = o.ctx
        parts.append(np.asarray(o.data, bool))
    y = np.concatenate(parts)
    if name.startswith("biphase"):
        assert all(y[2*i - 1] != y[2*i] for i in range(1, 400)) # a transition at every bit start
    else: # Miller: never more than two bits without a transition
        runs = np.diff(np.flatnonzero(np.diff(y.astype(int)) != 0))
        assert runs.max() <= 4
    assert np.array_equal(np.asarray((rp.Data(y) / line).data, bool), x)
    assert np.array_equal(np.asarray((rp.Data(~y) / line).data, bool), x) # an inverted line decodes the same

@pytest.mark.parametrize("name", ["biphase_m", "biphase_s", "miller"])
def test_two_level_codes_without_a_loop_are_the_state_machine(name, rng):
    """The level as a running parity of the transitions is the state machine stepped a bit at a time"""
    from rpps.code.line import MID
    def machine(x, level=False, prev=True):
        y = []
        for b in x:
            if name != "miller" or (not b and not prev):
                level = not level
            y.append(level)
            if bool(b) == MID[name]:
                level = not level
            y.append(level)
            prev = bool(b)
        return np.array(y, bool)
    b = rng.integers(0, 2, 3001).astype(bool)
    L = rp.code.Line(name)
    outs, ctx, i = [], None, 0
    for c in [1, 0, 2, 500, 2498]:
        d = rp.Data(b[i:i + c]) if ctx is None else rp.Data(b[i:i + c], ctx=ctx)
        o = d * L
        ctx, i = o.ctx, i + c
        outs.append(hard(o))
    assert np.array_equal(np.concatenate(outs), machine(b))
