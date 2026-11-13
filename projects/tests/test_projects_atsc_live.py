"""The live ATSC monitor's parts: the batched Reed-Solomon decoder (atsc_decode.rs_decode, against the Python rp.code.RS), the native Viterbi (against the Python one), PSI tables, the stitching clock and the channel plan"""
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("pysdrlib")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401
import atsc_decode  # noqa: E402
import atsc_live  # noqa: E402
import rpps as rp  # noqa: E402
from rpps import _accel  # noqa: E402

RNG = np.random.default_rng(7)


def codewords(n):
    """n valid ATSC RS(207, 187) codewords from rp.code.RS's encoder"""
    enc = rp.code.RS(207, 187)._dec
    msg = RNG.integers(0, 256, (n, 187), dtype=np.uint8)
    return np.array([np.asarray(enc.encode(m.astype(np.int64), symbols=True)).astype(np.uint8) for m in msg]).reshape(n, 207), msg


def test_rs_clean_codewords_pass():
    cw, msg = codewords(20)
    data, err = atsc_decode.rs_decode(cw)
    assert (err == 0).all() and (data == msg).all()


@pytest.mark.parametrize("nerr", [1, 5, 10])
def test_rs_corrects_up_to_ten_bytes(nerr):
    cw, msg = codewords(40)
    bad = cw.copy()
    for i in range(len(bad)):
        pos = RNG.choice(207, nerr, replace=False)
        bad[i, pos] ^= RNG.integers(1, 256, nerr, dtype=np.uint8) # includes errors in the parity bytes
    data, err = atsc_decode.rs_decode(bad)
    assert (err == nerr).all() and (data == msg).all()


def test_rs_eleven_errors_are_not_miscorrected_silently():
    cw, msg = codewords(60)
    bad = cw.copy()
    for i in range(len(bad)):
        pos = RNG.choice(207, 11, replace=False)
        bad[i, pos] ^= RNG.integers(1, 256, 11, dtype=np.uint8)
    data, err = atsc_decode.rs_decode(bad)
    assert (err == -1).mean() > 0.95 # beyond t a decoder may land on another codeword (~1e-6 per word), almost never
    assert (data[err == -1] == bad[err == -1][:, :187]).all() # an uncorrectable word comes back as received


def test_rs_matches_rp_on_mixed_damage():
    cw, _ = codewords(30)
    bad = cw.copy()
    for i in range(len(bad)):
        k = int(RNG.integers(0, 14))
        pos = RNG.choice(207, k, replace=False)
        bad[i, pos] ^= RNG.integers(1, 256, k, dtype=np.uint8)
    d0, e0 = atsc_decode.rs_decode(bad)
    ref = rp.code.RS(207, 187)._dec
    saved = _accel.native
    _accel.native = None # the Python reference loop
    try:
        d1 = np.asarray(ref.decode(bad.reshape(-1).astype(np.int64), symbols=True)).reshape(-1, 187).astype(np.uint8); e1 = np.asarray(ref.errors)
    finally:
        _accel.native = saved
    assert (e0 == e1).all() and (d0[e0 >= 0] == d1[e0 >= 0]).all()


def test_native_viterbi_matches_python():
    if not _accel.available():
        pytest.skip("rpps._native is not built")
    tcm = atsc_decode._TCM
    x = RNG.uniform(-7, 7, (24, 600))
    a = tcm.decode_batch(x)
    saved = _accel.native
    _accel.native = None
    try:
        b = tcm.decode_batch(x)
    finally:
        _accel.native = saved
    assert (a == b).all()


def test_viterbi_recovers_a_noisy_trellis_path():
    tcm = atsc_decode._TCM
    bits = RNG.integers(0, 2, (8, 400)).astype(bool)
    tx = np.array([tcm.modulate(tcm.encode(b)) for b in bits]) # amplitudes the transmitter would send
    rx = tx + RNG.normal(0, 0.45, tx.shape)
    got = tcm.decode_batch(rx)
    assert (got == bits).mean() > 0.995


def test_channel_plan():
    assert atsc_live.lower_edge(14) == 470 and atsc_live.lower_edge(35) == 596 and atsc_live.lower_edge(7) == 174 and atsc_live.lower_edge(11) == 198
    assert atsc_live.centre(35) == 604e6
    assert abs(atsc_live.FIELD_S - 0.0241980) < 1e-6 and atsc_live.FIELD_CW == 312


def make_ts(packets):
    out = np.zeros((len(packets), 188), np.uint8)
    for i, (pid, payload) in enumerate(packets):
        out[i, 0] = 0x47; out[i, 1] = 0x40 | (pid >> 8); out[i, 2] = pid & 0xFF; out[i, 3] = 0x10
        out[i, 4:4 + len(payload)] = list(payload); out[i, 4 + len(payload):] = 0xFF
    return out


def section(table_id, body):
    sec = bytes([table_id]) + (0xB000 | (len(body) + 4)).to_bytes(2, "big") + body
    crc = rp.link.load("crc", "crc32_mpeg2")._enc
    c = int(crc.compute(sec)) if not hasattr(crc.compute(sec), "tobytes") else int(crc.compute(sec))
    return sec + c.to_bytes(4, "big")


def test_parse_psi_programs_and_names():
    pat = section(0, bytes([0, 1, 0xC1, 0, 0]) + (3).to_bytes(2, "big") + (0xE000 | 0x31).to_bytes(2, "big"))
    pmt = section(2, (3).to_bytes(2, "big") + bytes([0xC1, 0, 0]) + (0xE000 | 0x31).to_bytes(2, "big") + (0xF000).to_bytes(2, "big")
                  + bytes([0x02]) + (0xE000 | 0x41).to_bytes(2, "big") + (0xF000).to_bytes(2, "big") + bytes([0x81]) + (0xE000 | 0x44).to_bytes(2, "big") + (0xF000).to_bytes(2, "big"))
    name = "KTST".encode("utf-16-be").ljust(14, b"\0")
    chan = name + (0xF000 | (13 << 6) | (1 >> 8)).to_bytes(2, "big")[:1] + bytes([(13 << 2) & 0xFF | 0, 1]) + bytes(7) + bytes([0x00, 0x00]) + (3).to_bytes(2, "big") + bytes([0, 0]) + (7).to_bytes(2, "big") + bytes([0xFC, 0x00])
    vct = section(0xC8, bytes([0, 1, 0xC1, 0, 0, 0, 1]) + bytes([1]) + chan)
    ts = make_ts([(0, [0] + list(pat)), (0x31, [0] + list(pmt)), (0x1FFB, [0] + list(vct))])
    psi = atsc_live.parse_psi(ts)
    assert set(psi["programs"]) == {3}
    assert psi["programs"][3]["pmt"] == 0x31 and [t for t, _ in psi["programs"][3]["streams"]] == [0x02, 0x81]
    assert atsc_live.program_pids(psi, 3) >= {0, 0x1FFB, 0x31, 0x41, 0x44}
    assert atsc_live.resolve_program(psi, None) == 3 and atsc_live.resolve_program(psi, "3") == 3
    with pytest.raises(SystemExit):
        atsc_live.resolve_program(psi, "4")


# ---- the streaming receiver (atsc_rx.Front, atsc_rx, rs C, SIMD Viterbi) ------------------------------------------------------------------------------------------------------------

import atsc_rx  # noqa: E402

CAPTURE = pathlib.Path(__file__).resolve().parents[2]/"data"/"atsc"/"atsc35_6s.cf64"


def test_front_continuity():
    x = (RNG.standard_normal(300000) + 1j*RNG.standard_normal(300000)).astype(np.complex64)
    step = 20e6/atsc_rx.SR
    whole = atsc_rx.Front(604e6, 604.5e6, step).process(x)
    f = atsc_rx.Front(604e6, 604.5e6, step)
    parts = np.concatenate([f.process(x[i:i + 70001]) for i in range(0, len(x), 70001)])
    assert len(whole) == len(parts) > 150000 and np.abs(whole - parts).max() < 1e-3 # blocks of any size give the same stream


def test_front_moves_a_tone_to_the_symbol_rate_baseband():
    fs = 20e6; n = 2_000_000
    x = np.exp(2j*np.pi*(604.3e6 - 604e6)*np.arange(n)/fs).astype(np.complex64) # a tone 0.3 MHz above the centre
    z = atsc_rx.Front(604e6, 604.3e6, fs/atsc_rx.SR).process(x) # centred on it: DC
    assert abs(np.mean(z[1000:])) > 0.95 and np.std(z[1000:]) < 0.02


def test_front_clock_trim_changes_the_rate_from_the_next_block():
    f = atsc_rx.Front(604e6, 604.5e6, 20e6/atsc_rx.SR)
    x = np.ones(400000, np.complex64)
    a = len(f.process(x))
    f.step *= 1 + 100e-6
    b = len(f.process(x))
    want = 400000*atsc_rx.SR/20e6
    assert abs(a - want) < 20 and abs(b - want*(1 - 100e-6)) < 20 and a - b >= 8 # 100 ppm of ~215k symbols is ~21 fewer (the blocks differ by the few the filter holds back)


def test_ts_builder_matches_offline_chain():
    """The streaming de-interleaver / Reed-Solomon / de-randomizer fed field by field gives the same packets as atsc_decode on the whole stream"""
    stream = RNG.integers(0, 256, 4*312*207, dtype=np.uint8) # arbitrary trellis output bytes: both chains decode garbage, identically
    ref = atsc_decode.deinterleave(stream)[atsc_decode.DELAY:]
    n = len(ref)//207
    ref_data, ref_err = atsc_decode.reed_solomon(atsc_decode.deinterleave(stream))
    tb = atsc_rx.TsBuilder(); outs = []
    for i in range(4):
        ts, err = tb.push(stream[i*312*207:(i + 1)*312*207]); outs.append(err)
    err = np.concatenate(outs)
    assert len(err) == n and (err == ref_err[:n]).all()


@pytest.mark.skipif(not CAPTURE.exists(), reason="the 6 s ch 35 capture is not in data/atsc")
def test_receiver_decodes_a_real_capture():
    x = np.asarray(np.memmap(CAPTURE, np.complex64, "r")[:int(20e6*3.0)])
    ts, err, st = atsc_rx.decode_array(x, 604e6, 596e6)
    assert st["state"] == "locked" and len(ts) > 15000
    assert np.mean(err >= 0) > 0.85 # the offline receiver gets 97% on this capture
    assert (ts[:, 0] == 0x47).all() and 15 < st["snr"] < 20 and abs(st["clock_ppm"] + 15.4) < 1.0
    psi = atsc_live.parse_psi(ts[err >= 0])
    assert len(psi["programs"]) >= 5 and any(nm.strip() == "KHNL-DT" for nm, _, _ in psi["vct"].values())


@pytest.mark.skipif(not CAPTURE.exists(), reason="the 6 s ch 35 capture is not in data/atsc")
def test_threaded_receiver_runs_in_real_time():
    import time
    x = np.memmap(CAPTURE, np.complex64, "r")
    n_ts = []
    rx = atsc_rx.Receiver(604e6, 596e6, on_ts=lambda ts, err: n_ts.append(float(np.mean(err >= 0)))).start()
    try:
        for i in range(0, int(20e6*4.5), 1 << 18):
            rx.push(np.array(x[i:i + (1 << 18)]))
        t = time.time()
        while (rx.in_q.qsize() or rx.z_q.qsize() or rx.dec_q.qsize()) and time.time() - t < 30:
            time.sleep(0.1)
        time.sleep(0.5)
    finally:
        rx.stop()
    assert len(n_ts) > 20 and np.mean(n_ts) > 0.8 and rx.overruns == 0 and rx.stats.get("error") is None
