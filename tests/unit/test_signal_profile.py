"""rp.analysis.profile: the steps that process each signal of a capture, and what it might be where unsolved"""
import json

import numpy as np
import pytest

import rpps as rp
from rpps.analysis import signal_profile as SP
from rpps.analysis.result import Hypothesis

def capture(path, x, Fs, shift, rng, noise=0.05):
    n = np.arange(len(x))
    y = x*np.exp(2j*np.pi*shift/Fs*n) + noise*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    iq = np.empty(2*len(y), np.int16)
    iq[0::2], iq[1::2] = np.round(y.real*8000), np.round(y.imag*8000)
    iq.tofile(path)
    return rp.source.File("ci16t", str(path), Fs=Fs)

@pytest.fixture(scope="module")
def dvbs_capture(tmp_path_factory):
    """A DVB-S signal (the dvbs/qpsk_3_4 transmit side at 8 samples a symbol) off the centre of an 8 MS/s capture"""
    rng = np.random.default_rng(0)
    ts = np.zeros((900, 188), np.uint8)
    ts[:, 0] = 0x47
    ts[:, 1:] = rng.integers(0, 256, (900, 187))
    w = rp.profile.load("dvbs/qpsk_3_4", sps=8).tx()(rp.Data(np.unpackbits(ts.reshape(-1)).astype(bool), Fs=1e6, sps=1, flush=True))
    return capture(tmp_path_factory.mktemp("cap")/"dvbs.cs16", np.asarray(w.data), 8e6, 1.7e6, rng)

@pytest.mark.kernels
def test_a_dvbs_capture_profiles_to_its_shipped_profile(dvbs_capture):
    profiles = rp.analysis.profile(dvbs_capture, blind=False)
    p = min(profiles, key=lambda p: abs(p.signal.mid - 1.7e6) - p.signal.width) # (the widest one there: the carrier)
    assert p.solved and p.profile_name == "dvbs/qpsk_3_4"
    kinds = [s.kind for s in p.steps]
    assert kinds[:4] == ["channel", "acquire", "timing", "carrier"]
    assert kinds[4:] == ["map", "demap", "puncture", "fec", "align", "deinterleave", "fec", "align", "descramble"]
    text = str(p)
    assert "QPSK at" in text and abs(p.det.baud - 1e6) < 100 and "rs/204_188" in text and "UNSOLVED" not in text
    d = json.loads(json.dumps(p.to_dict(), default=str))
    assert d["solved"] and d["steps"][7]["kind"] == "fec"

@pytest.mark.kernels
def test_a_signal_short_of_symbols_reads_on(dvbs_capture):
    """12 blocks hold ~25k of the carrier's symbols, too few to settle the chain and decide its code there. Allowed to
    read on, the chain keeps going until its code is decided (and only it: the profile is the full one)"""
    pick = lambda ps: min(ps, key=lambda p: abs(p.signal.mid - 1.7e6) - p.signal.width)
    fresh = lambda: rp.source.File("ci16t", dvbs_capture.path, Fs=8e6) # (a File reads on from where it stopped)
    short = pick(rp.analysis.profile(fresh(), blocks=12, max_blocks=12, blind=False))
    assert not short.solved and (short.fec is None or not short.fec.found)
    full = pick(rp.analysis.profile(fresh(), blocks=12, blind=False))
    assert full.solved and full.profile_name == "dvbs/qpsk_3_4" and "rs/204_188" in str(full)

@pytest.mark.kernels
def test_uncoded_symbols_end_in_an_unsolved_step(tmp_path):
    rng = np.random.default_rng(1)
    q = rp.symbol.load("qpsk")
    syms = q.points[rng.integers(0, 4, 160000)]
    x = np.zeros(8*len(syms), complex)
    x[::8] = syms
    taps = rp.signal.filters.rrc_pulse((np.arange(-64, 65))/8, 0.35)
    x = np.convolve(x, taps, mode="same").astype(np.complex64)
    p = max(rp.analysis.profile(capture(tmp_path/"qpsk.cs16", x/np.abs(x).max()*0.5, 8e6, -1.5e6, rng), blind=False),
            key=lambda p: p.signal.width)
    assert not p.solved and p.steps[-1].kind == "fec" and p.steps[-1].status == SP.UNSOLVED
    assert [s.kind for s in p.steps[:4]] == ["channel", "acquire", "timing", "carrier"]
    assert "UNSOLVED" in str(p) and "!" in str(p).splitlines()[-1 - len(p.steps[-1].candidates) - bool(p.steps[-1].note)]

def test_candidates_are_one_per_code_with_softmax_weights():
    h = lambda z, stream, g=((1, 0, 1), (1, 1, 1)): Hypothesis("conv", "rate 1/2 K=3", z, {"g": {0: g[0], 1: g[1]}, "stream": stream})
    c = SP.weights([h(4.0, ("abs", 0)), h(3.9, ("abs", 2)), h(2.0, ("abs", 1), ((1, 1, 1), (1, 1, 0)))])
    assert len(c) == 2 # The same generators on two bit maps are one candidate, at its best
    assert c[0][1] == 4.0 and abs(sum(w for _, _, w in c) - 1) < 1e-9 and c[0][2] > c[1][2]
    assert "generators 5 7" in c[0][0]

def test_a_weak_code_does_not_confirm_a_doubtful_rate():
    from rpps.analysis.identify import CodeDetection
    weak = CodeDetection(found=True, kind="conv", name="k3_5_7/1_2", mismatch=0.041, baseline=0.139)
    strong = CodeDetection(found=True, kind="conv", name="k7_133_171/3_4", mismatch=0.0009, baseline=0.051)
    framed = CodeDetection(found=True, kind="conv", name="k7_133_171/3_4", mismatch=0.02, baseline=0.051)
    framed.profile_name = "dvbs/qpsk_3_4"
    assert not SP._strong(weak) and SP._strong(strong) and SP._strong(framed)

def test_symbols_far_noisier_than_the_band_make_the_fit_doubtful():
    from rpps.sync import Detection
    sig = rp.analysis.Signal(0.0, 1.2e6, 0.0, 1.0, snr=20.0)
    locked = Detection(baud=1e6, esn0=20.5)
    lost = Detection(baud=1e6, esn0=2.0) # A 20 dB band whose symbols measure 2 dB: not on this constellation
    want, gap = SP.esn0_gap(sig, locked)
    assert abs(want - (20 + 10*np.log10(1.2))) < 1e-9 # The band's SNR at this rate
    assert abs(gap - (want - 20.5)) < 1e-9 and gap < SP.ESN0_GAP
    assert SP.esn0_gap(sig, lost)[1] > SP.ESN0_GAP
    assert SP.esn0_gap(rp.analysis.Signal(0.0, 1.2e6, 0.0, 1.0), lost) == (None, None) # No band SNR: nothing to compare

def test_detection_keeps_the_rate_line_it_came_from():
    rng = np.random.default_rng(2)
    q = rp.symbol.load("qpsk")
    x = np.zeros(8*4000, complex)
    x[::8] = q.points[rng.integers(0, 4, 4000)]
    x = np.convolve(x, rp.signal.filters.rrc_pulse((np.arange(-64, 65))/8, 0.35), mode="same")
    det = rp.sync.detect(x.astype(np.complex64), 8e6)
    assert det.line > 5 # A clean signal's rate line stands far above the floor
    noise = (rng.standard_normal(32000) + 1j*rng.standard_normal(32000)).astype(np.complex64)
    assert rp.sync.detect(noise, 8e6).line < 5

def test_the_preamble_turning_backwards_means_an_inverted_spectrum():
    rng = np.random.default_rng(3)
    pre = rp.symbol.load_header("dvbs2_pl").preamble
    data = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, 3000) + 1)) # QPSK: the same conjugated, says nothing on its own
    z = np.concatenate([data[:1000], pre, data[1000:2000], pre, data[2000:]])
    z = z*np.exp(2j*np.pi*0.003*np.arange(len(z)) + 0.7j) + 0.2*(rng.standard_normal(len(z)) + 1j*rng.standard_normal(len(z)))
    assert SP.rotation(z, pre) is False and SP.rotation(np.conj(z), pre) is True
    assert SP.rotation(data, pre) is None

def test_the_carrier_offset_left_comes_from_the_preamble_steps():
    rng = np.random.default_rng(4)
    pre = rp.symbol.load_header("dvbs2_pl").preamble
    data = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, 3000) + 1))
    z = np.concatenate([data[:1000], pre, data[1000:2000], pre, data[2000:]])
    for w in (0.0, 0.3, -1.1, 2.9):
        y = z*np.exp(1j*w*np.arange(len(z)) + 0.7j) + 0.2*(rng.standard_normal(len(z)) + 1j*rng.standard_normal(len(z)))
        assert abs(SP.preamble_offset(y, pre) - w) < 0.02

def test_framed_takes_the_carrier_offset_from_the_preamble():
    """The chain's offset (an M-th power's) can be off by a good part of the rate on APSK: the headers still decode"""
    from types import SimpleNamespace
    frames = [{"modcod": 4, "pilots": 1, "sis": 1, "isi": 0}]*6
    p = rp.profile.load("dvbs2/normal")
    dfl = rp.profile.table("dvbs2/modcod").row({**frames[0], "short": 0})["kbch"] - 80
    data = np.random.default_rng(1).integers(0, 2, dfl*len(frames)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=frames)
    y = rp.Data(o.data, Fs=o.Fs, sps=o.sps, delay=o.delay, flush=True) + rp.wave.Matched(rp.signal.filters.load("pulse/rrc/0p35"))
    z = np.tile(np.asarray(y.data)[::o.sps], 3)
    z = z*np.exp(2j*np.pi*0.06*np.arange(len(z))) # 60 kHz at 1 MBd that the chain didn't see
    got = {name: headers for name, _, headers, _, _, _ in SP.framed(z, SimpleNamespace(cfo=0.0, baud=1e6), skip=0)}
    assert got and max(got.values()) >= 3

def test_framed_reads_the_headers_of_a_profile_ending_in_branches():
    """A profile that ends in branches gives {branch: output}; its headers are in the trunk's stream and the branches'"""
    from types import SimpleNamespace
    frames = [{"modcod": 4, "pilots": 1, "sis": 1, "isi": 0}]*6
    p = rp.profile.load("dvbs2/normal")
    dfl = rp.profile.table("dvbs2/modcod").row({**frames[0], "short": 0})["kbch"] - 80
    data = np.random.default_rng(1).integers(0, 2, dfl*len(frames)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=frames)
    y = rp.Data(o.data, Fs=o.Fs, sps=o.sps, delay=o.delay, flush=True) + rp.wave.Matched(rp.signal.filters.load("pulse/rrc/0p35"))
    z = np.asarray(y.data)[::o.sps]
    got = {name: headers for name, _, headers, _, _, _ in SP.framed(np.tile(z, 3), SimpleNamespace(cfo=0.0, baud=1e6), skip=0)}
    assert "dvbs2/mis" in got and got["dvbs2/mis"] >= 3

@pytest.mark.kernels
def test_progress_reports_each_stage(dvbs_capture):
    seen = []
    ps = rp.analysis.profile(rp.source.File("ci16t", dvbs_capture.path, Fs=8e6), blocks=12, max_blocks=12, blind=False,
                             progress=lambda *a: seen.append(a))
    reads = [a for a in seen if a[0] == "read"]
    assert [a[1] for a in reads] == list(range(1, 13)) and all(a[2] == 12 for a in reads)
    builds = [a for a in seen if a[0] == "build"]
    assert builds[-1][1:] == (len(ps), len(ps), 0)

@pytest.mark.kernels
def test_jobs_profile_the_signals_in_parallel_with_the_same_result(tmp_path):
    """Two carriers in a File: jobs=2 forks two workers, each seeking back to where the profile's reading starts"""
    rng = np.random.default_rng(9)
    def carrier(sps, f, n_pk):
        ts = rng.integers(0, 256, (n_pk, 188)).astype(np.uint8)
        ts[:, 0] = 0x47
        w = rp.profile.load("dvbs/qpsk_3_4", sps=sps).tx()(rp.Data(np.unpackbits(ts.reshape(-1)).astype(bool), Fs=1e6, sps=1, flush=True))
        x = np.asarray(w.data)
        return x*np.exp(2j*np.pi*f*np.arange(len(x)))
    a, b = carrier(8, 1.7e6/8e6, 250), carrier(16, -2.1e6/8e6, 125)
    n = min(len(a), len(b))
    path = tmp_path/"two.cs16"
    capture(path, a[:n] + b[:n], 8e6, 0.0, rng)
    one = rp.analysis.profile(rp.source.File("ci16t", str(path), Fs=8e6), blind=False, blocks=60)
    two = rp.analysis.profile(rp.source.File("ci16t", str(path), Fs=8e6), blind=False, blocks=60, jobs=2)
    assert [str(p) for p in one] == [str(p) for p in two] and sum(p.solved for p in two) == 2

def test_guessed_convolutional_codes_go_into_a_profile():
    """A code no definition has (generators found, or a mother code's puncture) is a "fec" step with its guess"""
    from rpps.analysis.code.conv.mother import guess_conv
    h = Hypothesis("conv", "rate 1/2 K=6", 40.0, {"n": 2, "K": 6, "offset": 1, "streams": (0, 1), "stream": ("abs", 0),
                                                  "g": {0: (1, 1, 0, 1, 0, 1), 1: (1, 0, 1, 1, 1, 1)}})
    h.guess = guess_conv(h)
    p = SP.SignalProfile("t")
    SP._guess_steps(p, [h])
    assert p.steps[0].kind == "fec" and "generators" in p.steps[0].note and str(p)

@pytest.mark.kernels
def test_a_hint_skips_what_an_earlier_look_found(dvbs_capture):
    """The rate and the code an earlier capture of the signal found: given, the profile comes out the same"""
    fresh = lambda: rp.source.File("ci16t", dvbs_capture.path, Fs=8e6)
    pick = lambda ps: min(ps, key=lambda p: abs(p.signal.mid - 1.7e6) - p.signal.width)
    near = lambda s: {"baud": 1e6, "codes": ["k7_171_133/3_4"], "blind": False} if abs(s.mid - 1.7e6) < 2e5 else None
    plain = pick(rp.analysis.profile(fresh(), blind=False))
    hinted = pick(rp.analysis.profile(fresh(), blind=False, hint=near))
    assert hinted.solved and hinted.profile_name == plain.profile_name == "dvbs/qpsk_3_4"
    assert [s.kind for s in hinted.steps] == [s.kind for s in plain.steps]

@pytest.mark.kernels
def test_tdma_bursts_are_demodulated_one_by_one_and_their_code_found():
    """QPSK bursts (a shared 32-symbol preamble, then BCH(63,51) words), each with its own carrier phase and offset:
    the continuous chain can't follow them; each is acquired on its preamble and the code found in the payloads"""
    from rpps.code.types.bchgf import bchgf
    from rpps.logger import Logger
    rng = np.random.default_rng(1)
    q = rp.symbol.load("qpsk")
    pre = np.asarray(q.points)[rng.integers(0, 4, 32)]
    code = bchgf(Logger().Child("t", 0), 63, 51, 6, 0x43, 2)
    Fs, sps = 1.6e6, 16
    shape = rp.wave.Shape(sps, rp.signal.filters.load("pulse/rrc/0p35"))
    xs = []
    for _ in range(60):
        bits = np.asarray(code.encode(rng.integers(0, 2, 51*16).astype(bool)), np.uint8)
        w = np.asarray((rp.Data(np.concatenate([pre, q.modulate(bits)]).astype(np.complex64), Fs=Fs/sps, sps=1) + shape).data)
        xs += [np.zeros(int(rng.integers(1200, 2400)), np.complex64),
               w*np.exp(1j*(2*np.pi*rng.uniform(-200, 200)/Fs*np.arange(len(w)) + rng.uniform(0, 2*np.pi)))]
    x = np.concatenate(xs + [np.zeros(2000, np.complex64)])
    x = x*np.exp(2j*np.pi*150e3/Fs*np.arange(len(x)))
    y = (x + 0.02*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))).astype(np.complex64)
    p = min(rp.analysis.profile(rp.source.Memory(y, Fs=Fs), blind=False), key=lambda p: abs(p.signal.mid - 150e3))
    kinds = [s.kind for s in p.steps]
    assert "demodulate" in kinds and "60 bursts" in str(p) and "BCH(63,51)" in str(p)

@pytest.mark.kernels
def test_a_box_holding_two_carriers_is_split_at_the_cyclic_rate():
    """Two DVB-S carriers side by side, given as one survey box: no lock on the box, the cyclic spectrum's 1 MBd needs
    well under its width, so each carrier in it is profiled at that rate"""
    import copy
    rng = np.random.default_rng(12)
    def carrier(f, n_pk=400):
        ts = rng.integers(0, 256, (n_pk, 188)).astype(np.uint8)
        ts[:, 0] = 0x47
        w = rp.profile.load("dvbs/qpsk_3_4", sps=8).tx()(rp.Data(np.unpackbits(ts.reshape(-1)).astype(bool), Fs=1e6, sps=1, flush=True))
        x = np.asarray(w.data)
        return x*np.exp(2j*np.pi*f/8e6*np.arange(len(x)))
    a, b = carrier(-0.75e6), carrier(0.75e6)
    n = min(len(a), len(b))
    y = (a[:n] + b[:n] + 0.03*(rng.standard_normal(n) + 1j*rng.standard_normal(n))).astype(np.complex64)
    src = rp.source.Memory(y, Fs=8e6)
    sv = list(rp.analysis.survey(src))
    box = copy.copy(sv[0])
    box.mid, box.width = 0.0, 2.9e6
    box.lo, box.hi = -1.45e6, 1.45e6
    ps = rp.analysis.profile(src, signals=[box], survey=sv + [box], blind=False)
    solved = [p for p in ps if p.solved and p.profile_name == "dvbs/qpsk_3_4"]
    assert len(solved) == 2 and all("." in p.name.split(":")[0] for p in solved)
    assert all(abs(abs(p.signal.mid) - 0.75e6) < 30e3 for p in solved) and sorted(np.sign(p.signal.mid) for p in solved) == [-1, 1]

@pytest.mark.kernels
def test_a_box_wider_than_its_one_carrier_is_narrowed_to_it():
    """One DVB-S carrier off centre in a survey box twice its width: profiled on the carrier, not the box"""
    import copy
    rng = np.random.default_rng(5)
    ts = rng.integers(0, 256, (400, 188)).astype(np.uint8)
    ts[:, 0] = 0x47
    w = rp.profile.load("dvbs/qpsk_3_4", sps=8).tx()(rp.Data(np.unpackbits(ts.reshape(-1)).astype(bool), Fs=1e6, sps=1, flush=True))
    x = np.asarray(w.data)
    x = x*np.exp(2j*np.pi*0.6e6/8e6*np.arange(len(x)))
    y = (x + 0.03*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))).astype(np.complex64)
    src = rp.source.Memory(y, Fs=8e6)
    sv = list(rp.analysis.survey(src))
    box = copy.copy(sv[0])
    box.mid, box.width = 0.0, 2.9e6
    box.lo, box.hi = -1.45e6, 1.45e6
    ps = rp.analysis.profile(src, signals=[box], survey=sv + [box], blind=False)
    assert len(ps) == 1 and ps[0].solved and ps[0].profile_name == "dvbs/qpsk_3_4", ps
    assert round(ps[0].signal.mid/1e5) == 6

@pytest.mark.kernels
@pytest.mark.parametrize("fixed_start", [False, True])
def test_a_product_code_laid_out_in_each_frame_is_found_from_few_frames(fixed_start):
    """Continuous QPSK frames (a 16-symbol unique word, then a (64,57) x (16,11) product codeword no definition names):
    far fewer frames than a frame-long codeword's rank test needs, but the rows laid end to end from each payload start
    show the row code, and their columns (a few a frame) the column code. fixed_start: every codeword's first symbol the
    same, which the frame finder takes for part of the unique word: the payload's start is looked for back into it"""
    rng = np.random.default_rng(3)
    q = rp.symbol.load("qpsk")
    rows, cols = rp.code.load("block/hamming/ext_64_57"), rp.code.load("block/hamming/dmr_16_11")
    def word():
        r = np.array([np.asarray(rows.encoder.encode(m), np.uint8) for m in rng.integers(0, 2, (11, 57)).astype(bool)])
        return np.array([np.asarray(cols.encoder.encode(c.astype(bool)), np.uint8) for c in r.T]).T.ravel()
    uw = np.asarray(q.points)[rng.integers(0, 4, 16)]
    def framed():
        w = word()
        while fixed_start and w[:2].any(): # (a quarter of the words start 00)
            w = word()
        return np.concatenate([uw, q.modulate(w)])
    frames = [framed() for _ in range(150)]
    sps, Fs = 8, 8e5
    x = np.asarray((rp.Data(np.concatenate(frames).astype(np.complex64), Fs=Fs/sps, sps=1)
                    + rp.wave.Shape(sps, rp.signal.filters.load("pulse/rrc/0p35"))).data)
    x = x*np.exp(2j*np.pi*(100e3/Fs*np.arange(len(x)) + 0.3))
    y = (x + 0.02*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))).astype(np.complex64)
    p = min(rp.analysis.profile(rp.source.Memory(y, Fs=Fs), blind=False), key=lambda p: abs(p.signal.mid - 100e3))
    fec = next(s for s in p.steps if s.kind == "fec")
    assert p.solved and "product (64,57) x (16,11)" in fec.what, p
    assert ("into the unique word" in fec.note) == fixed_start and (f"{16 + fixed_start}-symbol unique word" in str(p))

def test_header_ids_group_a_branch_points_headers():
    """A framed profile's payload decoded when one of its last headers did: the branches' headers count together"""
    assert SP._header_ids(rp.profile.load("dvbs2/normal").obj["steps"]) == [["pls"], ["bbheader"]]
    assert SP._header_ids(rp.profile.load("dvbs2/mis").obj["steps"]) == [["pls"], ["bb_a", "bb_b"]]

@pytest.mark.kernels
def test_carriers_a_notch_separates_are_each_profiled():
    """Three QPSK carriers at 400 kBd, 520 kHz apart, given as one survey box: a fine spectrum of the box's own samples
    shows the notches between them, and each is profiled on its own, every one listed"""
    import copy
    rng = np.random.default_rng(21)
    Fs, sps = 8e6, 20
    shape = rp.wave.Shape(sps, rp.signal.filters.load("pulse/rrc/0p25"))
    q = rp.symbol.load("qpsk")
    n_sym = 40000
    y = np.zeros(n_sym*sps, np.complex64)
    for f in (-520e3, 0.0, 520e3):
        s = q.modulate(rng.integers(0, 2, 2*n_sym).astype(np.uint8)).astype(np.complex64)
        w = np.asarray((rp.Data(s, Fs=Fs/sps, sps=1) + shape).data)[:len(y)]
        y[:len(w)] += w*np.exp(2j*np.pi*f/Fs*np.arange(len(w)))
    y = (y + 0.02*(rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y)))).astype(np.complex64)
    src = rp.source.Memory(y, Fs=Fs)
    sv = list(rp.analysis.survey(src))
    box = copy.copy(sv[0])
    box.mid, box.width = 0.0, 1.55e6
    box.lo, box.hi = -0.775e6, 0.775e6
    ps = rp.analysis.profile(src, signals=[box], survey=sv + [box], blind=False)
    mids = sorted(round(p.signal.mid/1e4) for p in ps)
    assert len(ps) == 3 and mids == [-52, 0, 52], [(p.name, p.signal.mid) for p in ps]
    assert all(any(s.kind == "split" and "notches" in s.what for s in p.steps) for p in ps)
