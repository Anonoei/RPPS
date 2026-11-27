"""solve and identify: peeling layers off unknown bits, and what is not a code"""
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.analysis import identify as F
from rpps.analysis.code.block import rs as B
from rpps.analysis.sequence import scrambler as SC
from rpps.logger import Logger
from rpps.analysis.code.block import linear as BK
from rpps.code.types.bchgf import bchgf
from rpps.code.types.conv import conv
from support.blind import _feedthrough, scramble, rs_bits, _qpsk

def test_guesses_save_as_definitions_that_load(tmp_path, monkeypatch, hdlc):
    monkeypatch.setattr(_config, "_EXTRA", [tmp_path])
    y = scramble(hdlc, (20, 3))
    g = SC.search(y)[2]
    g.save("found", tmp_path/"scramble")
    sc = rp.code.scram.load("feedthrough/guess/found")
    d = np.asarray((rp.Data(y.astype(bool), flush=True) / sc).data).astype(np.uint8)
    assert np.array_equal(d[20:], hdlc[20:])
    bits = rs_bits(219, 201, 0x187, 120, 1, phase=0, offset=0)
    g = B.search(bits, lengths=(219,))[2]
    g.save("found", tmp_path/"fec")
    code = rp.code.load("block/guess/found").encoder
    words = np.packbits(bits[:len(bits)//(8*219)*8*219]).reshape(-1, 219).astype(np.int64)
    assert not code.syndromes(words).any()

@pytest.mark.kernels
def test_solve_peels_scrambler_then_rs(hdlc):
    y = scramble(rs_bits(219, 201, 0x187, 120, 1, words=40, payload=hdlc, phase=0, offset=0), (20, 3))
    layers = F.solve(y, rs_lengths=(219,))
    assert [h.kind for h, g in layers] == ["scrambler", "rs-blind"]
    assert layers[0][1].definition["poly"] == [20, 3] and layers[1][1].definition["k"] == 201

@pytest.mark.kernels
def test_solve_finds_rs_of_random_payload_behind_a_known_scrambler():
    y = scramble(rs_bits(219, 201, 0x187, 120, 1, words=40, phase=0, offset=0), (20, 3))
    layers = F.solve(y, rs_lengths=(219,))
    assert layers[0][0].params.get("config") == "v35" and layers[1][1].definition["n"] == 219

def _repeated(sym, times, rng, sigma=0.2, cfo=1e-3):
    y = np.tile(sym, times)
    y = y*np.exp(2j*np.pi*cfo*np.arange(len(y)))
    return y + sigma*(rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y)))

def test_a_repeating_stream_has_its_period_and_data_has_none():
    rng = np.random.default_rng(6)
    q = rp.symbol.load("qpsk")
    one = q.modulate(rng.integers(0, 2, 2000).astype(bool))
    assert F.repeat_period(_repeated(one, 15, rng)) == 1000
    assert F.repeat_period(_repeated(q.modulate(rng.integers(0, 2, 30000).astype(bool)), 1, rng)) is None
    assert F.repeat_period(_repeated(np.ones(1), 20000, rng)) == 1 # A carrier: every symbol the last one, turned

def test_fill_is_tested_on_one_period():
    # 1000 fixed symbols again and again: the copies make any test on them hold; one period is too short to tell
    rng = np.random.default_rng(7)
    q = rp.symbol.load("qpsk")
    f = F.identify(_repeated(q.modulate(rng.integers(0, 2, 2000).astype(bool)), 16, rng), "qpsk", blind=True)
    assert not f.found and f.repeat == 1000 and f.complete

@pytest.mark.kernels
def test_a_code_over_a_long_repeating_frame_is_still_found():
    rng = np.random.default_rng(8)
    code = rp.code.load("conv/k7_133_171/1_2")
    cw = code.encoder.encode(rng.integers(0, 2, 8000).astype(bool))[:16000]
    f = F.identify(_repeated(rp.symbol.load("qpsk").modulate(cw), 5, rng, sigma=0.1, cfo=0), "qpsk")
    assert f.repeat == 8000 and f.found and f.name in ("k7_133_171/1_2", "k7_171_133/1_2") # (I and Q swapped)

def test_an_unmodulated_carrier_is_not_a_code():
    rng = np.random.default_rng(4)
    s = (np.ones(40000) + 0.3*(rng.standard_normal(40000) + 1j*rng.standard_normal(40000))).astype(np.complex64)
    assert not F.identify(s, "bpsk").found

def test_bpsk_read_as_qpsk_is_not_a_code_with_equal_generators():
    # I == Q makes (c_a * g) ^ (c_b * g) = 0 for every g: a relation of the bits, not a code
    from rpps.analysis.code.conv.mother import scan_conv
    rng = np.random.default_rng(5)
    u = rng.integers(0, 2, 20000).astype(np.uint8)
    hyps, _ = scan_conv(np.repeat(u, 2), ks=range(3, 6), rates=(2,))
    assert not [h for h in hyps if h.z > 6]
    c = np.asarray(conv(Logger().Child("t", 0), 1, 2, [[1, 1, 1], [1, 0, 1]], 3).encode(u.astype(bool))).astype(np.uint8)
    best = max(scan_conv(c, ks=range(3, 6), rates=(2,))[0], key=lambda h: h.z) # a real code is still found
    assert best.params["K"] == 3 and best.z > 20

def test_acquire_fec_says_it_is_still_acquiring():
    from rpps.sync import Detection
    d = rp.Data(np.ones(1000, np.complex64), Fs=1e4, sps=1)
    d.meta.stream.det = Detection(mod="qpsk", baud=1e4, sps=1.0)
    d + rp.sync.AcquireFEC()
    f = d.meta.stream.est["fec"]
    assert not f.found and not f.complete and f.n == 1000

@pytest.mark.kernels
def test_blind_identify_peels_a_punctured_code_and_the_bch_inside():
    rng = np.random.default_rng(5)
    log = Logger().Child("t", 0)
    inner = bchgf(log, 63, 51, 6, 0x43, 2).encode(rng.integers(0, 2, 51*1200).astype(bool))
    outer = conv(log, 1, 2, [[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]], 7, [[1, 1, 1, 1], [1, 0, 0, 0]]).encode(inner.astype(bool))[5:]
    q = rp.symbol.load("qpsk")
    s = (q.modulate(outer[:len(outer)//2*2])*1j + 0.12*(rng.standard_normal(len(outer)//2) + 1j*rng.standard_normal(len(outer)//2))).astype(np.complex64)
    f = F.identify(s, "qpsk", blind=True)
    assert not f.found and [h.kind for h in f.inner] == ["conv-punctured", "block-blind"]
    assert f.guesses[1].definition["type"] == "bch"

def test_fixed_and_copied_bits_are_framing_not_a_code():
    # Every 32 bits one bit is fixed and two are equal: two checks of weight 1 and 2, which a code never has alone
    from rpps.analysis.code.block import linear as B
    rng = np.random.default_rng(0)
    W = rng.integers(0, 2, (3000, 32)).astype(np.uint8)
    W[:, 5] = 1
    W[:, 20] = W[:, 10]
    (h, g), = B.solve_block(W.ravel(), n=32, offset=0)
    assert h.kind == "frame-blind" and g is None and h.params["fixed"] == [5] and h.params["same"] == [(10, 20)]
    W2 = rng.integers(0, 2, (3000, 4)).astype(np.uint8)
    W2[:, 2:] = W2[:, :2] # "cyclic (4,2), g = x^2 + 1": each bit repeated two later
    assert B.solve_block(W2.ravel())[0][0].kind == "frame-blind"
    G = np.array([[1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]]) # Hamming (7,4)
    c = (rng.integers(0, 2, (4000, 4)) @ G) % 2
    assert B.solve_block(c.ravel().astype(np.uint8))[0][0].kind == "block-blind" # A real code's checks are heavier

def test_a_prbs_or_noisy_fill_is_not_a_low_rate_code():
    """What a blind search calls "linear (192,1)" and the like: rows of a short recurrence (a PRBS) are short of rank at
    any length past its degree, and fill repeating with a few bit errors is nearly rank one: neither is a code"""
    from rpps.analysis.code.block import linear as B
    rng = np.random.default_rng(11)
    s = list(rng.integers(0, 2, 15))
    for i in range(15, 30000):
        s.append(s[i - 15] ^ s[i - 14]) # PRBS15
    s = np.array(s, np.uint8)
    out = B.solve_block(s, n=64, offset=0)
    assert out and out[0][0].kind == "frame-blind" and out[0][0].params["kind"] == "recurrence"
    assert out[0][0].params["degree"] == 15
    fill = np.tile(rng.integers(0, 2, 192).astype(np.uint8), 300)
    flips = rng.random(len(fill)) < 0.03
    out = B.solve_block(fill ^ flips, n=192, offset=0)
    assert not out or out[0][0].kind == "frame-blind"

@pytest.mark.kernels
def test_a_code_carrying_random_bits_says_so():
    # Nothing inside a K=7 1/2 code carrying random bits: identify says what the bits are like (random-looking: keyed or
    # compressed data, no layer to find); over an idle word through a scrambler it finds the scrambler instead
    rng = np.random.default_rng(8)
    g = [[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]]
    enc = lambda d: conv(Logger().Child("t", 0), 1, 2, g, 7, [[1], [1]]).encode(np.asarray(d, bool))[2:]
    f = F.identify(_qpsk(enc(rng.integers(0, 2, 70000)), rng)[:2**15], "qpsk", blind=True)
    assert f.found and not f.inner and f.payload.kind == "random-looking"
    word = rng.integers(0, 2, 24).astype(np.uint8)
    f = F.identify(_qpsk(enc(_feedthrough(np.tile(word, 3000), (23, 18))), rng)[:2**15], "qpsk", blind=True)
    assert f.found and any(h.kind.startswith("scrambler") for h in f.inner) and f.payload is None

def test_a_code_laid_out_from_each_frames_start():
    # Frames (a burst's payload after its preamble) of BCH(63,51) words from their start, cut to uneven lengths: the
    # rank of every frame's words at 0, 63, 126... is short by the code's checks; random frames give nothing
    rng = np.random.default_rng(10)
    code = bchgf(Logger().Child("t", 0), 63, 51, 6, 0x43, 2)
    rows = [np.asarray(code.encode(rng.integers(0, 2, 51*16).astype(bool)), np.uint8)[:int(rng.integers(950, 1008))]
            for _ in range(40)]
    assert BK.frame_units(rows) == (63, 12)
    n, d = BK.frame_units([r ^ (rng.random(len(r)) < 1e-3) for r in rows]) # bit errors: n + 48 words at a time
    assert n == 63 and 0 < d <= 12
    assert BK.frame_units([rng.integers(0, 2, 1000).astype(np.uint8) for _ in range(40)]) == (None, 0)

@pytest.mark.parametrize("seed", [0, 5])
def test_soft_conv_scan_reaches_further_and_keeps_its_false_alarm(seed):
    """A K=6 code no definition has, BPSK at 0 dB, 16384 coded bits. Over 12 seeds the true pair's z is 17-22 hard and
    25-30 soft (asserted: soft 5 over hard); on noise alone soft's best is 4.0 at most, under the 4.7 of its 512 trials"""
    from rpps.analysis.code.conv.mother import scan_conv
    from rpps.math.prob import z_critical
    K6 = [[1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 1]]
    code = conv(Logger().Child("t", 0), 1, 2, K6, 6, [[1]]*2)
    want = {tuple(K6[0]), tuple(K6[1])}
    rng = np.random.default_rng(seed)
    c = np.asarray(code.encode(rng.integers(0, 2, 8192).astype(bool))).astype(np.uint8)
    sigma = np.sqrt(0.5)
    y = (1 - 2.0*c) + sigma*rng.standard_normal(len(c))
    true = lambda hyps: max([h.z for h in hyps if h.kind == "conv" and set(h.params["g"].values()) == want], default=0.0)
    hard, trials = scan_conv((y < 0).astype(np.uint8), ks=[6], rates=(2,))
    soft, _ = scan_conv((y < 0).astype(np.uint8), ks=[6], rates=(2,), llr=2*y/sigma**2)
    assert true(soft) > true(hard) + 5
    n = sigma*rng.standard_normal(len(c))
    noise, _ = scan_conv((n < 0).astype(np.uint8), ks=[6], rates=(2,), llr=2*n/sigma**2)
    assert max([h.z for h in noise], default=0.0) < z_critical(trials)

def test_rs_scan_finds_words_with_errors_by_decoding_them(rng):
    """RS(204,188) words each with 3 byte errors (behind a Viterbi at a few dB): none has zero syndromes, all decode;
    the scan finds the code at its phase by decodes. At effort "fast" it only looks for clean words"""
    from rpps.analysis.identify._scans import scan_rs
    from rpps.code.coding import cached
    from rpps.math.prob import z_critical
    code = cached("block/rs/204_188").encoder
    cw = np.asarray(code.encode(rng.integers(0, 256, 188*40), symbols=True)).reshape(40, 204)
    for w in cw:
        w[rng.choice(204, 3, replace=False)] ^= rng.integers(1, 256, 3)
    bits = np.concatenate((rng.integers(0, 2, 8*57), np.unpackbits(cw.astype(np.uint8).reshape(-1)))).astype(np.uint8)
    hyps, trials, _ = scan_rs(bits, names=["rs/204_188"], interleavers=False)
    best = max(hyps, key=lambda h: h.z)
    assert best.z > z_critical(trials) and best.params["decoded"] and best.params["byte_phase"] == 57 % 204
    with rp.effort("fast"):
        hyps, trials, _ = scan_rs(bits, names=["rs/204_188"], interleavers=False)
    assert not any(h.z > z_critical(trials) for h in hyps)

def test_pair_z_by_matrix_products_is_each_relation_built():
    """scan_conv's z for every generator pair (mother._pair_z, two matrix products) against the relations built one by
    one: soft (_soft_z) and hard (the count of relation bits that hold)"""
    from rpps.analysis.code.conv import mother
    from rpps.math.prob import deviance_z
    rng = np.random.default_rng(5)
    K, polys = 5, mother._polys(5)
    streams = [rng.integers(0, 2, 3000).astype(np.uint8) for _ in range(2)]
    rel = [np.log(np.clip(np.tanh(np.abs(rng.normal(size=3000))*2), 1e-12, 1)) for _ in range(2)]
    conv = [mother._convolve_all(c, polys, K) for c in streams]
    sums = [mother._sum_all(r, polys, K) for r in rel]
    soft, hard = mother._pair_z(conv, sums)[(0, 1)], mother._pair_z(conv, None)[(0, 1)]
    for ia in range(len(polys)):
        s = conv[0] ^ conv[1][ia]
        assert np.allclose(soft[:, ia], mother._soft_z(s, sums[0] + sums[1][ia]))
        assert np.allclose(hard[:, ia], deviance_z(s.sum(axis=1), s.shape[1]))

def test_zero_syndrome_words_at_every_phase_by_prefix_xor():
    """_rs_zero (every window's first syndrome from one prefix XOR) against the syndromes word by word: random bytes,
    codewords after a stray lead (all 8 at their phase, none elsewhere), one value repeated (no codeword); every
    configured byte-wide RS code, the dual-basis one too"""
    from rpps.analysis.identify import _scans as S
    from rpps.code.coding import cached
    rng = np.random.default_rng(0)
    for name in rp.code.names("block", "rs"):
        code = cached(f"block/{name}").encoder
        if getattr(code, "m", 8) != 8:
            continue
        n = code.den
        cw = np.asarray((rp.Data(rng.integers(0, 2, code.num*8*30).astype(bool))*rp.code.load(f"block/{name}")).data)
        for seg in (rng.integers(0, 256, 40*n), np.concatenate((rng.integers(0, 256, 57), code.to_symbols(cw))),
                    np.full(30*n, 7)):
            phases = list(range(n))
            first = (len(seg) - n)//n - 8
            rows = np.stack([seg[p + first*n:p + (first + 8)*n].reshape(8, n) for p in phases])
            want = (~code.syndromes(rows.reshape(-1, n)).any(axis=1).reshape(n, 8) &
                    ~(rows == rows[:, :, :1]).all(axis=2)).sum(axis=1)
            assert np.array_equal(S._rs_zero(code, seg, n, phases, 8), want)
