"""Blind tools for what no definition covers: feedthrough scramblers (rpps.analysis.sequence.scrambler), Reed-Solomon codes
(rpps.analysis.code.block.rs), the layers inside a decoded stream (fec.solve), and what must not be taken for a code"""
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.analysis import identify as F
from rpps.analysis.code.block import rs as B
from rpps.analysis.sequence import scrambler as SC
from rpps.code.types.rs import rs
from rpps.logger import Logger

def scramble(d, taps):
    """y[n] = d[n] ^ y[n-t] for each tap (a feedthrough scrambler, register starting at zero)"""
    y = np.zeros(len(d), np.uint8)
    for n in range(len(d)):
        v = int(d[n])
        for t in taps:
            if n >= t:
                v ^= int(y[n - t])
        y[n] = v
    return y

@pytest.fixture(scope="module")
def hdlc():
    rng = np.random.default_rng(1)
    fr = rp.link.load("frame/hdlc")
    frames = [bytes(rng.integers(0, 256, int(rng.integers(20, 200))).astype(np.uint8)) for _ in range(120)]
    return np.asarray(fr.encoder.encode(frames)).astype(np.uint8)[:100000]

def rs_bits(n, k, poly, fcr, prim, words=24, payload=None, seed=0, phase=37, offset=5):
    rng = np.random.default_rng(seed)
    code = rs(Logger().Child("t", 0), k, n, 8, poly, fcr, prim)
    msg = rng.integers(0, 256, (words, k)) if payload is None else np.packbits(payload[:words*k*8]).reshape(words, k)
    cw = np.asarray(code.encode(np.asarray(msg).reshape(-1), symbols=True)).astype(np.uint8)[phase:]
    return np.concatenate((rng.integers(0, 2, offset).astype(np.uint8), np.unpackbits(cw)))

def test_descramble_is_the_feedthrough_definition():
    y = np.random.default_rng(0).integers(0, 2, 2000).astype(bool)
    d = np.asarray((rp.Data(y, flush=True) / rp.code.scram.load("feedthrough/v35")).data).astype(np.uint8)
    assert np.array_equal(rp.code.scram.Feedthrough.descramble(y, (20, 3))[20:], d[20:])

def test_scrambler_found_from_the_structure_under_it(hdlc):
    h, n, g = SC.search(scramble(hdlc, (20, 3)))
    assert h[0].params["taps"] == [20, 3] and g.definition["poly"] == [20, 3]
    h, n, g = SC.search(scramble(np.ones(60000, np.uint8), (17, 12))) # An idle line of ones: the strongest structure
    assert g.definition["poly"] == [17, 12]

def test_three_taps(hdlc):
    assert not SC.search(scramble(hdlc, (23, 18, 5)))[0]
    assert SC.search(scramble(hdlc, (23, 18, 5)), max_taps=3)[2].definition["poly"] == [23, 18, 5]

def test_white_stays_white(hdlc):
    rng = np.random.default_rng(2)
    for y in (rng.integers(0, 2, 100000).astype(np.uint8), scramble(rng.integers(0, 2, 60000).astype(np.uint8), (20, 3))):
        h, n, g = SC.search(y)
        assert not h and g is None
    h, n, g = SC.search(hdlc) # Structure with no scrambler: "none" stands out, nothing to save
    assert h[0].params["taps"] == [] and g is None

def test_blind_rs_finds_field_generator_alignment_and_polarity():
    bits = rs_bits(219, 201, 0x187, 120, 1) ^ 1
    h, n, g = B.search(bits, lengths=(204, 219))
    p = h[0].params
    assert (p["n"], p["k"], p["poly"], p["fcr"], p["prim"], p["polarity"], p["bit_offset"]) == (219, 201, 0x187, 120, 1, 1, 5)
    assert g.definition == {**g.definition, "n": 219, "k": 201, "poly": 0x187, "fcr": 120, "prim": 1}

def test_blind_rs_generator_with_a_root_spacing():
    roots = [(11*(112 + i)) % 255 for i in range(32)]
    assert B.fit_generator(roots) == (112, 11)
    assert B.fit_generator([3, 7, 100]) is None

def test_blind_rs_nothing_in_random_or_constant_bits():
    rng = np.random.default_rng(3)
    assert not B.search(rng.integers(0, 2, 60000).astype(np.uint8), lengths=(204, 255))[0]
    assert not B.search(np.zeros(60000, np.uint8), lengths=(204, 255))[0] # Zero words are codewords of any RS code

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

# --- Binary block codes, products, unknown puncture patterns ---------------------------------------------------------

from rpps.analysis.code.block import linear as BK
from rpps.analysis.code.conv import puncture as PU
from rpps.code.types.bchgf import bchgf
from rpps.code.types.conv import conv
from rpps.math.gf2 import nullspace, rank, rref

def test_packed_gf2_matches_its_definition():
    rng = np.random.default_rng(0)
    for R, n in ((5, 9), (40, 64), (100, 130), (70, 70)):
        M = rng.integers(0, 2, (R, n)).astype(np.uint8)
        N = nullspace(M)
        A, piv = rref(M)
        assert not ((M.astype(int) @ N.T) % 2).any() and rank(M) == len(piv) == n - len(N)

def test_bch_found_with_its_field_and_t_shortened_too():
    log = Logger().Child("t", 0)
    rng = np.random.default_rng(1)
    for n, k in ((63, 51), (55, 43)):
        b = np.asarray(bchgf(log, n, k, 6, 0x43, 2).encode(rng.integers(0, 2, k*600).astype(bool))).astype(np.uint8)[3:]
        (h, g), = BK.solve_block(b)
        assert g.definition["type"] == "bch" and (g.definition["n"], g.definition["k"], g.definition["t"]) == (n, k, 2)
        assert g.definition["poly"] == "0x43" and h.params["offset"] == n - 3

def test_cyclic_code_found_by_its_generator():
    rng = np.random.default_rng(2)
    b = np.asarray(rp.code.load("block/golay/23_12").encoder.encode(rng.integers(0, 2, 12*600).astype(bool))).astype(np.uint8)
    (h, g), = BK.solve_block(b[5:])
    assert g.definition["type"] == "cyclic" and int(g.definition["poly"], 16) == 0xC75

def test_an_ldpc_code_from_its_sparse_checks():
    rng = np.random.default_rng(6)
    H = np.asarray(rp.code.load("block/ldpc/gallager_96_48").encoder.H, dtype=np.uint8)[:, rng.permutation(96)]
    cw = (rng.integers(0, 2, (1500, len(nullspace(H)))) @ nullspace(H) % 2).astype(np.uint8).reshape(-1)[13:]
    (h, g), = BK.solve_block(cw[:32768])
    assert h.params["kind"] == "ldpc" and (h.params["n"], h.params["k"]) == (96, 50)
    assert {len(c) for c in g.definition["checks"]} == {6} # Gallager's column-weight-3, row-weight-6 checks

def test_soft_values_find_sparse_checks_at_a_bit_error_rate_hard_bits_cannot():
    rng = np.random.default_rng(6)
    H = np.asarray(rp.code.load("block/ldpc/gallager_96_48").encoder.H, dtype=np.uint8)[:, rng.permutation(96)]
    G = nullspace(H)
    cw = (rng.integers(0, 2, (3000, len(G))) @ G % 2).astype(np.uint8)
    y = (1 - 2.0*cw) + 0.4*rng.standard_normal(cw.shape) # BER about 0.6%
    S, _ = BK.sparse_checks((y < 0).astype(np.uint8), np.abs(y), k=50)
    assert rank(BK.lightest(S)) == 46

def test_product_code_rows_and_columns():
    rng = np.random.default_rng(3)
    r, c = rp.code.load("block/hamming/15_11").encoder, rp.code.load("block/hamming/7_4").encoder
    blocks = []
    for _ in range(500):
        rows = np.array([r.encode(x) for x in rng.integers(0, 2, (4, 11)).astype(bool)])
        blocks.append(np.array([c.encode(col) for col in rows.T]).T.reshape(-1))
    (h, g), = BK.solve_block(np.concatenate(blocks).astype(np.uint8)[29:])
    assert h.params["kind"] == "product" and (h.params["n1"], h.params["k1"], h.params["n2"], h.params["k2"]) == (15, 11, 7, 4)

def test_no_block_code_in_random_or_constant_bits():
    assert BK.solve_block(np.random.default_rng(4).integers(0, 2, 40000).astype(np.uint8)) == []
    assert BK.solve_block(np.ones(40000, np.uint8)) == []
    pattern = np.random.default_rng(5).integers(0, 2, 128).astype(np.uint8) # Idle fill repeating every 128 bits
    assert BK.solve_block(np.tile(pattern, 400)[37:]) == []

@pytest.mark.kernels
def test_unknown_puncture_pattern_on_a_known_mother_code():
    rng = np.random.default_rng(2)
    g = [[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]]
    pat = [[1, 0, 1, 0, 1, 1], [1, 1, 0, 1, 0, 0]] # Rate 6/7, in no definition
    bits = conv(Logger().Child("t", 0), 1, 2, g, 7, pat).encode(rng.integers(0, 2, 20000).astype(bool))[3:]
    q = rp.symbol.load("qpsk")
    s = q.modulate(bits[:len(bits)//2*2])*1j
    s = s + 0.15*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s)))
    v, guess = PU.best(*PU.search(F.bit_streams(s[:8000], q, differential=False)))
    assert guess.definition["extends"] == rp.code.serial.mother(g, 7) == "k7_133_171" # a child of the mother code it found
    found = np.array(guess.definition["puncture"], dtype=bool)
    assert any(np.array_equal(found, np.roll(np.array(pat, dtype=bool), sh, axis=1)) for sh in range(6)) # (up to a turn)
    assert v.params["mismatch"] < 0.01
    q2 = q.points[rng.integers(0, 4, 8000)] + 0.1*(rng.standard_normal(8000) + 1j*rng.standard_normal(8000))
    assert PU.best(*PU.search(F.bit_streams(q2, q, differential=False))) == (None, None)

@pytest.mark.parametrize("P", [2, 3, 5])
def test_puncture_mismatches_in_one_batch_are_each_patterns_own(P):
    """Every pattern of a mother decoded together on the mother's trellis: the counts mismatch_batch gives under each
    pattern's own punctured code, row for row (a codeword stream, a punctured one at every offset, random bits)"""
    from rpps.code.coding import mismatch_batch
    rng = np.random.default_rng(P)
    K, gen = 7, ((1, 0, 1, 1, 0, 1, 1), (1, 1, 1, 1, 0, 0, 1))
    pats = PU.patterns(P)
    sent = conv(Logger().Child("t", 0), 1, 2, [list(g) for g in gen], 7, pats[0].astype(int).tolist()).encode(
        rng.integers(0, 2, 3000).astype(bool)).astype(np.uint8)
    rows = []
    for pat in pats:
        m, per = int(pat.sum()), 400//P
        rows += [(pat, sent[off:off + per*m]) for off in range(m)]
        rows += [(pat, rng.integers(0, 2, per*m).astype(np.uint8)), (pat, rng.integers(0, 2, 4*per*m).astype(np.uint8))]
    k, n = PU.mismatches(K, gen, rows)
    for (pat, h), kk, nn in zip(rows, k, n):
        want, wn = mismatch_batch(PU._code(K, gen, pat), h[None])
        assert (kk, nn) == (int(want[0]), wn)
    assert min(k[:int(pats[0].sum())]) == 0 # (the stream's own pattern, at its offset: nothing differs)

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

def _lfsr(taps, n, rng):
    s = list(rng.integers(0, 2, max(taps)))
    while len(s) < n:
        s.append(int(np.bitwise_xor.reduce([s[-t] for t in taps])))
    return np.array(s[:n], np.uint8)

def _feedthrough(d, taps):
    """y[n] = d[n] ^ y[n-a] ^ y[n-b]...: the scrambler rpps.code.scram.Feedthrough.descramble undoes"""
    y = np.zeros(len(d), np.uint8)
    for n in range(len(d)):
        v = d[n]
        for t in taps:
            if n >= t:
                v ^= y[n - t]
        y[n] = v
    return y

def test_berlekamp_massey_finds_the_register():
    from rpps.math import gf2
    rng = np.random.default_rng(0)
    C, L = gf2.berlekamp_massey(_lfsr((23, 18), 400, rng))
    assert (L, gf2.lfsr_taps(C, L)) == (23, (23, 18))
    _, L = gf2.berlekamp_massey(rng.integers(0, 2, 400).astype(np.uint8))
    assert 190 <= L <= 210 # Random bits: about half their length

def test_a_prbs_and_an_additive_scrambler_over_idle_data():
    from rpps.analysis.sequence import scrambler as SC
    rng = np.random.default_rng(1)
    h, _, g = SC.additive_search(_lfsr((23, 18), 40000, rng))
    assert h[0].params["taps"] == [23, 18] and h[0].params["prbs"] and "PRBS23" in h[0].name
    d = rng.integers(0, 2, 40000).astype(np.uint8)
    for a in range(0, 40000, 4000):
        d[a:a + 1500] = 0 # Idle stretches: the scrambler's own sequence shows through
    h, _, g = SC.additive_search(d ^ _lfsr((15, 14), 40000, rng))
    assert h[0].params["taps"] == [15, 14] and not h[0].params["prbs"] and g is not None
    assert SC.additive_search(rng.integers(0, 2, 40000).astype(np.uint8))[0] == []

def test_cascaded_scramblers_over_a_repeating_word_are_explained():
    # Two scramblers (1+x^18+x^23) in cascade over a 24-bit word sent again and again
    from rpps.analysis.sequence import scrambler as SC
    word = np.random.default_rng(24).integers(0, 2, 24).astype(np.uint8)
    y = _feedthrough(_feedthrough(np.tile(word, 1700), (23, 18)), (23, 18))
    h, _, _ = SC.additive_search(y)
    p = h[0].params
    assert p["data_period"] == 24 and p["scramblers"][0]["taps"] == [23, 18] and p["scramblers"][0]["times"] == 2
    assert p["pattern_holds"] > 0.99 and p["pattern"] in "".join(map(str, np.tile(word, 3)))
    assert SC.explain([23, 18]) [1] == 1 # A scrambler over constant data: period 1

def test_an_inverted_output_is_the_code_with_that_output_flipped():
    # CCSDS 131.0-B 3.3: the basic K=7 code sends its G2 output inverted
    rng = np.random.default_rng(4)
    m = rng.integers(0, 2, 500).astype(bool)
    plain = rp.code.load("conv/k7_171_133/1_2").encoder.encode(m)
    ccsds = rp.code.load("conv/k7_171_133/ccsds_1_2")
    e = ccsds.encoder.encode(m)
    assert np.array_equal(e[0::2], plain[0::2]) and np.array_equal(e[1::2], ~plain[1::2])
    assert np.array_equal(ccsds.decoder.decode(e.astype(np.uint8))[:len(m)], m)

@pytest.mark.parametrize("name", ["k7_133_171/2_3", "k7_133_171/5_6", "k7_133_171/7_8", "k9_561_753/1_2",
                                  "k9_557_663_711/1_3", "k7_133_171_165/1_3", "k7_133_171_145_133/1_4"])
def test_standard_convolutional_codes_round_trip(name):
    rng = np.random.default_rng(5)
    c = rp.code.load(f"conv/{name}")
    m = rng.integers(0, 2, 1260).astype(bool) # A whole number of every puncture period (2, 3, 5, 7)
    e = c.encoder.encode(m)
    assert np.array_equal(c.decoder.decode(e.astype(np.uint8))[:len(m)], m)

def test_a_ccsds_stream_on_bpsk_is_identified():
    # On BPSK the G2 inversion turns every other symbol: no map of the plain code fits, the CCSDS definition does
    rng = np.random.default_rng(6)
    e = rp.code.load("conv/k7_171_133/ccsds_1_2").encoder.encode(rng.integers(0, 2, 20000).astype(bool))
    y = (1 - 2*e.astype(float)) + 0.3*rng.standard_normal(len(e))
    out = rp.analysis.identify.identify(y.astype(np.complex64), "bpsk")
    assert out.found and out.name == "k7_171_133/ccsds_1_2"

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

def _qpsk(bits, rng, sigma=0.1):
    q = rp.symbol.load("qpsk")
    s = q.modulate(np.asarray(bits[:len(bits)//2*2], np.uint8))*1j
    return (s + sigma*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s)))).astype(np.complex64)

@pytest.mark.kernels
def test_an_unknown_convolutional_code_is_not_taken_for_a_block_code():
    # K=6 (65, 57), in no definition: a convolutional code is linear over any block length, so the block search would
    # name it a code of some length; the convolutional scan goes first and gives its generators
    rng = np.random.default_rng(6)
    g = [[1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 1]]
    s = _qpsk(conv(Logger().Child("t", 0), 1, 2, g, 6, [[1], [1]]).encode(rng.integers(0, 2, 40000).astype(bool))[1:], rng)
    f = F.identify(s[:2**14], "qpsk", blind=True)
    assert [h.kind for h in f.inner][:1] == ["conv"]
    assert sorted(map(list, f.guesses[0].definition["generator"])) == sorted(g)

@pytest.mark.kernels
def test_an_odd_length_block_code_on_qpsk_is_found_at_its_length():
    # BCH(63,51) on QPSK: a map that swaps the bits of a symbol cuts across words of odd length and still shows
    # structure at 126; of every map's find, the shortest code is the one
    rng = np.random.default_rng(7)
    b = bchgf(Logger().Child("t", 0), 63, 51, 6, 0x43, 2).encode(rng.integers(0, 2, 51*1100).astype(bool))[5:]
    f = F.identify(_qpsk(b, rng)[:2**15], "qpsk", blind=True)
    h = f.inner[0]
    assert (h.params["n"], h.params["k"]) == (63, 51) and f.guesses[0].definition["type"] == "bch"

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

@pytest.mark.kernels
def test_soft_scoring_finds_a_configured_code_hard_bits_cannot():
    # K=7 rate 1/2 on QPSK at Es/N0 3 dB: the hard decisions are ~8% wrong, near what a wrong code gets on random bits,
    # so the hard re-encoding test can't take it; weighted by |LLR| the right code's disagreements fall on weak bits
    from rpps.analysis.code.conv import mother as M
    rng = np.random.default_rng(13)
    g = [[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]]
    b = conv(Logger().Child("t", 0), 1, 2, g, 7, [[1], [1]]).encode(rng.integers(0, 2, 40000).astype(bool))[1:]
    q = rp.symbol.load("qpsk")
    s = q.modulate(np.asarray(b[:len(b)//2*2], np.uint8))*1j
    sigma = np.sqrt(0.5*10**(-3/10))
    s = (s + sigma*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s)))).astype(np.complex64)
    f = F.identify(s[:2**15], "qpsk")
    assert f.found and f.kind == "conv" and f.name.endswith("/1_2")
    head = F.bit_streams(s[:2**15], q, True)
    hard, _ = M.scan_defined_conv(head)
    assert not any(h.coding[1].endswith("/1_2") and h.z > 5.6 for h in hard) # (hard alone: nothing the pick would take)

def test_product_columns_from_few_blocks():
    """A product code's column length from fewer blocks than columns the rank needs: several columns from each block"""
    from rpps.analysis.code.block import linear
    rng = np.random.default_rng(4)
    tpc = rp.code.load("block/product/tpc_64_57_sq")
    b = np.concatenate([np.asarray(tpc.encoder.encode(rng.integers(0, 2, 3249).astype(bool)), np.uint8) for _ in range(30)])
    assert linear.product(b, 64, 0, max_n2=80, per_block=1) is None # 30 blocks, one column each: too few for 64 + 24
    n2, ph, d = linear.product(b, 64, 0, max_n2=80)
    assert (n2, ph, d) == (64, 0, 7)
