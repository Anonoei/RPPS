"""coding - scram - mod end to end, with the operator syntax from the README"""
import itertools

import numpy as np
import pytest

import rpps as rp

CODES = [("block", "hamming/7_4"), ("block", "hamming/15_11"), ("block", "repeat/3")]
SCRAMS = ["v35", "fib"]
MODS = ["bpsk", "qpsk", "8psk"]

def build(code, scram, mod):
    m = rp.symbol.load(mod)
    m.set_mapping(m.maps[0])
    return rp.code.load(*code), rp.code.scram.load(scram), m

@pytest.mark.parametrize("code,scram,mod", itertools.product(CODES, SCRAMS, MODS))
def test_pipeline_round_trip(code, scram, mod, rng):
    ecc, scr, m = build(code, scram, mod)
    raw = bytes(rng.integers(0, 256, 66).astype(np.uint8)) # 528 bits: whole blocks and symbols for every combination
    want = rp.Data(raw).as_bytes().hex # The operators work in place, so keep the original
    syms = rp.Data(raw)*scr*ecc*m
    out = syms/m/ecc/scr
    assert out.as_bytes().hex == want

def test_readme_example():
    ecc, scr, m = build(("block", "hamming/7_4"), "v35", "qpsk")
    msg = rp.Data(b"Hello World!")
    syms = msg * scr * ecc * m
    out = syms / m / ecc / scr
    assert out.as_bytes().hex == b"Hello World!".hex()

def test_operators_do_not_modify_their_input():
    ecc, scr, m = build(("block", "hamming/7_4"), "v35", "bpsk")
    msg = rp.Data(b"Hello World!")
    a = msg * scr * ecc * m
    b = msg * scr * ecc * m          # the same input gives the same output
    assert msg.hex == b"Hello World!".hex()
    assert np.array_equal(a.data, b.data)

def test_bpsk_needs_no_sizing():
    ecc, scr, m = build(("block", "hamming/7_4"), "v35", "bpsk")
    out = (rp.Data(b"Hello World!") * scr * ecc * m) / m / ecc / scr
    assert out.as_bytes().hex == b"Hello World!".hex()

@pytest.mark.parametrize("mod", MODS)
def test_partial_last_block_is_held_until_flushed(mod):
    """17 bits is not a whole block for hamming 7/4 or the symbols: the stages hold the rest, and flushing pads it"""
    ecc, scr, m = build(("block", "hamming/7_4"), "v35", mod)
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
    held = rp.Data(bits) * scr * ecc * m
    done = rp.Data(bits, flush=True) * scr * ecc * m
    assert len(held) < len(done)
    out = done / m / ecc / scr
    assert np.array_equal(out.data[:len(bits)], bits)
    assert len(out.data) >= len(bits)  # the padding was added after scrambling, so it descrambles to noise

def test_flushing_a_decode_drops_the_partial_block():
    ecc, scr, m = build(("block", "hamming/7_4"), "v35", "bpsk")
    syms = rp.Data(np.ones(32, dtype=bool), flush=True) * ecc * m   # 8 words of 7 bits
    syms = rp.Data(syms.data[:-3], flush=True, sps=1)                    # cut: 53 bits, 7 whole words and 4 left over
    assert len((syms / m / ecc).data) == 28

def chunked(chain, data, cuts):
    """Run `chain` (Data -> Data) on pieces of data, handing the stream state on, and end the stream"""
    ctx, outs = None, []
    kw = {"sps": 1} if np.iscomplexobj(data) else {} # complex pieces are symbols
    for a, b in zip(cuts[:-1], cuts[1:]):
        out = chain(rp.Data(data[a:b], ctx=ctx, **kw))
        ctx = out.ctx
        outs.append(out.data)
    outs.append(chain(rp.Data(data[:0], ctx=ctx, flush=True, **kw)).data)
    return np.concatenate(outs)

def test_stream_in_chunks_matches_one_shot(rng):
    """Chunks that don't line up with any block give the one-shot result"""
    ecc, scr, m = build(("block", "hamming/7_4"), "v35", "qpsk")
    bits = rng.integers(0, 2, 2000).astype(bool)
    whole = rp.Data(bits, flush=True) * scr * ecc * m
    got = chunked(lambda d: d * scr * ecc * m, bits, [0, 7, 301, 302, 1003, 2000])
    assert np.allclose(got, whole.data)

def test_stream_decode_in_chunks(rng):
    ecc, scr, m = build(("block", "hamming/15_11"), "v35", "8psk")
    bits = rng.integers(0, 2, 33*10).astype(bool)
    syms = (rp.Data(bits, flush=True) * scr * ecc * m).data
    got = chunked(lambda d: d / m / ecc / scr, syms, [0, 5, 64, len(syms)])
    assert np.array_equal(got[:len(bits)], bits)

def test_scrambler_state_belongs_to_the_stream(rng):
    """Two streams through one scrambler do not disturb each other"""
    scr = rp.code.scram.load("v35")
    bits = rng.integers(0, 2, 100).astype(bool)
    a = rp.Data(bits) * scr
    b = rp.Data(bits) * scr
    assert np.array_equal(a.data, b.data)
    assert np.array_equal((a / scr).data, bits)

def test_interleaver_streams(rng):
    il = rp.code.load("interleave", "forney/12_17")
    sym = rng.integers(0, 256, 12*40).astype(np.uint8)
    whole = rp.Data(sym, flush=True) * il
    got = chunked(lambda d: d * il, sym, [0, 12*7, 12*7 + 5, len(sym)])
    assert np.array_equal(got, whole.data)

CONV = ["k3_5_7/1_2", "k7_133_171/1_2", "k7_133_171/3_4"]

@pytest.mark.parametrize("name", CONV)
def test_conv_stream_keeps_the_register(name, rng):
    """The encoder register belongs to the stream: chunked equals one piece, and a second stream starts clean"""
    ecc = rp.code.load("conv", name)
    bits = rng.integers(0, 2, 600).astype(bool)
    whole = rp.Data(bits, flush=True) * ecc
    got = chunked(lambda d: d * ecc, bits, [0, 11, 100, 333, 600])
    assert np.array_equal(got, whole.data)
    again = rp.Data(bits, flush=True) * ecc
    assert np.array_equal(again.data, whole.data)

@pytest.mark.parametrize("name", CONV)
def test_conv_stream_decodes_in_chunks(name, rng):
    ecc = rp.code.load("conv", name)
    bits = rng.integers(0, 2, 900).astype(bool)
    coded = (rp.Data(bits, flush=True) * ecc).data
    got = chunked(lambda d: d / ecc, coded, [0, 13, 250, 251, 1000, len(coded)])
    assert np.array_equal(got[:len(bits)], bits)

def test_conv_stream_corrects_errors_across_chunks(rng):
    ecc = rp.code.load("conv", "k7_133_171/1_2")
    bits = rng.integers(0, 2, 2000).astype(bool)
    coded = (rp.Data(bits, flush=True) * ecc).data.copy()
    coded[rng.choice(len(coded), 40, replace=False)] ^= True
    got = chunked(lambda d: d / ecc, coded, list(range(0, len(coded), 301)) + [len(coded)])
    assert np.array_equal(got[:len(bits)], bits)

TERMINATED = {
    "base": "conv", "type": "split",
    "encode": {"type": "conv", "num": 1, "den": 2, "generator": [[1, 0, 1], [1, 1, 1]], "constraint": 3, "terminated": True},
    "decode": {"type": "viterbi", "num": 1, "den": 2, "generator": [[1, 0, 1], [1, 1, 1]], "constraint": 3, "terminated": True},
}

def test_terminated_conv_gets_its_tail_when_the_stream_ends(rng):
    ecc = rp.code.init("t", TERMINATED)
    bits = rng.integers(0, 2, 100).astype(bool)
    coded = chunked(lambda d: d * ecc, bits, [0, 60, 100])
    assert len(coded) == 2*(100 + 2)                      # one tail for the stream, not one per chunk
    assert np.array_equal(coded, (rp.Data(bits, flush=True) * ecc).data)
    out = chunked(lambda d: d / ecc, coded, [0, 51, len(coded)])
    assert np.array_equal(out, bits)

def test_bcjr_conv_stream_decodes_at_the_end(rng):
    ecc = rp.code.load("conv", "k7_133_171/1_2", decoder="bcjr")
    bits = rng.integers(0, 2, 300).astype(bool)
    coded = (rp.Data(bits, flush=True) * ecc).data
    first = rp.Data(coded[:200]) / ecc
    assert len(first) == 0                                 # it waits for the end of the stream
    last = rp.Data(coded[200:], ctx=first.ctx, flush=True) / ecc
    assert np.array_equal(last.data[:300], bits)

def test_scrambling_changes_the_symbols(rng):
    ecc, scr, m = build(("block", "hamming/7_4"), "v35", "qpsk")
    msg = bytes(rng.integers(0, 256, 66).astype(np.uint8))
    with_scr = rp.Data(msg)*scr*ecc*m
    scr.reset()
    without = rp.Data(msg)*ecc*m
    assert not np.allclose(with_scr.data, without.data)
