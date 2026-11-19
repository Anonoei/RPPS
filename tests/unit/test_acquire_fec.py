"""Automatic FEC identification: rpps.analysis.identify.identify and the rp.sync.AcquireFEC stage"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis import identify as F
from rpps.analysis.modulation.detect import Detection

def noisy(s, rng, sigma=0.12):
    return (s + sigma*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s)))).astype(np.complex64)

@pytest.fixture(scope="module")
def dvbs():
    """DVB-S symbols (dvbs/qpsk_3_4 transmit side, no pulse shaping) of 220 TS packets"""
    rng = np.random.default_rng(1)
    p = rp.profile.load("dvbs/qpsk_3_4")
    ts = np.zeros((220, 188), np.uint8)
    ts[:, 0] = 0x47
    ts[:, 1:] = rng.integers(0, 256, (220, 187))
    d = rp.Data(np.unpackbits(ts.reshape(-1)).astype(bool), flush=True)
    for how, op in p.tx():
        if how == "*":
            d = d * op
    return np.asarray(d.data).astype(np.complex64), ts

def packets(out):
    b = np.packbits(np.asarray(out).astype(bool))
    return b[:len(b)//188*188].reshape(-1, 188)

def test_dvbs_is_identified_down_to_its_profile(dvbs):
    syms, ts = dvbs
    rng = np.random.default_rng(2)
    rx = noisy(np.conj(syms[5:]*np.exp(1j*np.pi/2)), rng) # Unknown alignment, carrier a quarter turn off, spectrum inverted
    f = F.identify(rx[:2**17], "qpsk")
    assert f.found and f.complete and f.kind == "conv" and f.name == "k7_171_133/3_4"
    assert [h.name for h in f.inner] == ["rs/204_188"] and f.inner[0].params["interleaver"] == "forney/12_17"
    assert f.profile_name == "dvbs/qpsk_3_4"
    pk = packets(f.rx()(rp.Data(f.correct(rx), sps=1, Fs=1e6, flush=True)).data)
    good = sum(np.array_equal(pk[j], ts[j]) for j in range(min(len(pk), len(ts))))
    assert good >= len(ts) - 12 # The descrambler's first frame, and the Viterbi / deinterleaver tail

def run_stage(x, blocks, stage=None):
    stage = stage or rp.sync.AcquireFEC()
    stream, outs, i = None, [], 0
    while i < len(x):
        d = rp.Data(x[i:i + blocks], Fs=1e6, sps=1, stream=stream)
        stream = d.meta.stream
        if stream.det is None: # What Acquire would have found
            stream.det = Detection(mod="qpsk", baud=1e6, sps=1.0)
        outs.append((d + stage).data)
        i += blocks
    return np.concatenate(outs), stream

def slipped(syms, carrier=None, symbol=None, seed=3):
    """The DVB-S symbols received, with the carrier slipping a quarter turn and/or the timing losing a symbol"""
    x = np.conj(syms[5:]*np.exp(1j*np.pi/2))
    if carrier is not None:
        x[carrier:] *= 1j
    if symbol is not None:
        x = np.delete(x, symbol) # The code's puncture phase moves
    return noisy(x, np.random.default_rng(seed))

def test_stage_gives_the_same_output_whatever_the_cut(dvbs):
    x = slipped(dvbs[0], carrier=160000, symbol=190000)
    outs = {}
    for blocks in (777, 50000):
        y, stream = run_stage(x, blocks)
        f = stream.est["fec"]
        assert f.profile_name == "dvbs/qpsk_3_4" and f.slips == 2
        outs[blocks] = y
    n = min(len(v) for v in outs.values())
    assert np.array_equal(outs[777][:n], outs[50000][:n])

def test_stage_turns_back_a_carrier_slip(dvbs):
    syms, ts = dvbs
    y, stream = run_stage(slipped(syms, carrier=160000), 50000)
    f = stream.est["fec"]
    assert f.slips == 1
    pk = packets(f.rx()(rp.Data(y, sps=1, Fs=1e6, flush=True)).data)
    k = next(j for j in range(len(ts)) if np.array_equal(pk[0], ts[j]))
    ok = [np.array_equal(pk[j], ts[k + j]) for j in range(min(len(pk), len(ts) - k))]
    assert sum(ok) >= len(ok) - 16 and all(ok[-40:]) # Lost: the window the slip hit, spread by the deinterleaver

def test_stage_realigns_after_a_symbol_slip(dvbs):
    """The decoder after it keeps its puncture alignment: the Viterbi output carries the packets' sync bytes after the
    slip as before (whether a symbol was lost or gained can't be told from the puncture phase: the bits after the Viterbi
    may have moved, which Align finds again)"""
    y, stream = run_stage(slipped(dvbs[0], symbol=190000), 50000)
    f = stream.est["fec"]
    assert f.slips == 1
    ops = list(f.rx())
    d = rp.Data(y, sps=1, Fs=1e6, flush=True)
    for how, op in ops[:3]: # Demap, puncture alignment, Viterbi
        d = d + op if how == "+" else d / op
    b = np.asarray(d.data).astype(np.uint8)
    tail = b[-40000:] # After the slip (at 190000 symbols, about 285000 bits in)
    v = np.lib.stride_tricks.sliding_window_view(tail ^ 1, 8) @ (1 << np.arange(7, -1, -1)) # (DVB-S inverts here)
    for bits in (v, 255 - v):
        hits = np.nonzero(np.isin(bits, (0x47, 0xB8)))[0]
        if np.bincount(hits % 1632).max() >= 0.95*len(tail)//1632:
            break
    else:
        pytest.fail("no sync bytes every 204 bytes after the slip")

def test_a_code_with_no_profile_gets_one(rng):
    code = rp.code.load("conv/k7_133_171/1_2")
    msg = rng.integers(0, 2, 40000).astype(bool)
    cw = code.encoder.encode(msg)[1:] # Not on a symbol boundary
    q = rp.symbol.load("qpsk")
    rx = noisy(q.modulate(cw[:len(cw)//2*2])*np.exp(-1j*np.pi/2), rng, 0.1)
    f = F.identify(rx, "qpsk")
    assert f.found and f.name == "k7_133_171/1_2" and f.profile_name is None and not f.inner
    u = np.asarray(f.rx()(rp.Data(f.correct(rx), sps=1, Fs=1e6, flush=True)).data).astype(bool)
    assert max(np.mean(u[64:20000] == msg[k + 64:k + 20000]) for k in range(4)) == 1.0 # (after the Viterbi's start)

def test_differential_coding_is_found(rng):
    code = rp.code.load("conv/k7_171_133/1_2")
    msg = rng.integers(0, 2, 30000).astype(bool)
    ph = rp.symbol.load("qpsk_axes").modulate(code.encoder.encode(msg))
    s = np.asarray((rp.Data(ph, sps=1, Fs=1e6, flush=True) * rp.symbol.Differential()).data)
    rx = noisy(np.conj(s*np.exp(1j*3*np.pi/4)), rng, 0.08) # On the 45 degree grid, a turn off, inverted
    f = F.identify(rx, "qpsk")
    assert f.found and f.differential and f.stream[1] == "qpsk_axes"
    u = np.asarray(f.rx()(rp.Data(f.correct(rx), sps=1, Fs=1e6, flush=True)).data).astype(bool)
    assert max(np.mean(u[64:20000] == msg[k + 64:k + 20000]) for k in range(4)) == 1.0 # (after the Viterbi's start)

def test_block_code_and_where_its_words_start(rng):
    h = rp.code.load("block/hamming/7_4")
    cw = h.encoder.encode(rng.integers(0, 2, 4*12000).astype(bool))[3:]
    f = F.identify(noisy(rp.symbol.load("bpsk").modulate(cw), rng, 0.1), "bpsk")
    assert f.found and f.kind == "block" and f.name == "hamming/7_4" and f.offset == 4 and f.period == 7

def test_hdlc_framing(rng):
    fr = rp.link.load("frame/hdlc")
    frames = [bytes(rng.integers(0, 256, int(rng.integers(20, 60))).astype(np.uint8)) for _ in range(300)]
    bits = np.asarray(fr.encoder.encode(frames)).astype(bool)
    rx = noisy(-rp.symbol.load("bpsk").modulate(bits)[7:], rng, 0.05)
    f = F.identify(rx, "bpsk")
    assert f.found and f.kind == "hdlc"
    got = np.packbits(np.asarray(f.rx()(rp.Data(f.correct(rx), sps=1, Fs=1e6, flush=True)).data).astype(bool)).tobytes()
    assert got == b"".join(frames)

def test_uncoded_symbols_have_no_code(rng):
    q = rp.symbol.load("qpsk")
    f = F.identify(noisy(q.points[rng.integers(0, 4, 2**15)].astype(np.complex64), rng, 0.1), "qpsk")
    assert not f.found and f.complete
    with pytest.raises(ValueError):
        f.profile()

def test_stage_releases_uncoded_symbols_unchanged(rng):
    q = rp.symbol.load("qpsk")
    x = noisy(q.points[rng.integers(0, 4, 40000)].astype(np.complex64), rng, 0.1)
    y, stream = run_stage(x, 4096, rp.sync.AcquireFEC(window=2**14))
    assert not stream.est["fec"].found and np.array_equal(y, x[:len(y)]) and len(y) >= 4096*9

def test_rs_scan_says_when_it_had_too_little():
    bits = np.random.default_rng(0).integers(0, 2, 8*204*10).astype(np.uint8)
    hyps, trials, short = F.scan_rs(bits)
    assert short and not hyps

def reference_conv_encode(enc, bits):
    """The register loop the vectorized encoder replaced: a zero tail when terminated, the register started with the
    frame's last bits when tail-biting, inverted outputs flipped"""
    reg = np.zeros(enc.con, dtype=bool)
    bits = np.asarray(bits, dtype=bool)
    if enc.termination == "tailbiting":
        reg[1:] = bits[::-1][:enc.con - 1]
    if enc.terminated:
        bits = np.concatenate((bits, np.zeros(enc.con - 1, dtype=bool)))
    out = np.empty((len(bits), enc.den), dtype=int)
    for i, b in enumerate(bits):
        reg[0] = b
        for j, g in enumerate(enc.generator):
            out[i, j] = np.bitwise_xor.reduce(reg[g])
        reg = np.roll(reg, 1)
        reg[0] = 0
    if enc.invert is not None:
        out ^= enc.invert[None, :]
    if enc.puncture is not None:
        P = enc.puncture.shape[1]
        full = len(out)//P*P
        out = out[:full][np.tile(enc.puncture.T, (full//P, 1))]
    return out.reshape(-1).astype(bool)

@pytest.mark.parametrize("name", rp.code.names("conv"))
def test_conv_encoder_matches_the_register_loop(name, rng):
    enc = rp.code.load(f"conv/{name}").encoder
    b = rng.integers(0, 2, 1001).astype(bool)
    assert np.array_equal(enc.encode(b), reference_conv_encode(enc, b))
    if enc.termination == "tailbiting": # Whole frames only: no stream to split
        return
    st = {}
    split = np.concatenate((enc.encode(b[:420], st, False), enc.encode(b[420:], st, True))) # Whole puncture periods (2, 3, 5, 7)
    assert np.array_equal(split, enc.encode(b))
