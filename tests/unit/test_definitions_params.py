"""Load-time parameters, conv termination modes, block interleavers and sequence scramblers"""
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.errors import ConfigError
from conftest import bits

# --- parameters -----------------------------------------------------------------------------------------------

def test_param_and_expr_substitution():
    obj = {"params": {"a": {"default": 3}, "b": None}, "x": {"param": "a"}, "y": [{"expr": "(a << 2) | (b >> 2)"}],
           "z": {"each": "i*a % 5", "n": {"param": "b"}}}
    out = _config.resolve(obj, {"b": 4})
    assert (out["x"], out["y"], out["z"]) == (3, [13], [0, 3, 1, 4])

def test_param_errors():
    obj = {"params": {"pci": None}, "seed": {"param": "pci"}}
    with pytest.raises(ConfigError, match="needed"):
        _config.resolve(obj, {})
    with pytest.raises(ConfigError, match="unknown parameter"):
        _config.resolve(obj, {"pcl": 3})
    assert _config.resolve(obj, {"pcl": 3, "pci": 1}, strict=False)["seed"] == 1

@pytest.mark.parametrize("expr", ["__import__('os')", "(1).real", "open('x')", "[x for x in range(3)]", "lambda: 1"])
def test_expressions_are_not_python(expr):
    with pytest.raises(ConfigError):
        _config.evaluate(expr, {})

def test_expression_operators():
    v = {"w": 0, "n": 5}
    assert _config.evaluate("w or 2**44 - 1", v) == 2**44 - 1
    assert _config.evaluate("[3, 4, 5][n % 3] if n > 2 else -1", v) == 5
    assert _config.evaluate("max(n, 7) // 2 + abs(-1) + (~n & 3)", v) == 6

# --- convolutional termination --------------------------------------------------------------------------------

def _circular_encode(msg, gens, K):
    """36.212 5.1.3.1: the register starts with the last K-1 message bits"""
    out = []
    n = len(msg)
    for k in range(n):
        window = [msg[(k - i) % n] for i in range(K)] # current input first
        out.append([sum(window[i] & ((g >> (K - 1 - i)) & 1) for i in range(K)) & 1 for g in gens])
    return np.array(out, dtype=bool).reshape(-1)

def test_tailbiting_encoder(rng):
    gens = [0o133, 0o171, 0o165]
    code = rp.code.Conv(gens, 7, termination="tailbiting")
    for n in (7, 40, 64):
        msg = bits(rng, n).astype(int)
        assert np.array_equal(code._enc.encode(msg), _circular_encode(msg, gens, 7))

@pytest.mark.parametrize("decoder", ["viterbi", "bcjr"])
def test_tailbiting_decodes_noisy_frames(decoder, rng):
    code = rp.code.Conv([0o133, 0o171, 0o165], 7, termination="tailbiting", decoder=decoder)
    good = 0
    for _ in range(20):
        msg = bits(rng, 40)
        cw = code._enc.encode(msg)
        llr = 2*((1 - 2.0*cw) + rng.normal(0, 0.7, len(cw)))/0.49
        good += np.array_equal(code._dec.decode(llr), msg)
    assert good >= 19

def test_tailbiting_in_a_pipeline(rng):
    code = rp.code.Conv([0o133, 0o171], 7, termination="tailbiting")
    msg = bits(rng, 48)
    cw = (rp.Data(msg, flush=True) * code).data
    assert len(cw) == 96 # Nothing added
    assert np.array_equal((rp.Data(cw, flush=True) / code).data, msg)

def test_termination_names_and_octal_generators(rng):
    obj = {"base": "conv", "type": "split",
           "encode": {"type": "conv", "num": 1, "den": 2, "constraint": 7, "generator": ["0o133", "0o171"], "termination": "zero"},
           "decode": {"type": "viterbi", "num": 1, "den": 2, "generator": ["0o133", "0o171"]}}
    a = rp.code.serial.init("t", obj)
    b = rp.code.load("conv", "k7_133_171/1_2")
    assert np.array_equal(a._enc.generator, b._enc.generator)
    assert a._enc.terminated and a._dec.terminated # The decoder takes the encoder's termination and constraint
    msg = bits(rng, 50)
    assert np.array_equal(a._dec.decode(a._enc.encode(msg)), msg)
    with pytest.raises(ValueError):
        rp.code.Conv([0o133, 0o171], 7, termination="circular")

# --- block interleavers ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("ncbps,nbpsc", [(48, 1), (96, 2), (192, 4), (288, 6)])
def test_wifi_interleaver(ncbps, nbpsc, rng):
    il = rp.code.load("interleave", "wifi_ofdm", ncbps=ncbps, nbpsc=nbpsc)._enc
    s = max(nbpsc//2, 1)
    k = np.arange(ncbps)
    i = (ncbps//16)*(k % 16) + k//16                   # 17-17
    j = s*(i//s) + (i + ncbps - (16*i)//ncbps) % s     # 17-18
    x = bits(rng, ncbps)
    y = il.encode(x)
    assert np.array_equal(y[j], x)
    assert np.array_equal(il.decode(y), x)

def test_p25_tsbk_interleave():
    order = [d for r in range(12) for d in (2*r, 2*r + 1, 26 + 2*r, 27 + 2*r, 50 + 2*r, 51 + 2*r, 74 + 2*r, 75 + 2*r)] + [24, 25]
    il = rp.code.load("interleave", "p25_tsbk")._enc
    rx = np.arange(98*2) # Dibits as pairs of positions
    assert np.array_equal(il.decode(rx).reshape(98, 2)[:, 0]//2, order)

@pytest.mark.parametrize("D", [40, 57])
def test_lte_subblock(D):
    P = [1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31, 0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30]
    R = -(-D//32)
    y = [None]*(R*32 - D) + list(range(D))
    want = [y[r*32 + P[c]] for c in range(32) for r in range(R)]
    want = [v for v in want if v is not None]
    il = rp.code.load("interleave", "lte_subblock", length=D)._enc
    assert list(il.encode(np.arange(D))) == want

def test_block_interleaver_rejects_non_permutation():
    with pytest.raises(ValueError):
        rp.code.types.block_interleaver(rp.logger.Logger(), [0, 1, 1])

# --- sequence scramblers --------------------------------------------------------------------------------------

def test_wifi_scrambler_known_sequence():
    """IEEE 802.11 17.3.5.5: the 127-bit sequence from the all-ones state"""
    ref = ("00001110 11110010 11001001 00000010 00100110 00101110 10110110 00001100 11010100 11100111 10110100 "
           "00101010 11111010 01010001 10111000 1111111").replace(" ", "")
    assert "".join(map(str, rp.code.scram.load("wifi_ofdm").sequence(127))) == ref

def _lte_c(c_init, n):
    """36.211 7.2, written out"""
    N = 1600 + n
    x1, x2 = [1] + [0]*30, [(c_init >> i) & 1 for i in range(31)]
    for i in range(N):
        x1.append(x1[i + 3] ^ x1[i])
        x2.append(x2[i + 3] ^ x2[i + 2] ^ x2[i + 1] ^ x2[i])
    return [x1[k + 1600] ^ x2[k + 1600] for k in range(n)]

@pytest.mark.parametrize("c_init", [0, 1, 77, (1 << 30) | 12345])
def test_lte_gold(c_init):
    assert list(rp.code.scram.load("lte_gold", c_init=c_init).sequence(400)) == _lte_c(c_init, 400)

def test_lte_channel_seeds():
    pci, ns, rnti = 301, 5, 0xFFFF
    assert list(rp.code.scram.load("lte_pcfich", pci=pci, ns=ns).sequence(32)) == _lte_c(((ns//2 + 1)*(2*pci + 1) << 9) + pci, 32)
    assert list(rp.code.scram.load("lte_pdsch", pci=pci, ns=ns).sequence(64)) == _lte_c((rnti << 14) + ((ns//2) << 9) + pci, 64)
    pbch = rp.code.scram.load("lte_pbch", pci=pci)
    assert np.array_equal(pbch.sequence(100, 1920), pbch.sequence(100)) # Restarts every four frames

@pytest.mark.parametrize("ch", [0, 17, 37, 39])
def test_ble_whitening_spec_register(ch):
    s = [1] + [(ch >> (5 - i)) & 1 for i in range(6)]
    want = []
    for _ in range(300):
        o = s[6]
        want.append(o)
        s = [o] + s[:6]
        s[4] ^= o
    assert list(rp.code.scram.load("ble_whiten", channel=ch).sequence(300)) == want

def test_p25p2_scrambler():
    w, sid, nac = 0xbee00, 0x8d0, 0x8d1
    seed = (w << 24) + (sid << 12) + nac
    s = [(seed >> (43 - i)) & 1 for i in range(44)]
    for t in range(4320 - 44):
        s.append(s[t] ^ s[t + 10] ^ s[t + 24] ^ s[t + 29] ^ s[t + 35] ^ s[t + 40])
    seq = rp.code.scram.load("p25p2", wacn=w, sysid=sid, nac=nac)
    assert list(seq.sequence(4320)) == s
    assert rp.code.scram.load("p25p2", wacn=0, sysid=0, nac=0).sequence(44).all() # A zero seed is replaced by all ones

def test_sequence_streams_and_soft(rng):
    s = rp.code.scram.load("lte_pcfich", pci=1, ns=0)
    x = bits(rng, 100)
    a = rp.Data(x[:37]) * s
    b = rp.Data(x[37:], ctx=a.ctx) * s
    assert np.array_equal(np.concatenate((a.data, b.data)), s.encode(x))
    llr = rng.normal(0, 1, 50)
    assert np.array_equal(s.soft(llr) < 0, s.encode(llr < 0))
