"""5G NR polar code and its rate matching (rpps.code.types.nr_polar) against 3GPP TS 38.212 5.3.1, 5.4.1"""
import sys

import numpy as np
import pytest

import rpps as rp

nrp = sys.modules["rpps.code.types.nr_polar"] # (the package's name nr_polar is the class)

def test_tables_are_the_standards():
    q = nrp.sequence(1024)
    assert sorted(q) == list(range(1024)) and q[:8].tolist() == [0, 1, 2, 4, 8, 16, 32, 3] and q[-1] == 1023
    assert nrp.sequence(8).tolist() == [0, 1, 2, 4, 3, 5, 6, 7] # The order below N kept
    pi = nrp.input_pattern(164)
    assert sorted(pi) == list(range(164)) and pi[:6].tolist() == [0, 2, 4, 7, 9, 14]
    assert sorted(nrp.input_pattern(56)) == list(range(56)) # A shorter message: the entries >= 164 - K, less 164 - K

@pytest.mark.parametrize("K,E,nmax,n", [(56, 864, 9, 9), (40, 108, 9, 7), (63, 432, 9, 9), (12, 20, 10, 5), (100, 120, 9, 7),
                                        (30, 290, 10, 8)])
def test_code_size(K, E, nmax, n):
    """5.3.1: n1 from E (one less when E is barely past a power of 2 and the rate is low), n2 = ceil(log2(8K)), n >= 5"""
    assert nrp.code_size(K, E, nmax) == n

def test_subblock_interleaver():
    J = nrp.subblock(64) # 32 sub-blocks of 2
    assert J[:8].tolist() == [0, 1, 2, 3, 4, 5, 8, 9] and sorted(J) == list(range(64))

def test_triangular_interleaver_is_a_permutation():
    for E in (1, 2, 3, 10, 100, 1000):
        assert sorted(nrp.triangle(E)) == list(range(E))
    assert nrp.triangle(6).tolist() == [0, 3, 5, 1, 4, 2] # T = 3: rows of 3, 2, 1, read by columns

@pytest.mark.parametrize("K,E,kind", [(56, 864, "repeat"), (40, 108, "puncture"), (100, 120, "shorten")])
def test_rate_matching_freezes_what_it_does_not_send(K, E, kind):
    c = nrp.nr_polar(rp.logger.Logger(), K, E)
    N = c.N
    assert len(c.info) == K and c.frozen.sum() == N - K
    sent = np.zeros(N, bool)
    if kind == "repeat":
        assert E >= N
        return
    if kind == "puncture":
        assert c.punctured
        sent[c.J[N - E:]] = True
    else:
        assert not c.punctured
        sent[c.J[:E]] = True
    assert not np.any(~sent[c.info]) # (an unsent coded bit's index is frozen)

def test_pbch_round_trip_at_low_snr(rng):
    code = rp.code.load("block/polar/nr_pbch")
    assert (code.encoder.N, code.encoder.num, code.encoder.den) == (512, 32, 864)
    good = 0
    for _ in range(10):
        m = rng.integers(0, 2, 32).astype(bool)
        x = code.encoder.encode(m)
        good += np.array_equal(code.decoder.decode(2*((1 - 2.0*x) + rng.normal(0, 1.0, len(x)))), m)
    assert good == 10

@pytest.mark.parametrize("K,E", [(63, 432), (40, 108), (45, 216), (100, 120), (39, 864)])
def test_dci_rnti_mask(K, E, rng):
    """The RNTI on the CRC: a decoder with that mask checks the path; any path's residue is its RNTI"""
    c = rp.code.load("block/polar/nr_dci", K=K, E=E, mask=0x1234)
    m = rng.integers(0, 2, K - 24).astype(bool)
    x = c.encoder.encode(m)
    llr = 4*((1 - 2.0*x) + rng.normal(0, 0.5, len(x)))
    out, err = c.decoder.decode(llr, status=True)
    assert np.array_equal(out, m) and err[0] >= 0
    paths, _ = c.decoder.paths(llr)
    assert c.decoder.residue(paths[:, :1])[0, 0] == 0x1234
    other = rp.code.load("block/polar/nr_dci", K=K, E=E, mask=0xFFFF)
    assert other.decoder.decode(llr, status=True)[1][0] == -1 # Another RNTI: no path checks

def test_bil_round_trip(rng):
    c = nrp.nr_polar(rp.logger.Logger(), 30, 200, nmax=10, interleave=False, bil=True,
                     crc=rp.link.load("crc/crc11_nr").encoder)
    m = rng.integers(0, 2, 19).astype(bool)
    x = c.encode(m)
    assert np.array_equal(c.decode(4*(1 - 2.0*x)), m)

# ---- against py3gpp (tests/unit/vectors/nr_py3gpp.json: what it is in its "source")

from pathlib import Path
import json

VECTORS = json.loads((Path(__file__).parent/"vectors"/"nr_py3gpp.json").read_text())

def unhex(h, n):
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), np.uint8))[:n]

@pytest.mark.parametrize("v", VECTORS["polar"], ids=lambda v: f"K{v['K']}_E{v['E']}")
def test_polar_matches_py3gpp(v):
    """Where py3gpp's code is NR's (E >= N: repetition) the bits sent are the same; elsewhere the encoder is, at its N with
    no bits frozen for rate matching"""
    from rpps.code.types.polar import encode_u
    K, N = v["K"], v["N"]
    c = unhex(v["c"], K)
    pi = nrp.input_pattern(K) if v["interleave"] else np.arange(K)
    u = np.zeros(N, np.uint8)
    u[np.sort(nrp.sequence(N)[-K:])] = c[pi]
    assert np.array_equal(encode_u(u), unhex(v["d"], N))
    if v["e"] is not None:
        code = nrp.nr_polar(rp.logger.Logger(), K, v["E"], v["nmax"], v["interleave"])
        assert code.N == N and np.array_equal(code.encode(c), unhex(v["e"], v["E"]).astype(bool))

@pytest.mark.parametrize("v", VECTORS["crc"], ids=lambda v: v["poly"])
def test_crcs_match_py3gpp(v):
    name = {"24C": "crc24_nr_c", "24A": "crc24_lte_a", "24B": "crc24_lte_b", "16": "crc16_xmodem", "11": "crc11_nr", "6": "crc6_nr"}[v["poly"]]
    m = unhex(v["m"], v["n"])
    width = rp.link.load(f"crc/{name}").encoder.width
    got = np.asarray(rp.link.load(f"crc/{name}").encoder.field(m)).astype(np.uint8)
    assert np.array_equal(got, unhex(v["c"], v["n"] + width)[v["n"]:])
