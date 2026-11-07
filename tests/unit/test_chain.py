import numpy as np
import pytest

import rpps as rp

pytestmark = pytest.mark.slow # The pure-Python Viterbi dominates: deselect with -m "not slow"

def ts_packets(rng, n):
    p = rng.integers(0, 256, (n, 188)).astype(np.uint8)
    p[:, 0] = 0x47
    return p

def test_dvbs_chain_round_trip(rng):
    chain = rp.Chain.load("dvbs_3_4")
    pk = ts_packets(rng, 8*16) # Whole 8-packet scrambler frames
    coded = chain.encode(pk)
    assert coded.dtype == bool
    res = chain.decode(coded)
    got = np.asarray(res.packets)
    assert res.stats["rs_ok"] > 0.9
    n = min(len(got), len(pk))
    assert n >= 8*8
    assert np.array_equal(got[:n][8:n-8], pk[:n][8:n-8]) # Ends: interleaver / Viterbi latency

def test_dvbs_chain_survives_inverted_bits(rng):
    chain = rp.Chain.load("dvbs_3_4")
    pk = ts_packets(rng, 8*16)
    coded = chain.encode(pk)
    res = chain.decode(~coded)
    assert res.stats["rs_ok"] > 0.9

def test_chain_in_the_operator_syntax(rng):
    chain, qpsk = rp.Chain.load("dvbs_3_4"), rp.mod.load("QPSK")
    pk = ts_packets(rng, 8*16)
    syms = rp.Data(pk.reshape(-1), flush=True) * chain * qpsk
    out = syms / qpsk / chain # Soft decisions from the demodulator go to the Viterbi decoder
    got = out.data.reshape(-1, 188)
    assert chain.stats(out)["rs_ok"] > 0.9
    n = min(len(got), len(pk))
    assert np.array_equal(got[8:n-8], pk[8:n-8])

def test_chain_handles_a_scrambler_without_the_dvb_sync_byte(rng):
    chain = rp.Chain.load("dvbs_3_4")
    assert chain._load("scram", ["adt", "dvb_energy"]).frame.get("sync", 0xB8 ^ 0xFF) == 0x47
