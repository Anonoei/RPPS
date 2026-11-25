"""DVB-S through its profile"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from support.signals import hard

# ---- DVB-S: what `chain` did, as a profile of ordinary steps

DVBS_KNOWN = {"timing": {"method": "known"}, "carrier": {"method": "none"}}

def ts(rng, n):
    pk = rng.integers(0, 256, (n, 188)).astype(np.uint8)
    pk[:, 0] = 0x47
    return pk

def dvbs_packets(out):
    b = np.packbits(hard(out))
    return b[:len(b)//188*188].reshape(-1, 188)

def first_match(got, pk, startup=1):
    """(output index, input index) of the first output packet found in the input; the first `startup` outputs may carry
    the decoder's start-up errors (a Viterbi that starts mid-trellis)"""
    for i in range(min(startup + 1, len(got))):
        k = next((k for k in range(len(pk)) if np.array_equal(got[i], pk[k])), None)
        if k is not None:
            return i, k
    return None, None

@pytest.mark.parametrize("turn", [1, -1]) # -1: the carrier 180 degrees round: every decoded bit inverted, and found
def test_dvbs_round_trip(turn):
    rng = np.random.default_rng(5)
    p = rp.profile.load("dvbs/qpsk_3_4", sync=DVBS_KNOWN)
    pk = ts(rng, 8*12)
    wave = p.tx()(rp.Data(pk.reshape(-1), Fs=1e6, sps=1, flush=True))
    out = p.rx()(rp.Data(turn*wave.data, Fs=wave.Fs, sps=wave.sps, delay=wave.delay, flush=True))
    got = dvbs_packets(out)
    est = out.stream.est
    assert est["packets"]["polarity"] == turn and est["frames"]["locked"]
    i, k = first_match(got, pk, startup=0)
    assert i == 0 and len(got) >= len(pk) - 16 # latency: the deinterleaver's 11 packets, the Viterbi tail
    assert np.array_equal(got, pk[k:k + len(got)])

def test_dvbs_finds_the_puncture_phase():
    """A receiver that starts mid-stream on channel bits: two coded bits missing, so the puncture pattern starts elsewhere"""
    rng = np.random.default_rng(6)
    p = rp.profile.load("dvbs/qpsk_3_4", sync=DVBS_KNOWN)
    pk = ts(rng, 8*10)
    tx_ops = [op for how, op in p.tx() if how == "*"][:-1] # the code steps: data bits to channel bits (no QPSK)
    d = rp.Data(pk.reshape(-1), flush=True)
    for op in tx_ops:
        d = d * op
    coded = np.asarray(d.data).astype(bool)[2:]
    rx = p.rx(start=Rep.CHANNEL)
    out = rx(rp.Data(coded, flush=True).derive(coded, rp.Category.CODE, rp.dtype.BITS, rep=Rep.CHANNEL))
    assert out.stream.est["puncture"]["offset"] != 0
    got = dvbs_packets(out)
    i, k = first_match(got, pk)
    assert i is not None and len(got) - i >= len(pk) - 16
    assert np.array_equal(got[i:], pk[k:k + len(got) - i])
