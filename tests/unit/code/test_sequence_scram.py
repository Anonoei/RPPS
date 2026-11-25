"""Sequence scramblers and whitening against their standards"""
import numpy as np
import pytest

import rpps as rp
from support.signals import bits

# --- sequence scramblers --------------------------------------------------------------------------------------

def test_wifi_scrambler_known_sequence():
    """IEEE 802.11 17.3.5.5: the 127-bit sequence from the all-ones state"""
    ref = ("00001110 11110010 11001001 00000010 00100110 00101110 10110110 00001100 11010100 11100111 10110100 "
           "00101010 11111010 01010001 10111000 1111111").replace(" ", "")
    assert "".join(map(str, rp.sequence.load("wifi_ofdm", seed=127).bits(127))) == ref

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
    assert list(rp.sequence.load("lte_gold", c_init=c_init).bits(400)) == _lte_c(c_init, 400)

def test_lte_channel_seeds():
    pci, ns, rnti = 301, 5, 0xFFFF
    key = lambda scr, n, start=0: list(scr.encode(np.zeros(n, bool), start).astype(int)) # a scrambler's keystream
    assert key(rp.code.scram.load("lte_pcfich", pci=pci, ns=ns), 32) == _lte_c(((ns//2 + 1)*(2*pci + 1) << 9) + pci, 32)
    assert key(rp.code.scram.load("lte_pdsch", pci=pci, ns=ns), 64) == _lte_c((rnti << 14) + ((ns//2) << 9) + pci, 64)
    pbch = rp.code.scram.load("lte_pbch", pci=pci)
    assert key(pbch, 100, 1920) == key(pbch, 100) # Restarts every four frames

@pytest.mark.parametrize("ch", [0, 17, 37, 39])
def test_ble_whitening_spec_register(ch):
    s = [1] + [(ch >> (5 - i)) & 1 for i in range(6)]
    want = []
    for _ in range(300):
        o = s[6]
        want.append(o)
        s = [o] + s[:6]
        s[4] ^= o
    assert list(rp.sequence.load("ble_whiten", channel=ch).bits(300)) == want

def test_p25p2_scrambler():
    w, sid, nac = 0xbee00, 0x8d0, 0x8d1
    seed = (w << 24) + (sid << 12) + nac
    s = [(seed >> (43 - i)) & 1 for i in range(44)]
    for t in range(4320 - 44):
        s.append(s[t] ^ s[t + 10] ^ s[t + 24] ^ s[t + 29] ^ s[t + 35] ^ s[t + 40])
    seq = rp.sequence.load("p25p2", wacn=w, sysid=sid, nac=nac)
    assert list(seq.bits(4320)) == s
    assert rp.sequence.load("p25p2", wacn=0, sysid=0, nac=0).bits(44).all() # A zero seed is replaced by all ones
    scr = rp.code.scram.load("p25p2", wacn=w, sysid=sid, nac=nac) # the scrambler restarts every superframe
    assert np.array_equal(scr.encode(np.zeros(100, bool), 4320).astype(int), s[:100])

def test_sequence_streams_and_soft(rng):
    s = rp.code.scram.load("lte_pcfich", pci=1, ns=0)
    x = bits(rng, 100)
    a = rp.Data(x[:37]) * s
    b = rp.Data(x[37:], ctx=a.ctx) * s
    assert np.array_equal(np.concatenate((a.data, b.data)), s.encode(x))
    llr = rng.normal(0, 1, 50)
    assert np.array_equal(s.soft(llr) < 0, s.encode(llr < 0))
