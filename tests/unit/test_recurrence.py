"""Every linear recurrence up to a degree at once: scrambler taps from hard bits (rpps.analysis.recurrence)"""
import numpy as np
import pytest

from rpps.analysis import recurrence as R

N = 200000
V35 = 1 | 1 << 3 | 1 << 20          # feedthrough, ITU V.35
DVB = 1 | 1 << 14 | 1 << 15         # additive, DVB energy dispersal
DENSE = 1 | 1 << 11 | 1 << 13 | 1 << 14 | 1 << 16 # primitive, five terms: out of a trinomial search's reach

def lfsr(poly, n):
    L = poly.bit_length() - 1
    taps = [j for j in range(1, L + 1) if poly >> j & 1]
    k = np.zeros(n + L, np.uint8)
    k[0] = 1
    for i in range(L, n + L):
        v = 0
        for j in taps:
            v ^= k[i - j]
        k[i] = v
    return k[L:]

def feedthrough(d, poly):
    taps = [j for j in range(1, poly.bit_length()) if poly >> j & 1]
    s = np.zeros(len(d), np.uint8)
    for i in range(len(d)):
        v = d[i]
        for j in taps:
            if i >= j:
                v ^= s[i - j]
        s[i] = v
    return s

@pytest.fixture(scope="module")
def data():
    return (np.random.default_rng(0).random(N) < 0.2).astype(np.uint8) # idle-ish: 80% zeros

def test_feedthrough_taps_and_data_back(data):
    s = feedthrough(data, V35)
    r = R.search(s, 20)[0]
    assert r.poly == V35 and r.data_bias(additive=False) == pytest.approx(0.6, abs=0.02)
    assert np.array_equal(R.descramble_feedthrough(s, V35)[20:], data[20:])

@pytest.mark.parametrize("poly", [DVB, DENSE])
def test_additive_taps(data, poly):
    r = R.search(data ^ lfsr(poly, N), 20)[0]
    assert r.poly == poly and r.data_bias(additive=True) == pytest.approx(0.6, abs=0.03)

def test_white_bits_have_none():
    assert R.search(np.random.default_rng(1).integers(0, 2, N), 20) == []

def test_multiples_are_dropped(data):
    found = R.search(feedthrough(data, DVB), 20)
    assert found[0].poly == DVB and all(R.pmod(r.poly, DVB) for r in found[1:])

def test_gcd():
    assert R.gcd(R.pmul(DVB, 0b1011), R.pmul(DVB, 0b111)) == DVB # 1 + x + x^3 and 1 + x + x^2: coprime

def test_degree_limit():
    with pytest.raises(ValueError):
        R.search(np.zeros(100, np.uint8), 30)

@pytest.mark.parametrize("poly, p_one", [(DVB, 0.2), (DENSE, 0.1)])
def test_keystream_by_fast_correlation(poly, p_one):
    """The data 80% (90%) zeros: the keystream decoded through f and its squares and regenerated: the data comes back"""
    rng = np.random.default_rng(5)
    n = 20000
    data = (rng.random(n) < p_one).astype(np.uint8)
    k = lfsr(poly, n)
    ks = R.keystream(data ^ k, poly)
    assert np.array_equal(ks.bits, k) and ks.agree > 0.95
    assert np.array_equal(ks.data, data)

def test_soft_finds_what_hard_misses():
    """A five-term feedthrough over 70/30 data at Es/N0 -4 dB (19% raw errors): soft scoring holds it well over the
    threshold, by more than hard decisions do"""
    rng = np.random.default_rng(6)
    data = (rng.random(N) < 0.3).astype(np.uint8)
    s = feedthrough(data, DENSE)
    sigma = np.sqrt(1/(2*10**(-4/10)))
    llr = 2*(1 - 2.0*s + sigma*rng.standard_normal(N))/sigma**2
    hard = R.search((llr < 0).astype(np.uint8), 20)
    soft = R.search(None, 20, llr=llr)
    assert soft[0].poly == DENSE
    zh = next((r.z for r in hard if r.poly == DENSE), 0.0)
    assert soft[0].z > 1.3*zh

def test_high_degree_by_berlekamp_massey():
    """A scrambler (1+x^18+x^23) applied twice over a repeating 24-bit word: degree 46 and more, past the transform; the
    windows' shortest LFSR repeats, holds everywhere, and its square divides it"""
    rng = np.random.default_rng(7)
    word = rng.integers(0, 2, 24).astype(np.uint8)
    data = np.tile(word, 4000)
    s = feedthrough(feedthrough(data, 1 | 1 << 18 | 1 << 23), 1 | 1 << 18 | 1 << 23)
    C, deg, holds = R.lfsr(s)
    assert holds == 1.0 and R.pmod(C, R.pmul(1 | 1 << 18 | 1 << 23, 1 | 1 << 18 | 1 << 23)) == 0 and deg <= 46 + 24
    assert R.berlekamp_massey(lfsr(DVB, 200))[1] == 15
