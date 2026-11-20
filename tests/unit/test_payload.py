"""Framing signatures in decoded bits, and the verdict on the rest (rpps.analysis.framing.payload)"""
import numpy as np

from rpps.analysis.framing import payload as P
from rpps.math import gf2

def rbits(rng, n):
    return rng.integers(0, 2, n).astype(np.uint8)

def bits_of(byts):
    return np.unpackbits(np.asarray(byts, dtype=np.uint8))

def shifted(rng, b, off=None):
    off = int(rng.integers(1, 200)) if off is None else off
    return np.concatenate((rbits(rng, off), b)), off

def test_e1():
    rng = np.random.default_rng(0)
    frames = []
    for f in range(400):
        fr = rbits(rng, 256)
        if f % 2 == 0:
            fr[1:8] = [0, 0, 1, 1, 0, 1, 1]
        else:
            fr[1] = 1
        frames.append(fr)
    b, off = shifted(rng, np.concatenate(frames))
    f = P.e1(b)
    assert f is not None and f.period == 512 and (f.offset - off - 1) % 512 == 0

def test_t1_superframe():
    rng = np.random.default_rng(1)
    pat = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0]
    frames = [np.concatenate(([pat[i % 12]], rbits(rng, 192))) for i in range(600)]
    b, off = shifted(rng, np.concatenate(frames))
    f = P.t1(b)
    assert f is not None and f.period == 193 and (f.offset - off) % 193 == 0 and "SF" in f.name

def test_atm():
    rng = np.random.default_rng(2)
    cells = []
    for _ in range(300):
        h = rng.integers(0, 256, 4)
        hec = int(P._crc(h[None, :], 0x07, 8)[0]) ^ 0x55
        cells.append(np.concatenate((h, [hec], rng.integers(0, 256, 48))))
    b, off = shifted(rng, bits_of(np.concatenate(cells)))
    f = P.atm(b)
    assert f is not None and (f.offset - off) % 424 == 0

def test_gfp():
    rng = np.random.default_rng(3)
    out = []
    for _ in range(60):
        pli = int(rng.integers(40, 200))
        core = np.array([pli >> 8, pli & 0xFF])
        chec = int(P._crc(core[None, :], 0x1021, 16)[0])
        hdr = np.array([pli >> 8, pli & 0xFF, chec >> 8, chec & 0xFF]) ^ np.array([0xB6, 0xAB, 0x31, 0xE0])
        out.append(np.concatenate((hdr, rng.integers(0, 256, pli))))
    b, off = shifted(rng, bits_of(np.concatenate(out)), 24)
    f = P.gfp(b)
    assert f is not None and f.hits > 40

def test_hdlc():
    rng = np.random.default_rng(4)
    out = []
    for _ in range(200):
        data = rbits(rng, 400)
        st, ones = [], 0
        for x in data: # bit stuffing
            st.append(x)
            ones = ones + 1 if x else 0
            if ones == 5:
                st.append(0)
                ones = 0
        out.append(np.concatenate(([0, 1, 1, 1, 1, 1, 1, 0], st)))
    assert P.hdlc(np.concatenate(out)) is not None

def test_ipv4():
    rng = np.random.default_rng(5)
    out = []
    for _ in range(20):
        h = rng.integers(0, 256, 20)
        h[0], h[2], h[3], h[10], h[11] = 0x45, 0, 60, 0, 0
        w = (h[0::2].astype(np.int64) << 8) | h[1::2]
        s = int(w.sum())
        while s >> 16:
            s = (s & 0xFFFF) + (s >> 16)
        c = ~s & 0xFFFF
        h[10], h[11] = c >> 8, c & 0xFF
        out.append(np.concatenate((h, rng.integers(0, 256, 40))))
    b, off = shifted(rng, bits_of(np.concatenate(out)), 3)
    f = P.ipv4(b)
    assert f is not None and f.hits >= 18

def test_ccsds_and_mpeg():
    rng = np.random.default_rng(6)
    asm = bits_of([0x1A, 0xCF, 0xFC, 0x1D])
    b = np.concatenate([np.concatenate((asm, rbits(rng, 2016))) for _ in range(30)])
    f = P.ccsds(b)
    assert f is not None and f.period == 2048
    ts = np.concatenate([bits_of(np.concatenate(([0x47], rng.integers(0, 256, 187)))) for _ in range(60)])
    f = P.mpeg_ts(shifted(rng, ts, 5)[0])
    assert f is not None and f.period == 1504 and f.offset == 5

def test_fixed_word():
    rng = np.random.default_rng(7)
    b = np.concatenate([np.concatenate((rbits(rng, 100), np.ones(24, np.uint8), rbits(rng, 500))) for _ in range(40)])
    f = P.fixed_word(b)
    assert f is not None and f.period == 624 and f.hits >= 24

def test_random_bits_have_nothing():
    assert P.signatures(rbits(np.random.default_rng(8), 200000)) == []

def test_verdicts():
    rng = np.random.default_rng(9)
    assert P.verdict(rbits(rng, 100000)).kind == "random-looking"
    # an LFSR keystream (taps 15, 14) over idle zeros
    k = np.zeros(100000, np.uint8)
    k[:15] = 1
    for i in range(15, len(k)):
        k[i] = k[i - 15] ^ k[i - 14]
    v = P.verdict(k)
    assert v.kind == "scrambled" and v.lc < 0.1
    assert P.verdict((rng.random(100000) < 0.3).astype(np.uint8)).kind == "structured"
    assert gf2.berlekamp_massey(k[:200])[1] == 15
