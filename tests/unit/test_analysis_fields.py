"""What each bit of a set of frames is (rpps.analysis.framing.fields), and the units bursts are made of
(rpps.analysis.spectrum.burst.units)"""
import numpy as np

import rpps as rp
from rpps.analysis.framing import fields as FM
from rpps.analysis.spectrum import burst

def bits(v, w, order="msb"):
    b = [(v >> i) & 1 for i in range(w)]
    return np.array(b[::-1] if order == "msb" else b, bool)

def headers(rng, n=200, terminals=4):
    """A sync word (16), a terminal id (8), a 12-bit counter per terminal, a 3-bit type, an 8-bit length, 24 random bits,
    then a payload of that length"""
    terms = rng.integers(0, terminals, n)
    tid = rng.permutation(np.arange(1, 128))[:terminals] | 0x80 # (top bit set: the id is 8 bits wide)
    cnt = {t: int(rng.integers(0, 4096)) for t in range(terminals)}
    frames, lens = [], []
    for t in terms:
        cnt[t] = (cnt[t] + 1) % 4096
        k = int(rng.integers(20, 250))
        f = np.concatenate((bits(0x1ACF, 16), bits(int(tid[t]), 8), bits(cnt[t], 12), bits(int(rng.choice([1, 2, 5])), 3),
                            bits(k, 8), rng.integers(0, 2, 24).astype(bool), rng.integers(0, 2, k).astype(bool)))
        frames.append(f)
        lens.append(len(f))
    return frames, terms, lens

def test_header_fields_of_bursts_from_several_terminals():
    """Measured over 30 seeds: every field right; in 2 the id's edge bit is the same in all four terminals (constant)"""
    frames, terms, lens = headers(np.random.default_rng(0))
    m = FM.map(frames, groups=terms, lengths=lens)
    kinds = {f.kind: f for f in m.fields}
    assert any(f.kind == "constant" and f.start == 0 and f.width >= 16 for f in m.fields)
    i, c = kinds["id"], kinds["counter"]
    assert 16 <= i.start <= 17 and c.stop == 36 and c.detail["step"] == 1 # (the id's top bit is set in every terminal's)
    # the counter's top bits that never moved within a terminal read as id; only constant bits between them
    assert all(f.kind == "constant" for f in m.fields if i.stop <= f.start < c.start)
    assert kinds["enum"].start == 36 and set(kinds["enum"].detail["values"]) == {1, 2, 5}
    ln = kinds["length"]
    assert ln.stop == 47 and ln.detail["scale"] == 1 and ln.detail["offset"] == -71
    assert any(f.kind == "random" and f.start <= 47 for f in m.fields)

def test_a_counter_in_either_bit_order_without_groups():
    rng = np.random.default_rng(1)
    for order, step in (("msb", 1), ("lsb", 1), ("msb", -1), ("msb", 3)):
        v0 = int(rng.integers(0, 1 << 16))
        frames = [np.concatenate((bits(0xB5, 8), bits((v0 + step*k) % (1 << 16), 16, order), rng.integers(0, 2, 40).astype(bool)))
                  for k in range(300)]
        c = FM.map(frames).of("counter")
        assert len(c) == 1 and c[0].start >= 8 and c[0].stop == (24 if order == "msb" else c[0].stop)
        assert c[0].detail["step"] == step and c[0].detail["order"] == order and c[0].width >= 8

def test_random_frames_are_random():
    """Measured: 0 of 50 sets of 100 random 64-bit frames give a field other than random / varying"""
    for s in range(20):
        F = np.random.default_rng(100 + s).integers(0, 2, (100, 64)).astype(bool)
        assert {f.kind for f in FM.map(F).fields} <= {"random", "varying"}

def test_the_map_saves_as_a_header(tmp_path):
    frames, terms, lens = headers(np.random.default_rng(2))
    m = FM.map(frames, groups=terms, lengths=lens)
    m.guess.save("found", path=tmp_path)
    h = rp.link.load("header/guess/found", path=tmp_path)
    assert h is not None

def test_units_of_burst_lengths():
    """Lengths known to +-2 symbols. Measured over 40 seeds of 4 layouts: 1 in 160 more than a symbol off"""
    rng = np.random.default_rng(3)
    L = [64 + 1008*int(rng.choice([1, 2, 3])) + int(rng.integers(-2, 3)) for _ in range(30)]
    u = burst.units(L)
    assert abs(u.unit - 1008) <= 1 and abs(u.overhead - 64) <= 3 and set(u.counts) == {1, 2, 3} and u.err < 1
    assert burst.units([1070 + int(rng.integers(-2, 3)) for _ in range(20)]) is None # one length: nothing to tell
    assert sum(burst.units(np.random.default_rng(s).integers(300, 3000, 40)) is not None for s in range(20)) == 0

def test_the_profile_of_bursts_notes_their_units_and_header():
    """TDMA bursts (a 32-symbol preamble; a header of a constant word, a counter and a type; then 1 to 3 units of 128
    symbols): the profile notes the unit and the header's fields"""
    rng = np.random.default_rng(1)
    q = rp.symbol.load("qpsk")
    pre = np.asarray(q.points)[rng.integers(0, 4, 32)]
    Fs, sps = 1.6e6, 16
    shape = rp.wave.Shape(sps, rp.signal.filters.load("pulse/rrc/0p35"))
    xs = []
    for i in range(40):
        head = np.concatenate((bits(0x1ACF, 16), bits(i % 256, 8), bits(int(rng.choice([1, 2, 5])), 3), bits(0, 5)))
        body = rng.integers(0, 2, 256*int(rng.integers(1, 4))).astype(bool)
        w = np.asarray((rp.Data(np.concatenate([pre, q.modulate(np.concatenate((head, body)))]).astype(np.complex64), Fs=Fs/sps, sps=1)
                        + shape).data)
        xs += [np.zeros(int(rng.integers(1200, 2400)), np.complex64),
               w*np.exp(1j*(2*np.pi*rng.uniform(-200, 200)/Fs*np.arange(len(w)) + rng.uniform(0, 2*np.pi)))]
    x = np.concatenate(xs + [np.zeros(2000, np.complex64)])
    x = x*np.exp(2j*np.pi*150e3/Fs*np.arange(len(x)))
    y = (x + 0.02*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))).astype(np.complex64)
    p = min(rp.analysis.profile(rp.source.Memory(y, Fs=Fs), blind=False), key=lambda p: abs(p.signal.mid - 150e3))
    units = [s for s in p.steps if s.kind == "frame" and "unit" in s.detail]
    header = [s for s in p.steps if s.kind == "header"]
    assert units and abs(units[0].detail["unit"] - 128) <= 1 and set(units[0].detail["counts"]) == {1, 2, 3}
    assert header and "counter" in header[0].what
