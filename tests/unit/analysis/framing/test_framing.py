"""rpps.analysis.framing.frames: the frame period, the unique word, frames cut and turned on it, frames that repeat"""
import numpy as np
from types import SimpleNamespace

import rpps as rp
from rpps.analysis.framing import frames as FR

def framed(period=600, uw_len=16, frames=120, snr_db=15, slips=True, repeat=None, seed=0):
    """QPSK symbols in frames: a fixed unique word, then random data (or data that repeats every `repeat` frames), a
    carrier that turns by a multiple of 90 degrees now and then (a feed-forward carrier's ambiguity)"""
    rng = np.random.default_rng(seed)
    q = rp.symbol.load("qpsk").points
    uw = q[rng.integers(0, 4, uw_len)]
    data = [q[rng.integers(0, 4, period - uw_len)] for _ in range(repeat or frames)]
    y = np.concatenate([np.concatenate((uw, data[k % len(data)])) for k in range(frames)])
    if slips:
        turn = np.repeat(rng.integers(0, 4, frames//10 + 1), 10*period)[:len(y)]
        y = y*np.exp(1j*np.pi/2*turn)
    lead = rng.integers(0, period)
    y = np.roll(y, -lead)
    n = (rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y)))*np.sqrt(10**(-snr_db/10)/2)
    return y + n, uw, (period - lead) % period

def test_the_period_and_unique_word_are_found():
    y, uw, at = framed()
    fr = FR.find(y, "qpsk")
    assert fr is not None and fr.period == 600 and fr.uw_start == at and abs(fr.uw_length - 16) <= 1
    assert fr.coherence > 0.8 and fr.floor < 0.2

def test_frames_are_turned_onto_the_first():
    y, uw, at = framed()
    fr = FR.find(y, "qpsk")
    f = FR.cut(y, fr, 4)
    u = f[:, :16]
    assert np.median(np.abs(np.sum(u*np.conj(u[0]), axis=1))/16) > 0.85 # Every frame's unique word agrees with the first (15 dB)
    bits = FR.payload_bits(f, fr, "qpsk")
    assert bits.shape == (len(f), 2*(600 - fr.uw_length))

def test_frames_that_repeat_are_found():
    # Data repeating every 4 frames: averaged over frames it no longer cancels, so `find` can't tell the unique word from
    # it (a limit: the frame is given here); `repeats` finds the 4
    y, uw, at = framed(repeat=4, slips=False, snr_db=30)
    fr = FR.Frame(period=600, uw_start=at, uw_length=16)
    p, same = FR.repeats(FR.payload_bits(FR.cut(y, fr, 4), fr, "qpsk"))
    assert p == 4 and same > 0.9

def test_random_symbols_have_no_frame():
    rng = np.random.default_rng(1)
    q = rp.symbol.load("qpsk").points
    assert FR.find(q[rng.integers(0, 4, 60000)], "qpsk") is None

def test_noisy_random_symbols_have_no_frame():
    """A coherent position or a few by chance (or one loud sample) is not a unique word: 65536 noisy QPSK symbols with no
    framing gave a different "period" in two searches of three (a pyspecan report)"""
    q = rp.symbol.load("qpsk").points
    for seed in range(8):
        rng = np.random.default_rng(seed)
        y = q[rng.integers(0, 4, 65536)] + np.sqrt(0.05)*(rng.standard_normal(65536) + 1j*rng.standard_normal(65536))
        assert FR.find(y, "qpsk") is None, seed

def test_few_frames_or_a_weak_signal_still_found():
    """A long word in only 13 frames (10 dB), and a 0 dB signal over many frames, stand out against chance"""
    y, uw, at = framed(period=1000, uw_len=26, frames=13, snr_db=10, slips=False, seed=3)
    fr = FR.find(y, "qpsk")
    assert fr is not None and fr.period == 1000 and fr.uw_start == at
    y, uw, at = framed(period=1000, uw_len=26, frames=65, snr_db=0, slips=False, seed=4)
    fr = FR.find(y, "qpsk")
    assert fr is not None and fr.period == 1000 and abs(fr.uw_start - at) <= 2

def test_a_short_word_is_found_from_few_frames():
    """A 16-symbol word every 2000 QPSK symbols at 5 dB Es/N0: the period is a candidate by the correlation summed over
    its multiples. Over 30 seeds, 40 frames found it every time (the plain correlation: half of them; 80 frames to reach
    nine in ten)"""
    q = rp.symbol.load("qpsk").points
    found = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        uw = q[rng.integers(0, 4, 16)]
        y = q[rng.integers(0, 4, 2000*40 + 333)]
        for k in range(40):
            y[333 + k*2000:333 + k*2000 + 16] = uw
        y = y + np.sqrt(10**-0.5/2)*(rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y)))
        fr = FR.find(y, "qpsk")
        found += fr is not None and fr.period == 2000 and abs(fr.uw_start - 333) <= 2
    assert found >= 9

def test_a_structure_few_frames_share_is_not_a_word():
    """Over thousands of short "frames" symbols that only some share (pilot blocks at a short lag) are significant,
    but a unique word is in every frame: at coherence 0.12 they are not one"""
    rng = np.random.default_rng(5)
    q = rp.symbol.load("qpsk").points
    y = q[rng.integers(0, 4, 131072)]
    for k in range(0, len(y) - 64, 32*8): # One period-32 block in 8 holds the same 3 symbols as the next one
        y[k + 5:k + 8] = y[k + 37:k + 40] = q[rng.integers(0, 4, 3)]
    y = y + np.sqrt(0.05)*(rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y)))
    fr = FR.find(y, "qpsk", max_lag=len(y)//8, min_coherence=0.0)
    assert fr is not None and fr.period == 32 and fr.coherence < 0.2 # (significant, without the floor)
    fr = FR.find(y, "qpsk", max_lag=len(y)//8)
    assert fr is None or fr.coherence > 0.2

def test_bursts_are_acquired_one_by_one_and_their_preamble_found():
    # Six QPSK bursts from different transmitters (each its own phase and carrier offset), a 48-symbol preamble then data
    rng = np.random.default_rng(3)
    fs, baud = 1e6, 100e3
    q = rp.symbol.load("qpsk").points
    pre = q[rng.integers(0, 4, 48)]
    x = np.zeros(400000, complex)
    for k in range(6):
        w = FR.word_waveform(np.concatenate((pre, q[rng.integers(0, 4, 400)])), fs/baud)
        w = w*np.exp(1j*rng.uniform(0, 2*np.pi) + 2j*np.pi*rng.uniform(-2e3, 2e3)/fs*np.arange(len(w)))
        a = 10000 + k*60000 + int(rng.integers(0, 50))
        x[a:a + len(w)] += w
    x += 0.05*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    B = FR.bursts(x, fs)
    assert len(B) == 6 and all(b.det.mod == "qpsk" and abs(b.det.baud - baud) < 200 for b in B)
    n, d, offs = FR.preamble(B)
    assert 44 <= n <= 48 # The preamble (its 47 phase steps), give or take the first symbols' filter edge
    assert FR.find_word(x, fs, pre, [baud])[0].score > 0.8 # Then searchable as a known word

def test_cut_puts_the_payload_on_the_grid_whatever_the_first_frame():
    # A first frame whose word is noisy must not turn the rest: the reference is the word averaged over the frames
    y, uw, at = framed(snr_db=12)
    fr = FR.find(y, "qpsk")
    y = y.copy()
    y[at:at + 600] *= np.exp(1j*0.5) # The first whole frame turned by 29 degrees
    f = FR.cut(y, fr)
    a = np.degrees(np.angle(f[1:, fr.uw_length:])) % 90
    assert np.median(np.abs(a - 45)) < 8 # On the QPSK diagonals

def test_find_sync_and_decode_conv():
    rng = np.random.default_rng(1)
    B = rng.integers(0, 256, 204*8, dtype=np.uint8)
    B[::204] = 0x47
    bits = np.concatenate((rng.integers(0, 2, 3, dtype=np.uint8), np.unpackbits(B)))
    frac, off, phase = rp.analysis.framing.bits.find_sync(bits)
    assert frac == 1.0 and off == 3 and phase == 0
    code = rp.code.load("conv/k7_133_171/1_2")
    msg = rng.integers(0, 2, 400).astype(bool)
    hyp = SimpleNamespace(coding=("conv", "k7_133_171/1_2"), params={"offset": 0})
    out = rp.analysis.code.conv.mother.decode_conv(code.encoder.encode(msg).astype(np.uint8), hyp)
    assert (np.asarray(out[:300]).astype(bool) == msg[:300]).all()
