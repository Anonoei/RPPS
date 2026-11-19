"""rpps.analysis.framing: the frame period, the unique word, frames cut and turned on it, frames that repeat"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis import framing as FR

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
