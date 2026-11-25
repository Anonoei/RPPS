"""Surveys: choosing, tracking and isolating the signals found"""
import numpy as np
import pytest

import rpps as rp
from support.signals import two_bands

def test_signal_fields_and_survey_pick(rng):
    Fs = 1e6
    x = two_bands(rng, 1 << 17, Fs, -200e3, 150e3, 60e3)
    sv = rp.analysis.find_signals(x, Fs, cf=100e6)
    assert isinstance(sv, list) and len(sv) == 2
    a, b = sv
    assert a.freq == pytest.approx(100e6 - 200e3, abs=5e3) and a.Fs == Fs and a.cf == 100e6
    assert a.lo < a.mid < a.hi and a.hi - a.lo == pytest.approx(a.width)
    assert a.duty > 0.9 and a.resolved
    assert sv.pick(near=140e3) is b and sv.pick(freq=100e6 - 190e3) is a and sv.pick(index=1) is b
    assert sv.pick(strongest=True) in sv
    assert [s.id for s in sv] == [0, 1] and all(s.power is not None for s in sv)

def test_survey_match_keeps_ids_frame_to_frame(rng):
    Fs = 1e6
    first = rp.analysis.find_signals(two_bands(rng, 1 << 17, Fs, -200e3, 150e3, 60e3), Fs)
    first[0].id, first[1].id = 7, 3 # Ids from earlier frames
    # The next frame: the same two, drifted a little, and a new one between them
    x = two_bands(rng, 1 << 17, Fs, -195e3, 155e3, 60e3) + two_bands(rng, 1 << 17, Fs, -20e3, 400e3, 30e3)
    nxt = rp.analysis.find_signals(x, Fs).match(first)
    by = {round(s.mid/1e4): s.id for s in nxt}
    near = lambda f: min(nxt, key=lambda s: abs(s.mid - f))
    assert near(-195e3).id == 7 and near(155e3).id == 3
    assert {near(-20e3).id, near(400e3).id} == {8, 9} # New ones count on from the highest
    assert len({s.id for s in nxt}) == len(nxt), by

def test_survey_ids_are_stable_through_a_dropout_and_a_new_signal(rng):
    """A signal missing for a survey gets its id back when it returns; a new one never takes or shifts another's id"""
    Fs = 1e6
    both = lambda: two_bands(rng, 1 << 17, Fs, -200e3, 150e3, 60e3)
    one = lambda: two_bands(rng, 1 << 17, Fs, -200e3, 600e3, 60e3) # (the second one out of band: gone)
    s0 = rp.analysis.find_signals(both(), Fs)
    ids = {round(s.mid/1e5): s.id for s in s0}
    s1 = rp.analysis.find_signals(one(), Fs).match(s0)
    assert len(s1) == 1 and s1[0].id == ids[-2] and [p.id for p, _ in s1.lost] == [ids[2]]
    s2 = rp.analysis.find_signals(both() + two_bands(rng, 1 << 17, Fs, -400e3, 600e3, 30e3), Fs).match(s1) # back, and a new one
    near = lambda sv, f: min(sv, key=lambda s: abs(s.mid - f))
    assert near(s2, -200e3).id == ids[-2] and near(s2, 150e3).id == ids[2] # (the one that was missing: its own id)
    assert near(s2, -400e3).id == 2 and s2.lost == [] # New: after every id used, never one of theirs
    s3 = rp.analysis.find_signals(both(), Fs).match(s2)
    s4 = rp.analysis.find_signals(both() + two_bands(rng, 1 << 17, Fs, 400e3, 600e3, 30e3), Fs).match(s3)
    assert near(s4, 400e3).id == 3 # (2 was used, by a signal now gone: never given again)
    s = s4
    for _ in range(9): # Remembered for `hold` surveys, then forgotten
        s = rp.analysis.find_signals(one(), Fs).match(s)
    assert all(p.id != ids[2] for p, _ in s.lost)

def test_signal_channel_isolates_and_decimates(rng):
    Fs = 1e6
    x = two_bands(rng, 1 << 17, Fs, -200e3, 150e3, 60e3)
    sv = rp.analysis.find_signals(x, Fs)
    ch = sv[1].channel()
    assert ch.decimation > 1 and ch.Fs == pytest.approx(Fs/ch.decimation)
    out = (rp.Data(x, Fs=Fs) + ch.front).data
    f, p = np.fft.fftshift(np.fft.fftfreq(len(out), 1/ch.Fs)), np.fft.fftshift(np.abs(np.fft.fft(out))**2)
    assert abs(f[np.argmax(p)]) < 40e3 # Brought to baseband
    assert p[np.abs(f) < 35e3].sum() > 50*p[np.abs(f) > 90e3].sum() # The neighbour is gone

@pytest.mark.parametrize("Fs, mid, width", [(20e6, 3e6, 200e3), (2e6, 0.0, 500e3), (10e6, -1e6, 25e3)])
def test_channel_settle_is_the_start_up_transient(Fs, mid, width):
    """Channel.settle: the first output samples of a new stream, before every filter is full, and no more (a carrier at
    the channel's centre is flat from there on); delay is what meta.delay gains"""
    from rpps.analysis.spectrum.signal import Signal
    ch = Signal(mid, width, 0, 0, Fs=Fs).channel()
    x = np.exp(2j*np.pi*mid/Fs*np.arange(int(Fs*0.01))).astype(np.complex64)
    out = rp.Data(x, Fs=Fs) + ch.front
    assert out.meta.delay == pytest.approx(ch.delay)
    y = np.abs(out.data)
    level = np.median(y[len(y)//2:])
    off = np.nonzero(np.abs(y - level) > 1e-3*level)[0]
    assert off.max() < ch.settle <= off.max() + 8 # all of the transient, a few samples spare at most

def test_survey_from_a_file_keeps_its_place(rng, tmp_path):
    Fs = 1e6
    x = two_bands(rng, 1 << 17, Fs, -200e3, 150e3, 60e3)
    path = tmp_path / "cap.cf32"
    x.tofile(path)
    src = rp.source.File("cf32", str(path), Fs=Fs, cf=50e6)
    src.seek(1000)
    sv = rp.analysis.survey(src, samples=1 << 16)
    assert src.cur_samp == 1000 # Peeking doesn't use samples up
    assert len(sv) == 2 and len(sv.head) == 1 << 16 and sv[0].cf == 50e6

def test_choose_without_a_display_takes_the_strongest(rng, capsys):
    x = two_bands(rng, 1 << 16, 1e6, -200e3, 150e3, 60e3)
    sv = rp.analysis.find_signals(x, 1e6)
    matplotlib = pytest.importorskip("matplotlib") # choose() asks matplotlib whether there is a display
    matplotlib.use("Agg")
    assert sv.choose() is sv.pick(strongest=True)
    assert "strongest" in capsys.readouterr().out

def test_baseband_one_call():
    Fs, n = 1e6, 200000
    t = np.arange(n)/Fs
    x = (np.exp(2j*np.pi*201e3*t) + np.exp(2j*np.pi*-300e3*t)).astype(np.complex64) # the channel at 200 kHz, a tone 1 kHz up; another far off
    y, Fs_out = rp.signal.baseband(x, Fs, 200e3, 20e3)
    assert Fs_out < 100e3 and abs(len(y) - n*Fs_out/Fs) < 200
    mid = y[len(y)//4:3*len(y)//4]
    f = np.angle(np.mean(mid[1:]*np.conj(mid[:-1])))*Fs_out/(2*np.pi)
    assert f == pytest.approx(1e3, abs=5) and np.mean(np.abs(mid)) == pytest.approx(1.0, rel=0.05)
