"""rp.analysis.spectrum.levels: the finder's level reading (find(...)), and the Finder streaming it:
a region kept from step to step, signals not found again held stale, bursts as Burst"""
import numpy as np
import pytest

from rpps.analysis import Burst
from rpps.analysis.spectrum import levels
from rpps.analysis.spectrum.find import Finder, _Tail, find

from test_find import psd_noise, rc

def test_a_line_raised_from_the_floor_reads_carriers_a_pedestal_and_a_hump():
    x = np.arange(3000.0)
    rect = lambda c, w: (np.abs(x - c) < w/2).astype(float)
    y = 10*np.log10(1 + 30*rect(1000, 1600) + 300*rect(700, 200) + 300*rect(1300, 200)  # carriers on a pedestal
                    + 100*rect(2300, 150) + 50*rect(2460, 150)                           # side by side, a notch
                    + 3*np.exp(-((x - 2800)/100)**2))                                    # a hump: sides never hold
    things = levels._read(y, 1.0)
    spans = np.array(sorted((t["lo"], t["hi"]) for t in things))
    assert spans == pytest.approx(np.array([(200, 1800), (600, 800), (1200, 1400), (2225, 2375), (2385, 2535)]), abs=1.0)
    ped = next(i for i, t in enumerate(things) if round(t["lo"]) == 200)
    assert all(t["over"] == ped for t in things if 600 <= t["lo"] < 1400)
    assert things[ped]["top"] == pytest.approx(10*np.log10(31), abs=0.1)

def test_noise_humps_and_lines_give_no_boxes():
    rng = np.random.default_rng(3)
    Fs, N = 50e6, 1 << 20
    for _ in range(2):
        assert len(find(psd_noise(rng, N, Fs, lambda f: 10**((0.5*np.sin(f/7e6) + f/25e6)/10)), Fs)) == 0
        assert len(find(psd_noise(rng, N, Fs, lambda f: 1 + 3*np.exp(-((f - 5e6)/5e6)**2)), Fs)) == 0
    t = np.arange(N)/Fs
    tone = (0.3*np.exp(2j*np.pi*7.3e6*t)).astype(np.complex64)
    sv = find(psd_noise(rng, N, Fs, lambda f: 1 + 10*rc(f - 7e6, 1e6)) + tone, Fs)
    assert len(sv) == 1 and sv[0].mid == pytest.approx(7e6, abs=10e3) # (the carrier under the tone, not the tone)

def test_tail_holds_the_newest_rows_in_order():
    rng = np.random.default_rng(0)
    t, ref = _Tail(10, (3,)), np.zeros((0, 3), np.float32)
    for _ in range(40):
        rows = rng.normal(size=(int(rng.integers(0, 14)), 3)).astype(np.float32)
        t.push(rows)
        ref = np.concatenate((ref, rows))[-10:]
        assert np.array_equal(t.view, ref)
    t.drop(4)
    assert np.array_equal(t.view, ref[4:])

def _stream(f, x, block):
    out = []
    for i in range(0, len(x) - block + 1, block):
        sv = f.step(x[i:i + block])
        if sv is not None:
            out.append(sv)
    return out

def test_finder_reads_what_find_does_over_the_frames_it_holds():
    rng = np.random.default_rng(8)
    Fs, nfft, M = 10e6, 1024, 128
    x = psd_noise(rng, 1 << 19, Fs, lambda f: 1 + 30*rc(f - 1e6, 500e3) + 10*rc(f + 2e6, 800e3))
    f = Finder(Fs, nfft=nfft, frames=M, overlap=0.0, hold=0)
    sv = _stream(f, x, 20000)[-1]
    held = x[(f._n - M)*nfft:f._n*nfft]
    ref = find(held, Fs, nfft=nfft)
    assert len(sv) == len(ref) == 2
    for a, b in zip(sv, ref):
        assert a.lo == pytest.approx(b.lo, abs=2*Fs/nfft) and a.hi == pytest.approx(b.hi, abs=2*Fs/nfft)
        assert a.snr == pytest.approx(b.snr, abs=0.3)

def test_a_signal_gone_is_held_stale_for_its_hold_and_gets_its_id_back():
    rng = np.random.default_rng(9)
    Fs, nfft, M, block = 10e6, 1024, 64, 16384
    on = psd_noise(rng, 1 << 19, Fs, lambda f: 1 + 30*rc(f - 1e6, 500e3) + 20*rc(f + 2e6, 500e3))
    off = psd_noise(rng, 1 << 19, Fs, lambda f: 1 + 30*rc(f - 1e6, 500e3))
    x = np.concatenate((on, off, on))
    dt = block/Fs
    for hold, kept in ((-1.0, None), (0.0, 0.0), (20*dt, 20*dt)):
        f = Finder(Fs, nfft=nfft, frames=M, overlap=0.0, hold=hold)
        svs = _stream(f, x, block)
        first = next(s for s in svs[len(svs)//3 - 1] if s.mid < 0)
        for sv in svs[len(svs)//3 + 8:2*len(svs)//3]: # (gone, the window past it)
            gone = [s for s in sv if s.mid < 0]
            if kept == 0.0:
                assert not gone
            elif gone:
                assert gone[0].stale and gone[0].id == first.id
        if hold < 0:
            assert any(s.stale and s.id == first.id for s in svs[2*len(svs)//3 - 1])
        if 0 < hold:
            assert not any(s.mid < 0 for s in svs[2*len(svs)//3 - 1]) # (its hold is over)
        back = [s for s in svs[-1] if s.mid < 0]
        assert len(back) == 1 and not back[0].stale
        if hold < 0: # (held to the end: its id back; past its hold it is a new signal)
            assert back[0].id == first.id

def test_bursts_are_bursts():
    rng = np.random.default_rng(10)
    Fs, N = 20e6, 1 << 19
    y = psd_noise(rng, N, Fs, lambda f: 50*rc(f - 6e6, 500e3))
    on = (np.arange(N)//(N//16)) % 2 == 1
    x = psd_noise(rng, N, Fs, lambda f: 1 + 30*rc(f + 3e6, 1e6)) + (y*on).astype(np.complex64)
    sv = find(x, Fs, nfft=1024)
    b = [s for s in sv if isinstance(s, Burst)]
    assert len(b) == 1 and b[0].mid == pytest.approx(6e6, abs=50e3)
    assert b[0].duty == pytest.approx(0.5, abs=0.1) and len(b[0].bursts) == 8
    assert b[0].last == b[0].bursts[-1] and b[0].period == pytest.approx(2*(N//16)/Fs, rel=0.05)
    assert not any(isinstance(s, Burst) for s in sv if s.mid < 0)
