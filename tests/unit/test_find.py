"""rp.analysis.spectrum.find: boxes on signals at a false-alarm rate (tools/finder_bench.py scores it on many more scenes)"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis.spectrum.find import Finder, find, _gamma_isf, _norm_isf, _pelt, _pelt_py, _run_min

from conftest import shaped

def psd_noise(rng, N, Fs, psd):
    """Complex Gaussian samples whose power spectrum is psd(f) [linear, 1: the noise floor; f in Hz]"""
    f = np.fft.fftfreq(N, 1/Fs)
    g = (rng.standard_normal(N) + 1j*rng.standard_normal(N))/np.sqrt(2)
    return (np.fft.ifft(g*np.sqrt(psd(f)))*np.sqrt(N)).astype(np.complex64)

def rc(f, baud, beta=0.35):
    """A raised-cosine spectrum, 1 in its flat part"""
    a = np.abs(f)
    lo, hi = (1 - beta)*baud/2, (1 + beta)*baud/2
    return np.where(a <= lo, 1.0, np.where(a <= hi, 0.5*(1 + np.cos(np.pi/(beta*baud)*(a - lo))), 0.0))

def test_statistics():
    assert _norm_isf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _norm_isf(0.025) == pytest.approx(1.959964, abs=1e-5)
    assert _norm_isf(1e-6) == pytest.approx(4.753424, abs=1e-4)
    rng = np.random.default_rng(0) # Wilson-Hilferty against a sampled Gamma(n, 1/n)
    for n in (8, 64):
        x = rng.gamma(n, 1/n, 400000)
        assert np.mean(x > _gamma_isf(1e-3, n)) == pytest.approx(1e-3, rel=0.25)

def test_running_minimum():
    rng = np.random.default_rng(1)
    v = rng.normal(size=257)
    pad = np.concatenate((np.full(6, v[0]), v, np.full(6, v[-1])))
    assert np.allclose(_run_min(v, 6), np.lib.stride_tricks.sliding_window_view(pad, 13).min(axis=1))

def test_noise_gives_no_boxes_at_the_false_alarm_rate():
    rng = np.random.default_rng(2)
    boxes = 0
    for _ in range(40): # pfa 1e-3 a capture: 0.04 expected
        x = ((rng.standard_normal(1 << 16) + 1j*rng.standard_normal(1 << 16))/np.sqrt(2)).astype(np.complex64)
        boxes += len(find(x, 1e6, nfft=1024))
    assert boxes <= 1

def test_a_receivers_passband_and_a_smooth_hump_are_no_signals():
    rng = np.random.default_rng(3)
    Fs = 50e6
    def shape(f):
        a = np.abs(f)/(Fs/2)
        roll = np.where(a > 0.92, 10**(-4*((a - 0.92)/0.08)**2), 1.0)
        hump = 10**(5*np.exp(-((f - 8e6)/5e6)**2)/10)
        return roll*10**((3*f/(Fs/2) + 0.5*np.sin(2*np.pi*f/Fs*3))/10)*hump # 6 dB tilt, ripple, a 5 dB hump
    assert len(find(psd_noise(rng, 1 << 20, Fs, shape), Fs)) == 0

def test_carriers_side_by_side_at_different_levels_are_each_boxed():
    rng = np.random.default_rng(4)
    Fs, baud = 50e6, 1e6
    centres = -6e6 + np.arange(5)*1.25*1.35*baud
    levels = [10, 16, 11, 18, 13] # dB over the noise, steps of 5-7 dB, notches between
    x = psd_noise(rng, 1 << 20, Fs, lambda f: 1 + sum(10**(l/10)*rc(f - c, baud) for c, l in zip(centres, levels)))
    sv = find(x, Fs)
    assert len(sv) == 5
    for s, c, l in zip(sv, centres, levels):
        assert s.mid == pytest.approx(c, abs=0.03*baud)
        assert 1.0 < s.width/baud < 1.3 # the -10 dB width of an RRC carrier, beta 0.35
        # SNR over the noise in its box: the flat part at l dB, the skirts under it
        assert s.snr == pytest.approx(l - 0.6, abs=1.2)

def test_a_weak_carrier_between_strong_ones_is_found_over_the_floor():
    rng = np.random.default_rng(5)
    Fs, baud = 50e6, 500e3
    c = [2e6, 2e6 + 1.3*1.35*baud, 2e6 + 2.6*1.35*baud]
    x = psd_noise(rng, 1 << 20, Fs, lambda f: 1 + 100*rc(f - c[0], baud) + 2*rc(f - c[1], baud) + 100*rc(f - c[2], baud))
    sv = find(x, Fs)
    weak = min(sv, key=lambda s: abs(s.mid - c[1]))
    assert len(sv) == 3 and weak.mid == pytest.approx(c[1], abs=0.05*baud)
    assert weak.snr == pytest.approx(10*np.log10(2) - 0.6, abs=1.5) # its own level, not the neighbours' skirts

def test_carriers_on_a_pedestal_are_measured_over_it():
    rng = np.random.default_rng(6)
    Fs, baud = 50e6, 400e3
    ped = lambda f: 1 + 30*(np.abs(f + 2e6) < 12e6) # 15 dB plateau, 24 MHz wide, steep sides
    cs = [-10e6, -6e6, -1e6, 5e6]
    x = psd_noise(rng, 1 << 20, Fs, lambda f: ped(f) + sum(31*8*rc(f - c, baud) for c in cs)) # 9 dB over it
    sv = find(x, Fs)
    pedestal = [s for s in sv if s.holds]
    carriers = [s for s in sv if not s.holds]
    assert len(pedestal) == 1 and pedestal[0].holds == 4 and pedestal[0].width == pytest.approx(24e6, rel=0.05)
    assert len(carriers) == 4 and all(abs(s.mid - c) < 0.05*baud for s, c in zip(carriers, cs))
    for s in carriers:
        assert s.snr == pytest.approx(10*np.log10(8) - 0.6, abs=1.5)

@pytest.mark.parametrize("jam_baud, jam_db, off", [(100e3, 25, 0.0), (300e3, 20, 1.2e6), (1e6, 22, 0.5e6)])
def test_a_narrow_signal_on_a_wider_one_is_found_over_it(jam_baud, jam_db, off):
    # A jammer on a 4 MBd carrier at 15-20 dB: both listed, the carrier holding it, the jammer measured over it
    rng = np.random.default_rng(11)
    Fs, wide = 50e6, 4e6
    wide_db = 20 if jam_db == 22 else 15
    x = psd_noise(rng, 1 << 20, Fs, lambda f: 1 + 10**(wide_db/10)*rc(f - 2e6, wide) + 10**(jam_db/10)*rc(f - 2e6 - off, jam_baud))
    sv = find(x, Fs)
    assert len(sv) == 2
    carrier = max(sv, key=lambda s: s.width)
    jammer = min(sv, key=lambda s: s.width)
    assert carrier.holds == 1 and jammer.over is carrier
    assert carrier.mid == pytest.approx(2e6, abs=0.05*wide) and carrier.width == pytest.approx(1.17*wide, rel=0.1)
    assert jammer.mid == pytest.approx(2e6 + off, abs=0.05*jam_baud)
    over = 10*np.log10((10**(wide_db/10) + 10**(jam_db/10) + 1)/(10**(wide_db/10) + 1)) # its flat part over the carrier's
    assert jammer.snr == pytest.approx(10*np.log10(10**(over/10) - 1) - 0.6, abs=1.5)

def test_a_tone_is_a_narrow_box():
    rng = np.random.default_rng(7)
    Fs, N = 1e6, 1 << 18
    x = (rng.standard_normal(N) + 1j*rng.standard_normal(N))/np.sqrt(2) + 0.3*np.exp(2j*np.pi*123.4e3/Fs*np.arange(N))
    sv = find(x.astype(np.complex64), Fs)
    assert len(sv) == 1 and sv[0].mid == pytest.approx(123.4e3, abs=sv.bin_hz) and sv[0].width < 4*sv.bin_hz

@pytest.mark.parametrize("centred", [True, False])
def test_a_carriers_own_lines_are_not_signals(centred):
    """Lines at one spacing centred on a carrier (periodic sync bytes put them there: DVB-S) are the carrier's; the same
    comb off its centre is something else on it"""
    rng = np.random.default_rng(12)
    Fs, N, baud, g = 8e6, 1 << 20, 1e6, 15.625e3
    x = psd_noise(rng, N, Fs, lambda f: 1 + 20*rc(f - 1.7e6, baud))
    at = 1.7e6 + (0 if centred else 0.25e6)
    n = np.arange(N)
    for k in range(-2, 3):
        x = x + (0.6 if k == 0 else 0.4)*np.exp(2j*np.pi*(at + k*g)/Fs*n).astype(np.complex64)
    sv = find(x.astype(np.complex64), Fs)
    carrier = max(sv, key=lambda s: s.width)
    assert carrier.mid == pytest.approx(1.7e6, abs=0.05*baud) and carrier.width == pytest.approx(1.17*baud, rel=0.1)
    if centred:
        assert len(sv) == 1 and not carrier.holds
    else:
        assert len(sv) > 1 and all(s.over is carrier for s in sv if s is not carrier)

def test_bursts_get_their_times():
    rng = np.random.default_rng(8)
    Fs, N, baud = 10e6, 1 << 20, 200e3
    y = psd_noise(rng, N, Fs, lambda f: 30*rc(f - 2e6, baud))
    t = np.arange(N)/Fs
    on = [(0.010, 0.025), (0.050, 0.060), (0.080, 0.095)]
    gate = np.zeros(N)
    for a, b in on:
        gate[(t >= a) & (t < b)] = 1
    x = psd_noise(rng, N, Fs, lambda f: np.ones_like(f)) + y*gate.astype(np.complex64)
    sv = find(x, Fs)
    assert len(sv) == 1
    s = sv[0]
    assert s.duty == pytest.approx(0.4, abs=0.1) and len(s.bursts) == 3
    frame = 2*sv.nfft/Fs # a burst's edges to about a frame
    for (a, b), (fa, fb) in zip(on, s.bursts):
        assert fa == pytest.approx(a, abs=frame) and fb == pytest.approx(b, abs=frame)

def test_a_carrier_cut_by_the_band_edge_is_clipped():
    rng = np.random.default_rng(9)
    Fs = 20e6
    x = psd_noise(rng, 1 << 19, Fs, lambda f: 1 + 50*rc(f - 9.5e6, 2e6))
    sv = find(x, Fs)
    assert len(sv) == 1 and sv[0].clipped and sv[0].hi > 9e6

def test_snr_of_real_qpsk_agrees_with_its_esn0(rng):
    # Real symbols through an RRC (not shaped noise): box SNR + 10 log10(width / baud) is its Es/N0
    pts = [1+1j, 1-1j, -1+1j, -1-1j]
    for esn0 in (3.0, 10.0, 20.0):
        x, _ = shaped(rng, pts, 1 << 15, 8, beta=0.35, esn0_db=esn0)
        s = find(x, 1.0).pick(strongest=True)
        assert s.snr + 10*np.log10(s.width/(1/8)) == pytest.approx(esn0, abs=1.0)

@pytest.mark.parametrize("level", ["fast", "balanced", "max"])
def test_every_effort_finds_strong_carriers(level):
    rng = np.random.default_rng(10)
    Fs, baud = 50e6, 1e6
    cs = [-15e6, -3e6, 9e6]
    x = psd_noise(rng, 1 << 18, Fs, lambda f: 1 + sum(30*rc(f - c, baud) for c in cs))
    with rp.effort(level):
        sv = find(x, Fs)
    assert [round(s.mid/baud) for s in sv] == [round(c/baud) for c in cs]

def _scene(rng, N, Fs):
    """Three carriers (one bursting on and off) for the streaming tests"""
    x = psd_noise(rng, N, Fs, lambda f: 1 + 30*rc(f + 3e6, 1e6) + 10*rc(f - 2e6, 300e3))
    y = psd_noise(rng, N, Fs, lambda f: 50*rc(f - 6e6, 500e3))
    on = (np.arange(N)//(N//8)) % 2 == 1
    return (x + y*on).astype(np.complex64)

def boxes(sv):
    return [(round(s.lo), round(s.hi), round(s.snr, 6), s.bursts) for s in sv]

def test_finder_does_not_depend_on_how_the_stream_is_cut():
    Fs = 20e6
    x = _scene(np.random.default_rng(12), 1 << 18, Fs)
    ends = []
    for cuts in ([65536]*4, [1000, 50000, 3, 80000, 131141], [4096]*64):
        f = Finder(Fs, nfft=1024, frames=64, refit=8, track=False)
        pos, sv = 0, None
        for n in cuts:
            sv = f.step(x[pos:pos + n]) or sv
            pos += n
        assert pos == len(x)
        ends.append(boxes(sv))
    assert ends[0] == ends[1] == ends[2]

def test_finder_finds_what_find_does_over_the_frames_it_holds():
    Fs = 20e6
    x = _scene(np.random.default_rng(13), 1 << 17, Fs)
    with rp.effort("balanced"):
        f = Finder(Fs, nfft=1024, frames=255, refit=1000)
        sv = f.step(x)
        ref = find(x[:(255 - 1)*512 + 1024], Fs, nfft=1024)
    # (its groups of frames are each detected once, as they complete, where find tests them together: the same
    # boxes, their edges to a couple of bins)
    assert len(sv) == len(ref) == 3
    for s, r in zip(sv, ref):
        assert abs(s.lo - r.lo) <= 2*sv.bin_hz and abs(s.hi - r.hi) <= 2*sv.bin_hz
        assert s.snr == pytest.approx(r.snr, abs=0.1) and (s.bursts is None) == (r.bursts is None)
    assert [round(s.mid/1e5) for s in sv] == [-30, 20, 60]

def test_finder_times_run_from_the_streams_start_and_ids_stay():
    Fs = 20e6
    x = _scene(np.random.default_rng(14), 1 << 19, Fs)
    f = Finder(Fs, nfft=1024, frames=64)
    ids = {}
    for k in range(0, len(x), 32768):
        sv = f.step(x[k:k + 32768])
        if sv is None:
            continue
        held = (64 - 1)*f.hop + 1024
        for s in sv:
            assert (k + 32768 - held)/Fs - 1e-9 <= s.t0 <= s.t1 <= (k + 32768)/Fs + 1e-9
            assert ids.setdefault(round(s.mid/1e5), s.id) == s.id # each keeps its id (the bursting one too)
    assert sorted(ids) == [-30, 20, 60]

def test_tracked_boxes_agree_with_measuring_afresh():
    Fs = 20e6
    x = _scene(np.random.default_rng(15), 1 << 19, Fs)
    tracked, fresh = Finder(Fs, nfft=1024, frames=64), Finder(Fs, nfft=1024, frames=64, track=False)
    for k in range(0, len(x), 16384):
        a, b = tracked.step(x[k:k + 16384]), fresh.step(x[k:k + 16384])
        if a is None:
            continue
        assert len(a) == len(b)
        for s, r in zip(a, b):
            assert s.mid == pytest.approx(r.mid, abs=0.02*r.width) and s.width == pytest.approx(r.width, rel=0.05)
            assert s.snr == pytest.approx(r.snr, abs=0.5)
            assert s.t0 == r.t0 and s.t1 == r.t1

def test_staircase_fit_kernel_matches_the_python_loop():
    from rpps import _accel
    if _accel.native is None or not hasattr(_accel.native, "pelt"):
        pytest.skip("no C++ kernels")
    rng = np.random.default_rng(16)
    for n in (3, 40, 700, 2500):
        y = np.repeat(rng.normal(0, 4, n//25 + 1), 25)[:n] + rng.normal(size=n)
        w = 1/rng.uniform(0.5, 2, n)
        for lmin in (1, 2, 3):
            assert _pelt(y, w, 3*np.log(n + 1), lmin) == _pelt_py(y, w, 3*np.log(n + 1), lmin)

def test_staircase_fit_finds_the_steps():
    rng = np.random.default_rng(17)
    y = np.concatenate([np.full(100, 0.0), np.full(60, 6.0), np.full(80, 2.0)]) + 0.3*rng.normal(size=240)
    assert _pelt_py(y, np.full(240, 1/0.09), 3*np.log(240)) == [0, 100, 160, 240]
