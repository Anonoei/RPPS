"""projects/spectrum/sweep_monitor.py: the pieces that keep level steps and artifacts out of the stitched spectrum, on synthetic data, and a whole sweep against a simulated radio"""
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("pysdrlib")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401
import sweep_monitor as sm  # noqa: E402

# ---------------------------------------------------------------------------------------------- pieces

def tile_db(rng, nf=400, spur=300, tone=None):
    """A tile's dB spectrum: noise (mean 0 dB, averaged periodogram bins), roll-off over the last 10 % each side, a spur at a fixed offset, an optional narrow tone"""
    db = 10*np.log10(rng.gamma(40, 1/40, nf))
    roll = np.zeros(nf); roll[:40] = -np.linspace(8, 0, 40); roll[-40:] = -np.linspace(0, 8, 40)
    db = db + roll; db[spur] += 15
    if tone is not None:
        db[tone] += 20
    return db

def test_tile_shape_removes_common_features_keeps_signals(rng):
    ts = sm.TileShape(400)
    for _ in range(30):
        ts.update(tile_db(rng, tone=int(rng.integers(60, 340))))
    t = ts.template()
    assert t is not None and t[300] > 10 and t[5] < -4          # the spur and the roll-off are the shape
    x = tile_db(rng, tone=150) - t
    assert abs(x[300] - x[290:296].mean()) < 3                  # spur gone
    assert abs(x[:30].mean() - x[180:220].mean()) < 1.5         # edges level with the centre
    assert x[150] - np.median(x) > 15                           # the tone survives
    assert abs(np.median(x)) < 1.0                              # absolute level kept

def test_tile_shape_needs_history(rng):
    ts = sm.TileShape(400)
    ts.update(tile_db(rng))
    assert ts.template() is None

def test_gain_hold_stays_put_and_steps():
    ladder = [(1, 40, 32), (1, 40, 16), (1, 32, 16), (0, 32, 8), (0, 0, 4)]
    limit = [3]                                                  # highest rung index the "tile" accepts
    hold = sm.GainHold(ladder, up_every=3)
    ok = lambda g: ladder.index(g) >= limit[0]
    assert hold.pick(ok) == ladder[3]                            # walked down from the top
    seen = [hold.pick(ok), hold.pick(ok)]
    assert seen == [ladder[3]]*2                                 # held, no search
    limit[0] = 1
    assert hold.pick(ok) == ladder[2]                            # the periodic try one step up (not further)
    limit[0] = 4
    assert hold.pick(ok) == ladder[4]                            # a loud tile: down at once
    limit[0] = 0
    assert hold.pick(ok) == ladder[4]                            # no probing before the interval
    assert hold.pick(ok) == ladder[3]                            # then one step up

def test_gain_hold_gives_the_lowest_when_nothing_is_ok():
    ladder = [(1, 40, 32), (0, 0, 4)]
    assert sm.GainHold(ladder).pick(lambda g: False) == ladder[-1]

def test_persistent_drops_what_only_one_sweep_saw():
    db = np.full(100, -90.0); other = db.copy()
    db[10] = -60; db[40] = -60; other[40] = -60; other[70] = -60; db[80] = -85; other[80] = -83
    r = sm.persistent(sm.q(db), sm.q(other))
    assert sm.unq(r)[10] < -85 and sm.unq(r)[70] == pytest.approx(sm.unq(sm.q(db))[70], abs=0.1)    # seen once: lower value
    assert sm.unq(r)[40] > -62                                   # seen in both: kept
    assert sm.unq(r)[80] == pytest.approx(-85, abs=0.5)          # small differences are noise, not artifacts
    assert (sm.persistent(sm.q(db), None) == sm.q(db)).all()
    hole = sm.q(db).copy(); hole[5] = 0
    assert sm.persistent(hole, sm.q(db))[5] == sm.q(db)[5]       # a bin this sweep did not measure comes from the neighbour

def test_pmean_is_a_power_mean():
    a, b = sm.q(np.array([-100.0])), sm.q(np.array([-90.0]))
    assert sm.unq(sm.pmean([a, b]))[0] == pytest.approx(10*np.log10((1e-10 + 1e-9)/2), abs=0.5)   # -92.6 dB, not the -95 of a dB mean
    z = np.zeros(1, np.uint8)
    assert sm.pmean([z, z])[0] == 0 and sm.unq(sm.pmean([a, z]))[0] == pytest.approx(-100, abs=0.5)

def test_match_offset():
    idx_a = np.arange(0, 100); idx_b = np.arange(80, 180)
    a = np.zeros(100); b = np.full(100, -3.0)
    assert sm.match_offset(idx_a, a, idx_b, b) == pytest.approx(3.0)
    assert sm.match_offset(idx_a, a, idx_b, b - 10) is None      # too large to be a gain error
    assert sm.match_offset(idx_a, a, np.arange(95, 195), b) is None  # too little overlap

def test_rung_cal_learns_errors_and_leaves_slopes_alone():
    A, B, C = (1, 40, 32), (1, 32, 16), (0, 8, 4)
    cal = sm.RungCal()
    for _ in range(20):
        cal.seen(A); cal.observe(A, B, 3.0); cal.observe(B, A, -3.0); cal.observe(B, C, -2.0); cal.observe(A, A, 5.0)   # B reads 3 dB below A, C 2 dB above B (net +1 above A... sign: prev - cur)
    off = cal.solve()
    assert off[A] == pytest.approx(0, abs=0.3) and off[B] == pytest.approx(3.0, abs=0.4) and off[C] == pytest.approx(1.0, abs=0.5)
    assert cal.offset((9, 9, 9)) == 0.0                           # a setting no join has seen keeps the nominal level
    assert sm.RungCal().solve() == {}

def test_overlap():
    assert sm.overlap(20, 16) == pytest.approx(1.0) and sm.overlap(20, 19) == 0.0 and sm.overlap(20, 4) == 1.5

# ---------------------------------------------------------------------------------------------- whole sweeps against a simulated radio

class FakeSdr:
    """Complex noise (sigma 1 per component at the input), a tone at an absolute frequency, and a spur at a fixed offset from the tuned centre, all scaled by the gain; clips at +-1"""
    def __init__(self, tone_mhz, tone_amp=3.0, spur_off=3.1e6, spur_amp=1.0, seed=3, blocker_mhz=183.0, blocker_amp=300.0, imd_mhz=160.7, imd_amp=2.0):
        self.rng = np.random.default_rng(seed); self.fs = 20e6; self.cf = 100e6; self.g = 0.0; self.t0 = 0
        self.tone, self.tone_amp, self.spur_off, self.spur_amp = tone_mhz*1e6, tone_amp, spur_off, spur_amp
        self.blocker, self.blocker_amp, self.imd, self.imd_amp = blocker_mhz*1e6, blocker_amp, imd_mhz*1e6, imd_amp
    def set_rx_freq(self, f, flush=False): self.cf = f
    def set_rx_sample_rate(self, fs): self.fs = fs
    RANGE_DB = 83.0
    def set_rx_gain(self, v, channel=0): self.g = v
    def get_rx_gain_db(self, channel=0): return self.g*self.RANGE_DB
    def get_rx_sample_rate(self): return self.fs
    rx_channels = (0,)
    def get_samples(self):
        n = 2**16; t = (self.t0 + np.arange(n))/self.fs; self.t0 += n
        x = self.rng.normal(size=n) + 1j*self.rng.normal(size=n)
        x = x + self.tone_amp*np.exp(2j*np.pi*(self.tone - self.cf)*t)*(abs(self.tone - self.cf) < self.fs*0.45)
        x = x + self.spur_amp*np.exp(2j*np.pi*self.spur_off*t)
        db = self.g*self.RANGE_DB
        for f, a in ((self.blocker, self.blocker_amp), (self.imd, self.imd_amp*10**((db - 72)/10))):   # a strong real signal; a front-end product whose input-referred level rises 2 dB per dB of gain
            x = x + a*np.exp(2j*np.pi*(f - self.cf)*t)*(abs(f - self.cf) < self.fs*0.45)
        err = 4.0 if db >= 55 else 0.0                    # the radio's true gain is not the nominal one
        x = x*10**((db + err)/20)*1e-5
        return np.clip(x.real, -1, 1) + 1j*np.clip(x.imag, -1, 1)

def run(sweeps, **params):
    sw = sm.Sweeper(100, 230, **dict(dict(dwell=dict(unit="samples", value=2**16), settle=0.0), **params))
    sw.sdr = FakeSdr(147.3); sw.cur_fs = 20.0
    for _ in range(sweeps):
        sw.one_sweep()
    return sw

def lines(sw, f, tol=0.25):
    return [e for e in sw.emitters if abs(e["f"] - f) < tol]

@pytest.fixture(scope="module")
def raw_sweeps():
    return run(6, alt_fs=False, alt_gain=False, alt=False, flatten=False, match=False)

@pytest.fixture(scope="module")
def good_sweeps():
    return run(6)

def spur_freqs(sw, offset=3.1):
    out = []
    for k in (0, 1):
        fs, step, shift = sw.geometry(k)
        a = sw.lo - shift
        while a < sw.hi:
            out.append(a + step/2 + offset); a += step
    return [f for f in out if sw.lo < f < sw.hi]

def test_without_the_corrections_the_tile_spur_is_in_the_spectrum(raw_sweeps):
    assert lines(raw_sweeps, 147.3)
    assert sum(bool(lines(raw_sweeps, f)) for f in spur_freqs(raw_sweeps)) >= 4

def test_front_end_product_is_confirmed_only_at_one_gain(raw_sweeps, good_sweeps):
    assert lines(raw_sweeps, 160.7)                              # no gain test: it looks like a signal
    assert not lines(good_sweeps, 160.7)                         # the sweeps at another gain do not have it at the same level
    assert lines(good_sweeps, 183.0) and lines(good_sweeps, 147.3)   # real signals stay

def test_corrections_remove_the_spur_and_keep_the_signal(good_sweeps):
    sw = good_sweeps
    assert sw.sweeps == 6 and sw.flat
    assert lines(sw, 147.3)
    left = [f for f in spur_freqs(sw) if lines(sw, f) and abs(f - 147.3) > 0.5]
    assert not left, left

def test_every_cell_is_measured_except_the_tile_centres(good_sweeps):
    """Tile centres on the 50 kHz grid put FFT bins exactly on cell centres; with cells centred half a bin off they were ties and every other cell stayed empty (found on the real radio)"""
    sw = good_sweeps
    for _, row in list(sw.rows)[:4]:
        z = row == 0
        assert z.mean() < 0.035, z.mean()
        assert not (z[:-2] & z[2:] & ~z[1:-1]).any() or (z[:-2] & z[2:] & ~z[1:-1]).sum() < 3     # no alternate-cell pattern
    assert all((r > 0).mean() > 0.995 for _, r in list(sw.prows)[:4])       # the neighbour sweep fills the masked centres

def test_floor_is_flat_across_tiles(good_sweeps):
    sw = good_sweeps; db = sm.unq(sw.avg); have = sw.avg > 0
    f = sw.f0 + np.arange(sw.n)*sm.BIN/1e6
    quiet = have & (np.abs(f - 147.3) > 0.5) & (np.abs(f - 183.0) > 3) & (np.abs(f - 160.7) > 0.5)
    tiles = [np.median(db[quiet & (f >= a) & (f < a + 8)]) for a in np.arange(104, 224, 8)]
    assert np.ptp(tiles) < 1.5, tiles                                  # input-referred floor the same everywhere (eight steps of 8 MHz)

def test_alternate_sweeps_use_another_rate_and_grid(good_sweeps):
    (fa, sa, ha), (fb, sb, hb) = good_sweeps.geometry(0), good_sweeps.geometry(1)
    assert good_sweeps.params['alt_gain'] and fa == 20 and fb == 16 and sa == 16 and sb == pytest.approx(12.8) and ha == 0 and hb == pytest.approx(6.4)

def test_peak_hold_decays_and_resets(good_sweeps):
    sw = good_sweeps
    assert sw.peak.max() > 0
    sw.reset_peak()
    assert sw.peak.max() == 0 and not np.isfinite(sw.peak_f).any()

def test_configure_validates():
    sw = sm.Sweeper(100, 230)
    sw.configure(dict(headroom=0.6, peak_decay=0.0, flatten=False, match=False, alt_fs=False))
    assert sw.params["headroom"] == 0.6 and not sw.params["flatten"]
    for bad in (dict(headroom=2.0), dict(peak_decay=-1)):
        with pytest.raises(AssertionError):
            sw.configure(bad)

# ---------------------------------------------------------------------------------------------- the web side

def test_http_summary_and_controls(good_sweeps):
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    sw = good_sweeps
    srv = ThreadingHTTPServer(("127.0.0.1", 0), sm.make_handler(sw, b"<html></html>")); srv.allow_remote = False; srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert len(urllib.request.urlopen(base + "/api/summary").read()) == 3*sw.n      # confirmed average, peak hold, unconfirmed average
        k = sw.sweeps - 1
        assert len(urllib.request.urlopen(f"{base}/api/row/{k}").read()) == sw.n
        st = json.load(urllib.request.urlopen(base + "/api/status"))
        assert st["flat"] and st["params"]["alt_fs"] and len(st["geometry"]) == 2
        req = urllib.request.Request(base + "/api/control", json.dumps(dict(reset_peak=True, headroom=0.4, flatten=False)).encode(), {"Content-Type": "application/json"})
        st = json.load(urllib.request.urlopen(req))
        assert st["params"]["headroom"] == 0.4 and not st["params"]["flatten"] and sw.peak.max() == 0
    finally:
        srv.shutdown()
