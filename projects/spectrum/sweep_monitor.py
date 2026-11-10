"""Live wideband sweep in the browser: where the radio is looking, the stitched spectrum, and a spectrogram of the sweeps

    python projects/spectrum/sweep_monitor.py --lo 30 --hi 2000                          # then open http://127.0.0.1:8026/
    python projects/spectrum/sweep_monitor.py --remote tcp://HOST:5555 --lo 400 --hi 900

The HackRF steps through tiles (16 MHz wide at 20 MS/s by default, as wide_survey.py; the width, sample rate, dwell and settle time are settings): per tile the gain is stepped down its ladder until nothing clips, the spectrum is averaged into 50 kHz bins, the DC spike of
the tile masked, and the level is corrected by the gain used so tiles measured at different gains stitch together (units: dB relative to the antenna input at 0 dB gain, not calibrated). A sweep is one
pass over the range; consecutive sweeps alternate the tile grid by 8 MHz so tile-edge artifacts do not stay on the same frequencies. The page shows the sweep in progress with the tile the radio is on, the
last complete sweep, the average of the last 8 and the peak hold, a spectrogram of the completed sweeps (newest on top), the allocation bands, and a list of the strongest signals in the average (more than
8 dB over the local floor). Drag on the spectrum to zoom, double-click to reset. The range can be changed or the sweep paused from the page. The server listens on 127.0.0.1 only (--host to change it).

Keeping artifacts and level steps out of the display (each can be switched off, see --help):
* gain is held from tile to tile: a tile starts at the previous tile's setting and goes down only when its peak amplitude passes --headroom of full scale (not only when it clips: intermodulation shows well before
  clipping); every few tiles one step up is tried. Fewer changes means fewer steps in the floor.
* each tile is flattened: the shape every tile has in common (the analog filter's roll-off at the edges, spurs at a fixed offset from the tile centre) is the median over recent tiles of the tile minus its own local floor,
  and is subtracted. A real signal sits at a different offset in each tile, so it does not enter the median.
* the level error of each gain setting is learned from the joins between neighbouring tiles that were measured at different settings (the strip both measured beyond the kept width) and removed; a real slope of
  the floor along the band is left alone.
* alternate sweeps use another sample rate (80 % of the first), another tile grid and half a rung lower gain: combs and images move with the first two, overload and intermodulation products of the front end (the same
  frequencies whatever the tuning) fall with the gain, real signals stay put. The average, peak hold, spectrogram and signal list use a sweep
  only where it agrees with the neighbouring sweep (within 4 dB, else the lower value); the unconfirmed average stays on screen as a dim trace. The peak hold decays (--peak-decay) and can be reset.
Still not removed: anything that is the same at every sample rate and tile position, and a signal that only the first of two consecutive sweeps saw (bursts). wide_survey.py's two-pass confirmation remains the
tool for deciding what is real. Meant to be upstreamed into RPPS.
"""
import argparse
import json
import os
import sys
import threading
import time
import warnings
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from scipy.ndimage import percentile_filter

HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import radio_io  # noqa: E402
import wide_survey  # noqa: E402

BIN = 50e3; D0, D1 = -130.0, -10.0          # 50 kHz bins whatever the sample rate; the dB range the quantised bytes cover
DEFAULTS = dict(step=16.0, fs=0.0, dwell=dict(unit="samples", value=2**19), settle=0.04, gain=None, alt=True, alt_fs=True, alt_gain=True, flatten=True, match=True, headroom=0.5, peak_decay=0.25)   # fs 0 = automatic: the tile width / 0.8, at most 20 MS/s
KEEP = 240                                  # sweeps kept for the spectrogram


def q(db):
    """dB -> byte (0 = no data)"""
    return np.clip(np.rint((db - D0)/(D1 - D0)*254) + 1, 1, 255).astype(np.uint8)


def unq(b):
    return D0 + (b.astype(np.float32) - 1)/254*(D1 - D0)


def pmean(rows):
    """Mean power (not mean dB) of quantised rows, where they have data; 0 = no data anywhere. Averaging dB would bias noise-like bins low."""
    data = np.array([unq(r) for r in rows]); have = np.array([r > 0 for r in rows]); cnt = have.sum(0)
    pw = np.where(have, 10**(data/10), 0.0).sum(0)
    out = np.zeros(len(rows[0]), np.uint8); m = cnt > 0
    out[m] = q(10*np.log10(pw[m]/cnt[m]))
    return out


def persistent(row, other, tol=4.0):
    """`row` (quantised) kept only where `other` (the neighbouring sweep, taken with another sample rate or tile grid) agrees: where they differ by more than `tol` dB the lower value is used, since an artifact
    moves with the sweep geometry and a signal does not. Where `other` has no data the row is unchanged; where only `other` has data (the masked centre of a tile) it is used."""
    if other is None:
        return row.copy()
    both = (row > 0) & (other > 0)
    low = np.minimum(row, other)    # quantisation is monotonic: the lower byte is the lower level
    differ = both & (np.abs(row.astype(np.int16) - other.astype(np.int16))*(D1 - D0)/254 > tol)
    return np.where(differ, low, np.where(row == 0, other, row)).astype(np.uint8)    # a bin this sweep did not measure (the tile centre) is the neighbour's


class TileShape:
    """What every tile has in common. Per tile, the residual is the spectrum minus the level of its own middle (30th percentile of the inner 60 %: the noise floor there, not a signal); the shape is the median of
    the last `keep` residuals at each offset from the tile centre, with its own median removed so levels stay absolute. A real signal is at a different offset in each tile and falls out of the median; the
    filter's roll-off towards the tile edges and spurs at a fixed offset from the centre (the HackRF's, whatever the tuned frequency) stay. (A floor that followed the roll-off locally would remove it from the
    residual: it has to be referred to one level per tile.)"""
    def __init__(self, nf, keep=48, minimum=8):
        self.nf, self.minimum = nf, minimum; self.res = deque(maxlen=keep)

    def update(self, db):
        n = len(db); self.res.append(db - np.percentile(db[n//5:n - n//5], 30))

    def template(self):
        if len(self.res) < self.minimum:
            return None
        t = np.median(np.stack(self.res), axis=0)
        return t - np.median(t)


class GainHold:
    """Gain ladder position kept from tile to tile. `ok(gain)` says whether the tile is acceptable at that gain (no clipping, enough headroom). Start where the last tile ended, go down the ladder (lower gain)
    until ok, and every `up_every` tiles try one step up first."""
    def __init__(self, ladder, up_every=12):
        self.ladder, self.up_every, self.i, self.since = list(ladder), up_every, 0, 0

    def reset(self):
        self.i, self.since = 0, 0

    def pick(self, ok):
        i = self.i
        if self.since >= self.up_every and i > 0:
            self.since = 0
            if ok(self.ladder[i - 1]):
                i -= 1
        while i < len(self.ladder) - 1 and not ok(self.ladder[i]):
            i += 1
        self.i = i; self.since += 1
        return self.ladder[i]


def match_offset(prev_idx, prev_db, idx, db, limit=6.0, minimum=10):
    """dB that the tile (idx, db) lies below the previous tile over the bins both measured (median difference); None when there are too few bins or the difference is over `limit` (a signal, not a gain error)"""
    common, a, b = np.intersect1d(prev_idx, idx, return_indices=True)
    if len(common) < minimum:
        return None
    off = float(np.median(prev_db[a] - db[b]))
    return off if abs(off) <= limit else None


LADDER_ALT = [round(g - 0.04, 3) for g in radio_io.LADDER if g >= 0.04]   # half a rung lower than the main ladder, for the sweeps that test for overload products: their products move 3 dB per dB of gain, signals 1


class RungCal:
    """Level error of each gain setting, from the joins between neighbouring tiles that were measured at different settings. The nominal correction (the gain in dB) is not what the radio does, so the
    input-referred floor steps where the setting changes. At a join, `prev - this` over the strip both measured is the difference of the two settings' errors; the errors are the least-squares solution
    over all joins seen, each setting pulled weakly to 0 (settings no join connects keep the nominal level) and the most used one held at 0. Joins between tiles at the SAME setting give nothing: a real slope
    of the floor along the band is not a gain error and is left alone (a chain correction along the sweep removed it, then ran out of range and let the steps back)."""
    def __init__(self, keep=400, limit=15.0):
        self.obs = deque(maxlen=keep); self.off = {}; self.use = {}; self.limit = limit

    def seen(self, g):
        self.use[g] = self.use.get(g, 0) + 1

    def observe(self, g_prev, g_cur, d):
        if g_prev != g_cur and d is not None:
            self.obs.append((g_prev, g_cur, d))

    def solve(self):
        gs = sorted({g for o in self.obs for g in o[:2]})
        if len(gs) < 2:
            self.off = {}; return self.off
        ix = {g: i for i, g in enumerate(gs)}; rows, y = [], []
        for gp, gc, d in self.obs:
            r = np.zeros(len(gs)); r[ix[gc]] += 1; r[ix[gp]] -= 1; rows.append(r); y.append(d)    # o[cur] - o[prev] = prev - cur
        for g in gs:
            r = np.zeros(len(gs)); r[ix[g]] = 0.05; rows.append(r); y.append(0.0)
        ref = max(gs, key=lambda g: self.use.get(g, 0)); r = np.zeros(len(gs)); r[ix[ref]] = 10.0; rows.append(r); y.append(0.0)
        x = np.linalg.lstsq(np.array(rows), np.array(y), rcond=None)[0]
        self.off = {g: float(np.clip(x[ix[g]], -self.limit, self.limit)) for g in gs}
        return self.off

    def offset(self, g):
        return self.off.get(g, 0.0)


def overlap(fs, step):
    """MHz beyond each edge of the kept tile that are still usable for matching neighbours (0.45 fs is where the HackRF's band ends); at most 1.5"""
    return float(max(0.0, min(1.5, 0.45*fs - step/2)))


class Sweeper:
    def __init__(self, lo, hi, remote=None, gain=None, debug=False, driver=None, **params):
        self.params = dict(DEFAULTS, gain=gain, **params); self.cur_fs = None; self.tile_s = 0.0
        self.debug = debug; self.shapes = {}; self.holders = (GainHold(radio_io.LADDER), GainHold(LADDER_ALT)); self.cal = RungCal(); self.flat = False
        self.remote = remote; self.driver = driver; self.lock = threading.RLock(); self.stop = threading.Event(); self.pause = False
        self.sdr = None; self.error = None; self.set_range(lo, hi)
        self.rows = deque(maxlen=KEEP); self.sweeps = 0; self.t_sweep = 0.0

    def set_range(self, lo, hi):
        with self.lock:
            self.lo, self.hi = float(lo), float(hi)
            self.n = int(round((self.hi - self.lo)*1e6/BIN))
            self.f0 = self.lo     # cell i is centred on lo + i*BIN: the FFT bins of a tile on the 50 kHz grid fall exactly on these centres (a centre on a cell EDGE made every other bin a tie that rounded half-to-even: alternate cells empty)
            self.live = np.zeros(self.n, np.uint8); self.avg = np.zeros(self.n, np.uint8); self.peak = np.zeros(self.n, np.uint8); self.raw = np.zeros(self.n, np.uint8)
            self.peak_f = np.full(self.n, -np.inf, np.float32); self.prows = deque(maxlen=KEEP)
            self.rows = deque(maxlen=KEEP); self.sweeps = 0; self.tile = dict(cf=None, gain=None, i=0, n=0); self.emitters = []; self.changed = True
            self.restart = True
            for h in self.holders:
                h.reset()

    def hold(self, row, decay):
        """Peak hold (decaying) with a confirmed sweep"""
        pdb = np.where(row > 0, unq(row), -np.inf)
        self.peak_f = np.maximum(self.peak_f - decay, pdb).astype(np.float32)
        self.peak = np.where(np.isfinite(self.peak_f), q(np.where(np.isfinite(self.peak_f), self.peak_f, D0)), 0).astype(np.uint8)

    def reset_peak(self):
        with self.lock:
            self.peak_f[:] = -np.inf; self.peak[:] = 0; self.changed = True

    def rate_limits(self):
        """(lowest, highest) sample rate in MHz: pysdrlib's caps of the open radio, else of the driver; 0.05 (one bin) and 20 where the caps say nothing"""
        lo, hi = radio_io.rate_limits(self.sdr, self.driver, self.remote)
        return (lo if lo is not None else BIN/1e6), (hi if hi is not None else 20.0)

    def geometry(self, k):
        """(sample rate [MHz], tile width [MHz], grid shift [MHz]) of sweep number k. Odd sweeps change the grid (half a tile) and the sample rate (80 %), so what moves with them is not a signal; with `alt_gain` they
        also run at a slightly lower gain (see `self.holders`): overload and intermodulation products of the front end stay at the same frequency whatever the tuning or the sample rate, and only the gain moves them."""
        p = self.params; fs = self.fs_mhz(); step = p["step"]
        if p["alt_fs"] and k % 2:
            fs = max(self.rate_limits()[0], round(fs*0.8/0.05)*0.05); step = min(step, round(0.8*fs, 3))
        return fs, step, (step/2 if (p["alt"] and k % 2) else 0.0)

    def fs_mhz(self):
        """Sample rate in MHz: the chosen one, or the tile width / 0.8 (the HackRF's usable part of its band) up to 20 MS/s; never below the tile width, a multiple of the 50 kHz bin"""
        p = self.params
        fs = p["fs"] or min(20.0, max(2.0, np.ceil(p["step"]/0.8)))
        return float(min(20.0, max(2.0, np.ceil(max(fs, p["step"])/0.05 - 1e-9)*0.05)))

    def dwell_samples(self, fs_mhz):
        d = self.params["dwell"]
        n = d["value"]*fs_mhz*1e6 if d["unit"] == "seconds" else d["value"]
        return int(min(max(n, 4096), 5*20e6))

    def configure(self, d):
        """Change the sweep settings (validated); the sweep in progress restarts, the history is kept"""
        p = dict(self.params)
        if "step" in d:
            p["step"] = float(d["step"]); assert 1.0 <= p["step"] <= self.rate_limits()[1]
        if "fs" in d:
            p["fs"] = float(d["fs"] or 0.0); assert p["fs"] == 0 or self.rate_limits()[0] <= p["fs"] <= self.rate_limits()[1]
        if "dwell" in d:
            u, v = d["dwell"]["unit"], float(d["dwell"]["value"]); assert u in ("samples", "seconds") and (4096 <= v <= 1e8 if u == "samples" else 0.001 <= v <= 5.0)
            p["dwell"] = dict(unit=u, value=int(v) if u == "samples" else v)
        if "settle" in d:
            p["settle"] = float(d["settle"]); assert 0.0 <= p["settle"] <= 2.0
        if "gain" in d:
            g = d["gain"]; assert g is None or 0.0 <= float(g) <= 1.0; p["gain"] = None if g is None else float(g)
        for key in ("alt", "alt_fs", "alt_gain", "flatten", "match"):
            if key in d:
                p[key] = bool(d[key])
        if "headroom" in d:
            p["headroom"] = float(d["headroom"]); assert 0.1 <= p["headroom"] <= 1.0
        if "peak_decay" in d:
            p["peak_decay"] = float(d["peak_decay"]); assert 0.0 <= p["peak_decay"] <= 20.0
        with self.lock:
            self.params = p; self.restart = True

    # -- radio side
    def run(self):
        while not self.stop.is_set():
            try:
                if self.sdr is None:
                    self.cur_fs = self.rate_limits()[1]; self.sdr = radio_io.open_radio(cf=100e6, fs=self.cur_fs*1e6, remote=self.remote, driver=self.driver); self.sdr.start_rx(); self.error = None
                self.one_sweep()
            except Exception as e:  # a stalled stream or a lost connection: reopen
                self.error = f"{type(e).__name__}: {e}"
                try:
                    if self.sdr is not None:
                        self.sdr.stop_rx(); self.sdr.close()
                except Exception:
                    pass
                self.sdr = None; time.sleep(3)

    def capture_psd(self, nsamp, nf):
        """Power spectrum (nf bins, centred) averaged over nsamp samples; the samples are processed as they arrive, so a long dwell needs no memory"""
        self.sdr.get_samples()   # what queued up during the retune is from the old frequency
        win = np.hanning(nf); acc = np.zeros(nf); cnt = 0; got = 0; left = np.empty(0, np.complex64); t_last = time.time()
        while got < nsamp:
            b = self.sdr.get_samples()
            if not len(b):
                if time.time() - t_last > 3:
                    raise RuntimeError("no samples from the radio for 3 s")
                time.sleep(0.001); continue
            t_last = time.time(); got += len(b)
            x = np.concatenate((left, b)) if len(left) else b; m = len(x)//nf
            if m:
                acc += np.sum(np.abs(np.fft.fft(x[:m*nf].reshape(m, nf)*win, axis=1))**2, 0); cnt += m
            left = x[m*nf:]
        return np.fft.fftshift(acc/max(cnt, 1))

    def source_name(self):
        remote = self.remote or os.environ.get("PYSDRLIB_REMOTE")
        return f"{radio_io.driver_name(self.driver, remote)} " + (f"at {remote}" if remote else "(local)")

    def ok_at(self, headroom):
        """Predicate for GainHold: set the gain, look at 2**17 samples; acceptable when nothing clips and the peak is under `headroom` of full scale"""
        def ok(g):
            radio_io.set_gain(self.sdr, g); time.sleep(0.03); self.last_gain = g
            x = radio_io.grab(self.sdr, 2**17)
            return radio_io.clipped(x) < 1e-4 and float(max(np.abs(x.real).max(), np.abs(x.imag).max())) < headroom
        return ok

    def place(self, db_raw, cf, nf, step, ov, tmpl, state, gain, observe=True):
        """One tile into global bins: subtract the shape, add the level error of its gain setting (`self.cal`), keep the middle `step` MHz. `state` holds the previous tile's strip beyond the kept width; at a join
        between two settings the difference there is an observation for `self.cal`. Returns (bin indices in range, dB there, the level error applied, the kept dB)"""
        db = db_raw if tmpl is None else db_raw - tmpl
        fr = cf + (np.arange(nf) - nf//2)*BIN/1e6; off = np.abs(fr - cf)
        keep = (off < step/2) & (off > 0.15); ext = (off < step/2 + ov) & (off > 0.15)
        idx_e = np.rint((fr[ext] - self.lo)/(BIN/1e6) + 1e-6).astype(int)
        if ov and observe and state["prev"] is not None:
            self.cal.observe(state["prev"][2], gain, match_offset(state["prev"][0], state["prev"][1], idx_e, db[ext]))
        state["prev"] = (idx_e, db[ext], gain)
        adj = self.cal.offset(gain) if ov else 0.0; db = db + adj
        idx = np.rint((fr[keep] - self.lo)/(BIN/1e6) + 1e-6).astype(int); ok = (idx >= 0) & (idx < self.n)
        return idx[ok], db[keep][ok], adj, db[keep]

    def one_sweep(self):
        with self.lock:
            self.restart = False; lo, hi, n = self.lo, self.hi, self.n; p = dict(self.params); k_sweep = self.sweeps
            fs, step, shift = self.geometry(k_sweep)
            self.live = np.zeros(n, np.uint8); live = self.live
        nf = int(round(fs*1e6/BIN)); nsamp = self.dwell_samples(fs)
        if self.cur_fs != fs:
            self.sdr.set_rx_sample_rate(fs*1e6); self.cur_fs = fs; time.sleep(0.05)
        tiles = []; a = lo - shift   # the tile grid of this sweep starts `shift` below the range
        while a < hi:
            tiles.append(a + step/2); a += step
        shape = self.shapes.setdefault(round(fs, 3), TileShape(nf)); flat = p["flatten"] and len(tiles) >= 8   # fewer tiles than that: a signal could pass for the shape
        ov = overlap(fs, step) if p["match"] else 0.0; state = dict(prev=None); raw_tiles = []; odd = bool(k_sweep % 2 and p["alt_gain"]); holder = self.holders[1 if odd else 0]
        t0 = time.time()
        for k, cf in enumerate(tiles):
            while self.pause and not self.stop.is_set() and not self.restart:
                time.sleep(0.2)
            if self.stop.is_set() or self.restart:
                return
            tt = time.time(); gain = p["gain"]
            self.sdr.set_rx_freq(cf*1e6); time.sleep(p["settle"])
            if gain is not None:
                radio_io.set_gain(self.sdr, gain); time.sleep(0.03)
            else:
                self.last_gain = None
                gain = holder.pick(self.ok_at(p["headroom"]))
                if self.last_gain != gain:
                    radio_io.set_gain(self.sdr, gain); time.sleep(0.03)
            P = self.capture_psd(nsamp, nf)
            db = 10*np.log10(P + 1e-20) - radio_io.gain_db(self.sdr)   # relative to the input at 0 dB gain
            tmpl = shape.template() if flat else None
            if flat:
                shape.update(db)
            self.cal.seen(gain); raw_tiles.append((cf, db, gain))
            idx, vals, adj, d = self.place(db, cf, nf, step, ov, tmpl, state, gain)
            if self.debug:
                w = len(d)//4
                print(f"sweep {k_sweep} tile {k + 1:3d}/{len(tiles)} {cf:9.3f} MHz fs {fs:g} gain {gain:.2f} floor {np.percentile(d, 30):6.1f} dB "
                      f"edge-centre {np.median(np.r_[d[:w], d[-w:]]) - np.median(d[w:-w]):+5.1f} dB shape {'yes' if tmpl is not None else 'no '} level {adj:+5.1f} dB", file=sys.stderr, flush=True)
            with self.lock:
                if self.restart:
                    return
                live[idx] = q(vals); self.tile = dict(cf=cf, gain=gain, i=k + 1, n=len(tiles)); self.changed = True
                self.tile_s = 0.8*self.tile_s + 0.2*(time.time() - tt) if self.tile_s else time.time() - tt
        tmpl = shape.template() if flat else None
        if ov:
            self.cal.solve()
        if tmpl is not None or ov:    # the tiles measured before the shape and the gain errors were known are redone with what the whole sweep taught
            state = dict(prev=None); fresh = np.zeros(n, np.uint8)
            for cf, db, g in raw_tiles:
                idx, vals, _, _ = self.place(db, cf, nf, step, ov, tmpl, state, g, observe=False); fresh[idx] = q(vals)
            with self.lock:
                if not self.restart:
                    live[:] = fresh
        with self.lock:
            if self.restart:
                return
            confirm = p["alt_fs"] or p["alt"] or p["alt_gain"]      # the neighbouring sweeps differ in rate, grid or gain
            self.rows.appendleft((self.sweeps, live.copy()))
            self.sweeps += 1; self.t_sweep = time.time() - t0
            if confirm:
                # a sweep is confirmed by both neighbours: the previous one is final now (the newest only has its older neighbour, and is revised when the next one is done)
                self.prows.appendleft((self.sweeps - 1, persistent(live, self.rows[1][1] if len(self.rows) > 1 else None)))
                if len(self.rows) > 1:
                    prev = persistent(self.rows[1][1], self.rows[2][1] if len(self.rows) > 2 else None)
                    final = persistent(prev, live); self.prows[1] = (self.sweeps - 2, final); self.hold(final, p["peak_decay"])
            else:
                self.prows.appendleft((self.sweeps - 1, live.copy())); self.hold(live, p["peak_decay"])
            self.raw = pmean([r for _, r in list(self.rows)[:8]]); self.avg = pmean([r for _, r in list(self.prows)[:8]])
            self.emitters = self.find_emitters(unq(self.avg), self.avg > 0); self.flat = p["flatten"] and shape.template() is not None

    def find_emitters(self, db, have):
        """Peaks of the averaged sweep: more than 8 dB over the 20th percentile of a 10 MHz window"""
        if have.sum() < 100:
            return []
        p = np.where(have, db, np.nan); ctr = np.arange(len(p))
        p = np.interp(ctr, ctr[have], db[have])
        floor = percentile_filter(p, 20, size=min(len(p), int(10e6/BIN)) | 1, mode="nearest"); ex = p - floor
        f = self.f0 + np.arange(len(p))*BIN/1e6; r = f % 20; ex[np.minimum(r, 20 - r) < 0.13] = 0    # the HackRF's own spurs at multiples of 20 MHz
        on = ex > 8; out = []; i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j < len(on) and (on[j] or on[j:j + 3].any()):
                    j += 1
                out.append(dict(f=round(float((f[i] + f[j - 1])/2), 3), w=round(float((f[j - 1] - f[i]) + BIN/1e6), 3), ex=round(float(ex[i:j].max()), 1), db=round(float(p[i:j].max()), 1),
                                band=wide_survey.label((f[i] + f[j - 1])/2)))
                i = j
            else:
                i += 1
        return sorted(out, key=lambda e: -e["ex"])[:20]

    # -- for the web page
    def status(self):
        with self.lock:
            return dict(lo=self.lo, hi=self.hi, n=self.n, f0=self.f0, bin_mhz=BIN/1e6, d0=D0, d1=D1, params=dict(self.params, fs=self.fs_mhz(), fs_min=self.rate_limits()[0], fs_max=self.rate_limits()[1], fs_auto=not self.params['fs'], dwell_samples=self.dwell_samples(self.fs_mhz())), tile_s=round(self.tile_s, 2), tile=self.tile, sweeps=self.sweeps, t_sweep=round(self.t_sweep, 1), paused=self.pause, error=self.error,
                        source=self.source_name(), emitters=self.emitters, step=self.params['step'], flat=self.flat, geometry=[self.geometry(k) for k in (0, 1)], cal={f"{g:.2f}": round(o, 1) for g, o in self.cal.off.items()},
                        bands=[dict(lo=lo, hi=hi, name=nm) for lo, hi, nm in wide_survey.ALLOCATIONS], latest=(self.rows[0][0] if self.rows else None), keep=KEEP)


def make_handler(sw, page):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def local(self):
            host = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
            if host not in ("127.0.0.1", "localhost", "::1") and not self.server.allow_remote:
                self.send_error(403, "local access only"); return False
            return True

        def reply(self, body, ctype, code=200):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.local():
                return
            p = self.path.split("?")[0]
            if p == "/":
                self.reply(page, "text/html; charset=utf-8")
            elif p == "/api/status":
                self.reply(json.dumps(sw.status()).encode(), "application/json")
            elif p == "/api/live":
                with sw.lock:
                    b = sw.live.tobytes()
                self.reply(b, "application/octet-stream")
            elif p == "/api/summary":   # confirmed average, peak hold, unconfirmed average: n bytes each
                with sw.lock:
                    b = sw.avg.tobytes() + sw.peak.tobytes() + sw.raw.tobytes()
                self.reply(b, "application/octet-stream")
            elif p.startswith("/api/row/"):
                try:
                    k = int(p.rsplit("/", 1)[1])
                    with sw.lock:
                        row = next(r for i, r in sw.prows if i == k)   # the confirmed sweep
                        b = row.tobytes()
                    self.reply(b, "application/octet-stream")
                except (ValueError, StopIteration):
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            if not self.local():
                return
            if self.path != "/api/control" or "application/json" not in self.headers.get("Content-Type", ""):
                self.send_error(400); return
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if "pause" in d:
                    sw.pause = bool(d["pause"])
                if d.get("reset_peak"):
                    sw.reset_peak()
                if any(k in d for k in ("step", "fs", "dwell", "settle", "gain", "alt", "alt_fs", "alt_gain", "flatten", "match", "headroom", "peak_decay")):
                    sw.configure(d)
                if "lo" in d and "hi" in d:
                    lo, hi = float(d["lo"]), float(d["hi"])
                    if not (1 <= lo < hi <= 6000 and hi - lo >= 4):
                        raise ValueError
                    sw.set_range(lo, hi)
            except (ValueError, TypeError, AssertionError, KeyError):
                self.send_error(400); return
            self.reply(json.dumps(sw.status()).encode(), "application/json")
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lo", type=float, default=30.0, help="[MHz]"); ap.add_argument("--hi", type=float, default=2000.0, help="[MHz]")
    ap.add_argument("--gain", type=float, help="fixed gain (a fraction 0-1 of the radio's range) instead of the highest non-clipping rung per tile")
    radio_io.add_radio_args(ap)
    ap.add_argument("--tile", type=float, default=16.0, help="tile width [MHz] (1-20); the sample rate follows unless --fs is given")
    ap.add_argument("--fs", type=float, default=0.0, help="sample rate [MS/s] (within the radio's rates, default: tile / 0.8)")
    ap.add_argument("--dwell-samples", type=int, help="samples averaged per tile (default 524288)"); ap.add_argument("--dwell-seconds", type=float, help="time averaged per tile, instead of --dwell-samples")
    ap.add_argument("--settle", type=float, default=0.04, help="wait after each retune [s]")
    ap.add_argument("--headroom", type=float, default=0.5, help="auto gain: the largest peak amplitude allowed, as a fraction of full scale (0.1-1; lower keeps overload products out)")
    ap.add_argument("--peak-decay", type=float, default=0.25, help="peak hold falls by this many dB per sweep (0 holds for ever)")
    ap.add_argument("--no-alt-fs", action="store_true", help="do not alternate the sample rate between sweeps (the artifact test then only has the shifted grid)")
    ap.add_argument("--no-alt-gain", action="store_true", help="do not run the odd sweeps at a slightly lower gain (the test for overload and intermodulation products of the front end)")
    ap.add_argument("--no-flatten", action="store_true", help="do not subtract the shape the tiles have in common")
    ap.add_argument("--no-match", action="store_true", help="do not level neighbouring tiles against each other")
    ap.add_argument("--debug-tiles", action="store_true", help="print each tile's gain, floor, edge-versus-centre level, flattening and level match on stderr")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8026)
    a = ap.parse_args(argv)
    dwell = dict(unit="seconds", value=a.dwell_seconds) if a.dwell_seconds else dict(unit="samples", value=a.dwell_samples or 2**19)
    opts = dict(step=a.tile, fs=a.fs, dwell=dwell, settle=a.settle, headroom=a.headroom, peak_decay=a.peak_decay, alt_fs=not a.no_alt_fs, alt_gain=not a.no_alt_gain, flatten=not a.no_flatten, match=not a.no_match)
    sw = Sweeper(a.lo, a.hi, a.remote, a.gain, debug=a.debug_tiles, driver=a.radio, **opts)
    sw.configure(opts)   # validates
    threading.Thread(target=sw.run, daemon=True).start()
    page = open(os.path.join(HERE, "sweep_monitor.html"), "rb").read()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(sw, page)); srv.allow_remote = a.host not in ("127.0.0.1", "localhost", "::1"); srv.daemon_threads = True
    print(f"sweep monitor: http://{a.host}:{a.port}/  ({a.lo:g}-{a.hi:g} MHz)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sw.stop.set()


if __name__ == "__main__":
    main()
