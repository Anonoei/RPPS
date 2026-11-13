"""Live ATSC (8VSB) monitor: find the stations, list the programs of each, watch one live

    python projects/atsc/atsc_live.py                       # interactive monitor (curses): up/down select, Enter programs, w watch, x stop, r rescan, q quit
    python projects/atsc/atsc_live.py scan                  # one pass over the TV channels: pilot level of each, then exit
    python projects/atsc/atsc_live.py list 35               # decode channel 35 and print its programs (PSIP names, streams)
    python projects/atsc/atsc_live.py watch 35 3.1          # decode live and play virtual channel 3.1 (or a program number) of channel 35 in ffplay
    python projects/atsc/atsc_live.py watch 35 --player file:ch35.ts      # ... or write the transport stream (--player vlc, --player stdout)
    python projects/atsc/atsc_live.py --remote tcp://HOST:5555      # use a pysdrlib server's HackRF (or set PYSDRLIB_REMOTE)
    python projects/atsc/atsc_live.py --replay data/atsc/atsc35_6s.cf64 --replay-channel 35 --replay-cf 604    # no radio: play a capture as if it were air

How it works. A sweep tunes each channel (lower edge + 8 MHz, 20 MS/s, so the HackRF's DC spike is outside the 6 MHz) and measures the pilot over the data around it (cheap: an FFT).
Channels with a pilot are probed in background processes: capture ~2 s, `atsc_probe.recover` (pilot lock, clock, equalizer), trellis, `atsc_decode.rs_decode` (all Reed-Solomon codewords in one rpps._native call),
then the PAT / PMT / PSIP tables give the programs and their names. Watching streams the radio continuously: the capture is cut into chunks of ~2 s that overlap by a few fields, each chunk is
decoded by a pool of worker processes (about 8 CPU-seconds per second of air), the codewords of every chunk get an absolute field number from their time, so overlaps are dropped and the stream is
stitched, and the packets of the chosen program are paced out at the transport stream's own rate to the player. Latency is the chunk plus the decode (~15-25 s with 8+ workers).
The signal has to be good: ATSC needs ~15 dB SNR. The SNR column is measured on the segment syncs, which a decision-directed equalizer cannot flatter (atsc_probe's `mer_eq` can).
Meant to be upstreamed into RPPS.
"""
import argparse
import curses
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import atsc_decode as ad
import atsc_probe as ap
import atsc_rx
import radio_io
import ts_info

FS = 20e6
CF_OFFSET = 8.0 # tune centre above the channel's lower edge [MHz]: the DC spike is 2 MHz outside the channel, the whole channel inside the HackRF's filter
PILOT_MIN = 10.0 # dB over the data around the pilot to call a channel a station
FIELD_S = ap.FIELD*ap.SEG/ap.SYMBOL_RATE # one field: 24.2 ms
CW_S = FIELD_S/ad.FIELD_DATA # one transport packet (one Reed-Solomon codeword)
FIELD_CW = ad.FIELD_DATA
OVERLAP_FIELDS = 3 # chunks overlap by this many fields so the end of one (lost to the deinterleaver) is the middle of the next
GAIN_COARSE = [0.55, 0.45, 0.35] # gain fractions tried first by pick_gain, then the steps either side of the best (a HackRF decodes a given channel over a narrow window: too high overloads it, too low leaves the 8-bit ADC short of bits)
GAIN_STEP, GAIN_RANGE = 0.05, (0.2, 0.7)
GAIN_SECONDS = 2.0 # capture per trial (the receiver needs ~1.3 s to acquire)
DEFAULT_GAIN = 0.4 # where a channel is first looked at (the pilot sweep): clean on a HackRF, no overload from the neighbours
CHANNELS = list(range(7, 14)) + list(range(14, 37))


def lower_edge(ch):
    """Lower edge of an US ATSC channel [MHz]: UHF 14-36 (470 + 6 (n - 14)) and VHF-high 7-13 (174 + 6 (n - 7)); also 2-6 / 37-51 for completeness"""
    if ch >= 14:
        return 470.0 + 6*(ch - 14)
    if ch >= 7:
        return 174.0 + 6*(ch - 7)
    return {2: 54.0, 3: 60.0, 4: 66.0, 5: 76.0, 6: 82.0}[ch]


def centre(ch):
    return (lower_edge(ch) + CF_OFFSET)*1e6

# ------------------------------------------------------------------------------------------------ radios

class Radio:
    """A radio (local, or a pysdrlib server's with `remote`; `driver` picks which, see radio_io.driver_name): tune with a flush of the stale samples, per-channel gain, grab, read"""
    STALE = 0.45 # seconds of samples still queued from the old frequency after a retune (a remote HackRF has ~0.25 s in its pipeline)

    def __init__(self, remote=None, driver=None, channels=None, serial=None, label=None, program="RPPS atsc"):
        """channels: the TV channels this will tune across: a remote server is told the span and the sample rate it needs, so it hands out a radio that can (and not, say, an HF-only one)"""
        remote = remote or os.environ.get("PYSDRLIB_REMOTE")
        cfs = [centre(c) for c in channels or CHANNELS]
        self.sdr = radio_io.open_radio(cf=500e6, fs=FS, remote=remote, driver=driver, program=program, serial=serial, label=label,
                                       need=dict(freq=(min(cfs), max(cfs)), rate=FS), want=dict(freq=cfs[0], rate=FS))
        self.sdr.start_rx()
        self.lock = threading.RLock() # one user of the radio at a time
        self.name = f"{getattr(self.sdr, 'id', None) or radio_io.driver_name(driver, remote)} " + (f"at {remote}" if remote else "(local)")
        self.cf = 500e6

    @property
    def dropped(self):
        return getattr(self.sdr, "dropped_blocks", 0)

    def tune(self, cf, gain=None):
        self.sdr.set_rx_freq(cf); self.cf = cf
        if gain is not None:
            radio_io.set_gain(self.sdr, gain)
        time.sleep(0.05)
        self.grab(self.STALE)

    def gain(self, g):
        radio_io.set_gain(self.sdr, g)

    def grab(self, seconds):
        return radio_io.grab(self.sdr, int(FS*seconds))

    def get_samples(self):
        return self.sdr.get_samples()

    def close(self):
        try:
            self.sdr.stop_rx(); self.sdr.close()
        except Exception:
            pass


class ReplayRadio:
    """No radio: a capture file stands in for one channel's air, paced like a real stream (`speed` x real time), other channels are noise"""
    STALE = 0.0

    def __init__(self, path, channel, cf_mhz, speed=1.0):
        self.x = np.memmap(path, np.complex64, "r")
        self.channel = channel; self.file_cf = cf_mhz*1e6; self.speed = speed
        self.pos = 0; self.t = time.monotonic(); self.cf = 0.0
        self.gain_matters = False; self.lock = threading.RLock(); self.name = f"replay of {os.path.basename(path)} as channel {channel}"; self.dropped = 0
        self.rng = np.random.default_rng(1)

    def tune(self, cf, gain=None):
        self.cf = cf; self.t = time.monotonic()

    def gain(self, g):
        pass

    def _next(self, n):
        if abs(self.cf - self.file_cf) > 1e3:
            return (self.rng.standard_normal(n) + 1j*self.rng.standard_normal(n)).astype(np.complex64)*0.02
        idx = (self.pos + np.arange(n)) % len(self.x)
        s = self.pos % len(self.x)
        out = self.x[s:s + n] if s + n <= len(self.x) else np.asarray(self.x[idx])
        self.pos += n
        return np.array(out, np.complex64)

    def grab(self, seconds):
        n = int(FS*seconds)
        if self.speed: # a real radio takes this long to deliver them
            time.sleep(seconds/self.speed)
        self.t = time.monotonic()
        return self._next(n)

    def get_samples(self):
        now = time.monotonic(); n = int((now - self.t)*FS*(self.speed or 50)); n = min(n, int(FS*0.5))
        if n < 100000:
            return np.zeros(0, np.complex64)
        self.t = now
        return self._next(n)

    def close(self):
        pass


def pilot_level(radio, ch, gain=None):
    """Tune to `ch`, return (pilot dB over the data around it, clipped fraction, samples)"""
    cf = centre(ch)
    radio.tune(cf, DEFAULT_GAIN if gain is None else gain)
    x = radio.grab(0.2).astype(np.complex64)
    clip = radio_io.clipped(x)
    try:
        _, over = ap.find_pilot(x, FS, cf, lower_edge(ch)*1e6)
    except Exception:
        over = 0.0
    return float(over), clip


def _gain_trial(radio, cf, ch, g):
    """(score, snr, rs_ok) of the channel at gain `g`: a capture through the streaming receiver. Overload (clipping) and no lock score worst; else the locked SNR, with the Reed-Solomon yield breaking ties"""
    radio.tune(cf, g)
    x = radio.grab(GAIN_SECONDS).astype(np.complex64)
    if radio_io.clipped(x) > 2e-4:
        return -1e3, None, 0.0
    _, err, st = atsc_rx.decode_array(x, cf, lower_edge(ch)*1e6)
    rs = float(np.mean(err >= 0)) if len(err) else 0.0
    snr = st.get("snr") if st["state"] == "locked" else None
    return (-100.0 if snr is None else snr) + 100*rs, snr, rs


def pick_gain(radio, ch, log=None):
    """The gain at which the channel decodes best: the coarse gains first, then the steps either side of the best. A radio whose gain does nothing (a replay) is not searched.
    No gain that locks: DEFAULT_GAIN. `log(text)` gets each trial"""
    if not getattr(radio, "gain_matters", True):
        return DEFAULT_GAIN
    cf = centre(ch); tried = {}
    def trial(g):
        g = round(min(max(g, GAIN_RANGE[0]), GAIN_RANGE[1]), 3)
        if g not in tried:
            tried[g] = _gain_trial(radio, cf, ch, g)
            if log:
                log(f"channel {ch} gain {g}: " + ("clipping" if tried[g][0] == -1e3 else f"SNR {tried[g][1]} dB, Reed-Solomon ok {100*tried[g][2]:.0f}%" if tried[g][1] is not None else "no lock"))
    for g in GAIN_COARSE:
        trial(g)
    best = max(tried, key=lambda g: tried[g][0])
    for g in (best - GAIN_STEP, best + GAIN_STEP):
        trial(g)
    best = max(tried, key=lambda g: tried[g][0])
    return best if tried[best][1] is not None else DEFAULT_GAIN

# ------------------------------------------------------------------------------------------------ transport stream tables

def parse_psi(ts):
    """Programs and names of a transport stream (bytes or (N, 188) array): {"programs": {n: {"pmt": pid, "pcr": pid, "streams": [(type, pid)]}}, "vct": {(major, minor): (name, program, source id)}}"""
    data = bytes(np.ascontiguousarray(ts)) if not isinstance(ts, bytes) else ts
    sec = ts_info.sections(data, {0, 0x1FFB})
    pat = {}
    for s in sec.get(0, []):
        if s[0] == 0 and len(s) > 12:
            for k in range(8, len(s) - 4, 4):
                n = (s[k] << 8) | s[k + 1]; pid = ((s[k + 2] & 0x1F) << 8) | s[k + 3]
                if n:
                    pat[n] = pid
    programs = {n: {"pmt": pid, "pcr": None, "streams": []} for n, pid in pat.items()}
    pm = ts_info.sections(data, set(pat.values())) if pat else {}
    for n, pid in pat.items():
        for s in pm.get(pid, [])[:1]:
            if s[0] != 2:
                continue
            pil = ((s[10] & 0x0F) << 8) | s[11]; k = 12 + pil; es = []
            while k + 5 <= len(s) - 4:
                es.append((s[k], ((s[k + 1] & 0x1F) << 8) | s[k + 2])); k += 5 + (((s[k + 3] & 0x0F) << 8) | s[k + 4])
            programs[n]["streams"] = es; programs[n]["pcr"] = ((s[8] & 0x1F) << 8) | s[9]
    vct = {}
    for s in sec.get(0x1FFB, []):
        if s[0] in (0xC8, 0xC9) and len(s) > 14:
            k = 10
            for _ in range(s[9]):
                if k + 32 > len(s) - 4:
                    break
                name = s[k:k + 14].decode("utf-16-be", "replace").rstrip("\x00")
                major = ((s[k + 14] & 0x0F) << 6) | (s[k + 15] >> 2); minor = ((s[k + 15] & 0x03) << 8) | s[k + 16]
                vct[(major, minor)] = (name, (s[k + 24] << 8) | s[k + 25], (s[k + 28] << 8) | s[k + 29])
                k += 32 + (((s[k + 30] & 0x03) << 8) | s[k + 31])
    return {"programs": programs, "vct": vct}


def merge_psi(old, new):
    out = {"programs": dict(old["programs"]) if old else {}, "vct": dict(old["vct"]) if old else {}}
    for n, p in new["programs"].items():
        if p["streams"] or n not in out["programs"]:
            out["programs"][n] = p
    out["vct"].update(new["vct"])
    return out


def describe(psi):
    """Lines for a program table: 'major.minor NAME  program N: MPEG-2 video, AC-3 audio'"""
    name_of = {p: (mm, nm) for mm, (nm, p, _) in psi["vct"].items()}
    rows = []
    for n, p in sorted(psi["programs"].items()):
        mm, nm = name_of.get(n, (None, ""))
        streams = ", ".join(ts_info.STREAM.get(t, f"type {t:#x}") for t, _ in p["streams"]) or "(PMT not seen yet)"
        label = f"{mm[0]}.{mm[1]}" if mm else "   ?"
        rows.append((mm or (999, n), f"{label:>7s} {nm:8s} program {n:<3d} {streams}"))
    return [r for _, r in sorted(rows)]


def resolve_program(psi, want):
    """Program number for `want`: 'N.M' (virtual channel), 'N' (program number), None (the lowest virtual channel with video)"""
    progs = psi["programs"]
    if want and "." in str(want):
        major, minor = (int(v) for v in str(want).split(".", 1))
        for (ma, mi), (_, n, _) in psi["vct"].items():
            if (ma, mi) == (major, minor) and n in progs:
                return n
        raise SystemExit(f"no virtual channel {want}; have: " + ", ".join(f"{a}.{b}" for a, b in sorted(psi["vct"])))
    if want:
        if int(want) in progs:
            return int(want)
        raise SystemExit(f"no program {want}; have: {sorted(progs)}")
    name_of = {p: mm for mm, (_, p, _) in psi["vct"].items()}
    video = [n for n, p in progs.items() if any(t in (0x02, 0x1B, 0x24) for t, _ in p["streams"])]
    return sorted(video or progs, key=lambda n: (name_of.get(n, (999, n)), n))[0]


def program_pids(psi, n):
    """PIDs to keep for program n: PAT, PSIP, its PMT, PCR and elementary streams"""
    p = psi["programs"][n]
    return {0, 0x1FFB, p["pmt"]} | ({p["pcr"]} if p["pcr"] is not None else set()) | {pid for _, pid in p["streams"]}

# ------------------------------------------------------------------------------------------------ worker: capture -> transport packets

def worker_init():
    import warnings
    warnings.filterwarnings("ignore")


def decode_capture(x, fs, cf, lower, want_psi=False):
    """Decode one capture (complex64) to transport packets. Returns a dict:
       locked, pilot [dB], ppm, mer_eq, snr (segment syncs), and when decoded: ts (codewords, 188) with errors (-1 uncorrectable), t_field0 (seconds into the capture of the first decoded field),
       rs_ok (fraction of codewords corrected), psi (if want_psi)"""
    t0 = time.time()
    r = ap.recover(np.asarray(x, np.complex64), fs, cf, lower, verbose=False)
    out = {"locked": r.fs_seg is not None, "ppm": float(r.ppm), "mer_raw": float(r.mer_raw), "mer_eq": None, "snr": None, "rs_ok": None, "ts": None, "err": None, "t_field0": None, "psi": None}
    if not out["locked"]:
        out["secs"] = time.time() - t0
        return out
    out["mer_eq"] = float(r.mer_eq)
    sy = r.soft[:, :4]*np.array([1, -1, -1, 1]); keep = (np.arange(len(sy)) % ap.FIELD) != r.fs_seg
    out["snr"] = float(10*np.log10(25/np.mean((sy[keep] - 5)**2)))
    fields = [(k0, b) for k0, b in ad.decode_fields(r.soft, r.fs_seg) if len(b) == ad.FIELD_DATA*207]
    runs, cur = [], []
    for f in fields: # one unbroken run of fields (the byte interleaver runs across them)
        if cur and f[0] != cur[-1][0] + ap.FIELD:
            runs.append(cur); cur = []
        cur.append(f)
    if cur:
        runs.append(cur)
    if not runs:
        out["secs"] = time.time() - t0
        return out
    run = max(runs, key=len)
    z = ad.deinterleave(np.concatenate([b for _, b in run]))
    cw = z[ad.DELAY:ad.DELAY + (len(z) - ad.DELAY)//207*207].reshape(-1, 207)
    data, err = ad.rs_decode(cw)
    ts = ad.transport_stream(ad.derandomize(data), err)
    out.update(ts=ts, err=err, rs_ok=float(np.mean(err >= 0)), t_field0=(r.start + (run[0][0] - 1)*ap.SEG)/ap.SYMBOL_RATE)
    if want_psi:
        out["psi"] = parse_psi(ts[err >= 0])
    out["secs"] = time.time() - t0
    return out


def probe_capture(x, fs, cf, lower):
    """decode_capture for the monitor: the numbers and the programs, not the packets"""
    o = decode_capture(x, fs, cf, lower, want_psi=True)
    o["ts"] = None; o["err"] = None
    return o

# ------------------------------------------------------------------------------------------------ the monitor

class Station:
    def __init__(self, ch):
        self.ch = ch; self.lower = lower_edge(ch)
        self.pilot = None; self.clip = 0.0; self.seen = 0.0; self.gain = None
        self.locked = None; self.snr = None; self.mer_eq = None; self.rs_ok = None; self.ppm = None
        self.psi = None; self.probe_t = 0.0; self.probing = False; self.note = ""

    @property
    def is_station(self):
        return self.pilot is not None and self.pilot >= PILOT_MIN

    @property
    def status(self):
        if not self.is_station:
            return "-"
        if self.probing and self.locked is None:
            return "probing"
        if self.locked is None:
            return "pilot"
        if not self.locked:
            return "no lock"
        if self.rs_ok is None:
            return "locked"
        return "OK" if self.rs_ok >= 0.9 else ("marginal" if self.rs_ok >= 0.5 else "weak")

    def names(self):
        if not self.psi or not self.psi["vct"]:
            return ""
        return "  ".join(f"{a}.{b} {nm}" for (a, b), (nm, _, _) in sorted(self.psi["vct"].items()))


class Monitor(threading.Thread):
    """Sweeps the channels with the radio, keeps a Station per channel, probes the ones that have a pilot in background processes"""
    def __init__(self, radio, pool, channels, probe_every=90.0, probe_seconds=2.0, log=None):
        super().__init__(daemon=True)
        self.radio, self.pool, self.channels = radio, pool, channels
        self.stations = {ch: Station(ch) for ch in channels}
        self.probe_every, self.probe_seconds = probe_every, probe_seconds
        self.log = log or (lambda *a: None)
        self.stop_flag = threading.Event(); self.paused = threading.Event(); self.rescan = threading.Event(); self.rescan.set()
        self.current = None; self.sweeps = 0; self.futures = []

    def _probe(self, st):
        """Capture on this channel (the radio is ours) and hand the capture to a worker"""
        st.gain = st.gain or pick_gain(self.radio, st.ch)
        cf = centre(st.ch)
        self.radio.tune(cf, st.gain)
        x = self.radio.grab(self.probe_seconds).astype(np.complex64)
        if radio_io.clipped(x) > 2e-4:
            st.gain = None # too hot: choose again next time
        st.probing = True
        fut = self.pool.submit(probe_capture, x, FS, cf, st.lower*1e6)
        fut.add_done_callback(lambda f, st=st: self._probed(st, f))
        self.log(f"channel {st.ch}: capture sent to a worker (gain {st.gain})")

    def _probed(self, st, fut):
        st.probing = False; st.probe_t = time.time()
        try:
            o = fut.result()
        except Exception as e:
            st.note = f"{type(e).__name__}: {e}"[:60]; self.log(f"channel {st.ch}: probe failed: {st.note}"); return
        st.locked, st.snr, st.mer_eq, st.rs_ok, st.ppm = o["locked"], o["snr"], o["mer_eq"], o["rs_ok"], o["ppm"]
        if o["psi"]:
            st.psi = merge_psi(st.psi, o["psi"])
        self.log(f"channel {st.ch}: " + (f"locked, SNR {st.snr:.1f} dB, Reed-Solomon ok {100*(st.rs_ok or 0):.0f}%, {len(st.psi['programs']) if st.psi else 0} programs" if st.locked else "no lock") + f" ({o['secs']:.0f} s)")

    def run(self):
        while not self.stop_flag.is_set():
            try:
                self._sweep()
            except Exception as e: # a radio hiccup (remote link, USB) ends one pass, not the monitor
                if self.stop_flag.is_set():
                    return
                self.log(f"sweep error: {type(e).__name__}: {str(e)[:80]}"); time.sleep(1.0)

    def _sweep(self):
        if self.paused.is_set():
            time.sleep(0.2); return
        if self.sweeps and self.sweeps % 10 == 0 and not self.rescan.is_set():
            self.rescan.set() # now and then look at every channel again: a station may have come or gone
        full = self.rescan.is_set(); self.rescan.clear()
        todo = self.channels if full else [c for c in self.channels if self.stations[c].is_station]
        for ch in todo:
            if self.stop_flag.is_set() or self.paused.is_set():
                break
            st = self.stations[ch]
            with self.radio.lock:
                if self.paused.is_set():
                    break
                self.current = ch
                st.pilot, st.clip = pilot_level(self.radio, ch, st.gain); st.seen = time.time()
                if st.is_station and not st.probing and time.time() - st.probe_t > self.probe_every:
                    self._probe(st)
        self.current = None; self.sweeps += 1
        if not any(s.is_station for s in self.stations.values()):
            self.rescan.set()
        time.sleep(0.1)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=10) # before the radio is closed under it

# ------------------------------------------------------------------------------------------------ watching

class Watcher:
    """Stream one channel: capture continuously, decode chunks in the pool, stitch by absolute field number, pace the program's packets out to `sink` (a file object)"""
    def __init__(self, radio, pool, ch, sink, psi=None, program=None, chunk=2.0, gain=None, workers=8, delay=0.0, log=None):
        self.radio, self.pool, self.ch, self.sink = radio, pool, ch, sink
        self.psi, self.want, self.chunk, self.gain, self.delay = psi, program, chunk, gain, delay
        self.log = log or (lambda *a: None)
        self.max_pending = max(2, workers + 2)
        self.stop_flag = threading.Event(); self.pending = []; self.out_q = queue.Queue()
        self.stats = SimpleNamespace(chunks=0, skipped=0, failed=0, packets=0, bytes=0, rs_ok=None, snr=None, latency=None, dropped=0, behind=0.0, state="starting", program=None)
        self.ref_n = None; self.ref_t = None; self.last_cw = None; self.keep = None; self.program = None
        self.lock = threading.Lock(); self.threads = []

    def start(self):
        for target in (self._capture, self._collect, self._write):
            t = threading.Thread(target=target, daemon=True); t.start(); self.threads.append(t)
        return self

    def stop(self):
        self.stop_flag.set()
        for t in self.threads:
            t.join(timeout=3)

    # capture thread: blocks of samples -> overlapping chunks
    def _capture(self):
        with self.radio.lock:
            st_gain = self.gain or pick_gain(self.radio, self.ch)
            self.radio.tune(centre(self.ch), st_gain)
            n_chunk = int(FS*self.chunk); n_over = int(FS*OVERLAP_FIELDS*FIELD_S)
            buf = []; have = 0; start = 0; d0 = self.radio.dropped; self.stats.state = "capturing"
            while not self.stop_flag.is_set():
                b = self.radio.get_samples()
                if not len(b):
                    time.sleep(0.005); continue
                buf.append(np.asarray(b, np.complex64)); have += len(b)
                if have >= n_chunk + n_over:
                    x = np.concatenate(buf)
                    chunk, rest = x[:n_chunk + n_over], x[n_chunk:] # the next chunk starts n_chunk in, so it repeats the overlap
                    self.stats.dropped = self.radio.dropped - d0
                    self.pending = [f for f in self.pending if not f[0].done()]
                    if len(self.pending) >= self.max_pending:
                        self.stats.skipped += 1 # the pool cannot keep up: this chunk of air is lost (a gap in the stream)
                    else:
                        fut = self.pool.submit(decode_capture, chunk, FS, centre(self.ch), lower_edge(self.ch)*1e6, self.psi is None)
                        self.pending.append((fut, start/FS, time.time()))
                    start += n_chunk; buf = [rest]; have = len(rest)

    # collect thread: finished chunks in order -> field numbers -> packets
    def _collect(self):
        done_upto = 0
        while not self.stop_flag.is_set():
            if not self.pending or not self.pending[0][0].done():
                time.sleep(0.05); continue
            fut, t_start, t_sub = self.pending.pop(0)
            try:
                o = fut.result()
            except Exception as e:
                self.stats.failed += 1; self.log(f"chunk failed: {type(e).__name__}: {e}"); continue
            self.stats.chunks += 1; self.stats.latency = time.time() - t_sub
            if o["snr"] is not None:
                self.stats.snr = o["snr"]
            if not o["locked"] or o["ts"] is None:
                self.stats.failed += 1; self.stats.state = "no lock"; continue
            self.stats.rs_ok = o["rs_ok"]; self.stats.state = "playing" if o["rs_ok"] > 0.5 else "weak signal"
            if o["psi"]:
                self.psi = merge_psi(self.psi, o["psi"])
            if self.keep is None and self.psi and self.psi["programs"]:
                try:
                    self.program = resolve_program(self.psi, self.want)
                    if self.psi["programs"][self.program]["streams"]:
                        self.keep = program_pids(self.psi, self.program); self.stats.program = self.program
                        self.log(f"program {self.program}: " + describe(self.psi)[0])
                except SystemExit as e:
                    self.log(str(e)); self.stats.state = "program not found"
            if self.keep is None:
                continue # waiting for the program table
            T = t_start + o["t_field0"] # when (radio clock) the first decoded field started
            if self.ref_n is None:
                n = 0
            else:
                n = self.ref_n + int(round((T - self.ref_t)/FIELD_S))
            self.ref_n, self.ref_t = n, T
            ts, err = o["ts"], o["err"]
            c0 = n*FIELD_CW
            c = c0 + np.arange(len(ts))
            pid = ((ts[:, 1].astype(np.int32) & 0x1F) << 8) | ts[:, 2]
            sel = (err >= 0) & np.isin(pid, list(self.keep)) & ((c > self.last_cw) if self.last_cw is not None else True)
            if (err >= 0).any():
                self.last_cw = max(self.last_cw if self.last_cw is not None else -1, int(c[err >= 0].max()))
            for f0 in range(0, len(ts), FIELD_CW): # one field per queue item, paced by its time
                m = sel[f0:f0 + FIELD_CW]
                if m.any():
                    self.out_q.put((int(c[f0]), ts[f0:f0 + FIELD_CW][m].tobytes()))

    # writer thread: pace to the stream's rate
    def _write(self):
        wall0 = None; c_first = None
        while not self.stop_flag.is_set():
            try:
                c, data = self.out_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if wall0 is None:
                wall0, c_first = time.time() + self.delay, c
            due = wall0 + (c - c_first)*CW_S
            wait = due - time.time()
            self.stats.behind = max(0.0, -wait)
            if wait > 0:
                self.stop_flag.wait(wait)
            try:
                self.sink.write(data); self.sink.flush()
            except (BrokenPipeError, ValueError, OSError):
                self.stats.state = "player closed"; self.stop_flag.set(); return
            self.stats.packets += len(data)//188; self.stats.bytes += len(data)


def open_player(spec, title):
    """(sink file object, process or None) for --player: ffplay (default), vlc, stdout, file:PATH"""
    if spec == "stdout":
        return sys.stdout.buffer, None
    if spec.startswith("file:"):
        return open(spec[5:], "wb"), None
    if spec == "vlc":
        cmd = ["vlc", "-", "--meta-title", title]
    else:
        cmd = ["ffplay", "-loglevel", "warning", "-window_title", title, "-probesize", "2000000", "-analyzeduration", "2000000", "-i", "-"]
    if not shutil.which(cmd[0]):
        raise SystemExit(f"{cmd[0]} not found: use --player file:PATH or stdout")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=None if spec != "vlc" else subprocess.DEVNULL)
    return p.stdin, p

# ------------------------------------------------------------------------------------------------ front ends

def make_radio(a):
    if a.replay:
        return ReplayRadio(a.replay, a.replay_channel, a.replay_cf, speed=a.replay_speed)
    return Radio(a.remote, a.radio, a.channels, a.serial, a.label, "RPPS atsc_live")


def make_pool(a):
    n = a.workers or max(2, min(12, (os.cpu_count() or 4) - 2))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1" # one thread per worker: BLAS threads in 12 workers cost 3x the CPU for the same result and starve the pool
    return ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=worker_init), n


def sweep_once(radio, channels, log=print):
    out = {}
    for ch in channels:
        with radio.lock:
            over, clip = pilot_level(radio, ch)
        out[ch] = (over, clip)
        log(f"channel {ch:2d} ({lower_edge(ch):g}-{lower_edge(ch) + 6:g} MHz)  pilot {over:5.1f} dB" + ("  <== station" if over >= PILOT_MIN else "") + (f"  (clipping {clip:.3f})" if clip > 1e-3 else ""))
    return out


def cmd_scan(a):
    radio = make_radio(a)
    try:
        r = sweep_once(radio, a.channels or CHANNELS)
        print("\nstations: " + ", ".join(str(c) for c, (o, _) in r.items() if o >= PILOT_MIN))
    finally:
        radio.close()


def probe_blocking(radio, pool, ch, seconds):
    with radio.lock:
        g = pick_gain(radio, ch)
        radio.tune(centre(ch), g)
        x = radio.grab(seconds).astype(np.complex64)
    return pool.submit(probe_capture, x, FS, centre(ch), lower_edge(ch)*1e6).result(), g


def cmd_list(a):
    radio = make_radio(a); pool, _ = make_pool(a)
    try:
        for ch in a.channel:
            over, _ = pilot_level(radio, ch)
            print(f"channel {ch} ({lower_edge(ch):g}-{lower_edge(ch) + 6:g} MHz): pilot {over:.1f} dB")
            if over < PILOT_MIN:
                print("  no station"); continue
            o, g = probe_blocking(radio, pool, ch, a.seconds)
            if not o["locked"]:
                print("  pilot but no lock (too weak, or multipath the equalizer cannot undo)"); continue
            print(f"  locked: SNR {o['snr']:.1f} dB, Reed-Solomon corrects {100*o['rs_ok']:.0f}% of codewords (gain {g:.2f}, clock {o['ppm']:+.1f} ppm)")
            for line in describe(o["psi"]) if o["psi"] else ["  no program tables decoded (signal too weak?)"]:
                print("  " + line)
    finally:
        pool.shutdown(cancel_futures=True); radio.close()


def format_stats(w):
    s = w.stats
    rs = f"{100*s.rs_ok:.0f}%" if s.rs_ok is not None else "-"
    snr = f"{s.snr:.1f} dB" if s.snr is not None else "-"
    lat = f"{s.latency:.0f} s" if s.latency else "-"
    return f"{s.state:12s} program {s.program or '-'}  RS ok {rs}  SNR {snr}  chunks {s.chunks} (skipped {s.skipped}, failed {s.failed})  decode latency {lat}  dropped blocks {s.dropped}  out {s.bytes/1e6:.1f} MB"


def cmd_watch(a):
    radio = make_radio(a); pool, n = make_pool(a)
    sink = proc = w = None
    try:
        print(f"channel {a.channel}: finding the programs ...", flush=True)
        psi = None
        o, g = probe_blocking(radio, pool, a.channel, 2.0)
        if not o["locked"]:
            raise SystemExit("no lock on this channel (pilot too weak or ISI too strong); try `scan` / `list`")
        psi = o["psi"]
        print("\n".join(describe(psi)) if psi else "(no program tables yet; will wait for them)", flush=True)
        title = f"ATSC {a.channel}" + (f" {a.program}" if a.program else "")
        sink, proc = open_player(a.player, title)
        w = Watcher(radio, pool, a.channel, sink, psi=psi, program=a.program, chunk=a.chunk, workers=n, delay=a.delay, log=lambda m: print(m, file=sys.stderr, flush=True)).start()
        t0 = time.time()
        while not w.stop_flag.is_set() and (not a.seconds or time.time() - t0 < a.seconds):
            time.sleep(1.0)
            print(format_stats(w), file=sys.stderr, flush=True)
            if proc is not None and proc.poll() is not None:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if w:
            w.stop()
        if sink and sink is not sys.stdout.buffer:
            try:
                sink.close()
            except Exception:
                pass
        if proc:
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        pool.shutdown(cancel_futures=True); radio.close()


def fmt_row(st, sel, width, mon):
    pilot = f"{st.pilot:5.1f}" if st.pilot is not None else "    -"
    snr = f"{st.snr:5.1f}" if st.snr is not None else "    -"
    rs = f"{100*st.rs_ok:4.0f}%" if st.rs_ok is not None else "    -"
    age = f"{time.time() - st.seen:4.0f}s" if st.seen else "    -"
    here = ">" if mon.current == st.ch else " "
    s = f"{here}{st.ch:3d} {st.lower:5.0f}-{st.lower + 6:<3.0f} {pilot} {snr} {rs}  {st.status:9s} {age}  {st.names()}"
    return s[:width - 1]


def tui(stdscr, a, radio, pool, nworkers):
    curses.curs_set(0); stdscr.timeout(250)
    if curses.has_colors():
        curses.start_color(); curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1); curses.init_pair(2, curses.COLOR_YELLOW, -1); curses.init_pair(3, curses.COLOR_RED, -1); curses.init_pair(4, curses.COLOR_CYAN, -1)
    mon = Monitor(radio, pool, a.channels or CHANNELS, probe_every=a.probe_every); mon.start()
    sel = 0; show = False; watcher = None; proc = None; sink = None; msg = "scanning ..."; wch = None
    msgs = []
    mon.log = lambda m: msgs.append(m)
    def stop_watch():
        nonlocal watcher, proc, sink, wch
        if watcher:
            watcher.stop()
        if sink and sink is not sys.stdout.buffer:
            try: sink.close()
            except Exception: pass
        if proc:
            try: proc.terminate()
            except Exception: pass
        watcher = proc = sink = wch = None
        mon.paused.clear()
    try:
        while True:
            rows = [s for s in mon.stations.values() if s.is_station or a.all] or []
            sel = min(sel, max(0, len(rows) - 1))
            h, w = stdscr.getmaxyx(); stdscr.erase()
            stdscr.addstr(0, 0, f" ATSC monitor   radio: {radio.name}   workers {nworkers}   sweeps {mon.sweeps}"[:w - 1], curses.A_BOLD)
            stdscr.addstr(1, 0, " up/down select   Enter programs   w watch   x stop   r rescan   q quit"[:w - 1], curses.A_DIM)
            stdscr.addstr(3, 0, "  CH  MHz       pilot   SNR  RS ok  status     seen  virtual channels"[:w - 1], curses.A_UNDERLINE)
            for i, st in enumerate(rows[:max(0, h - 14)]):
                col = {"OK": 1, "marginal": 2, "weak": 3, "no lock": 3}.get(st.status, 0)
                attr = (curses.color_pair(col) if col and curses.has_colors() else 0) | (curses.A_REVERSE if i == sel else 0)
                stdscr.addstr(4 + i, 0, fmt_row(st, i == sel, w, mon), attr)
            y = min(4 + len(rows), h - 12) + 1
            if not rows and y + 1 < h:
                stdscr.addstr(y - 1, 2, "no stations yet" + (" (sweeping)" if mon.current else ""), curses.A_DIM)
            if show and rows and y < h - 4:
                st = rows[sel]
                stdscr.addstr(y, 0, f" Channel {st.ch}: {st.status}" + (f", clock {st.ppm:+.1f} ppm" if st.ppm is not None else ""), curses.A_BOLD); y += 1
                lines = describe(st.psi) if st.psi else ["(no program tables yet: waiting for the probe)"]
                for ln in lines[:max(0, h - y - 5)]:
                    stdscr.addstr(y, 2, ln[:w - 3]); y += 1
            if watcher and h > 6:
                stdscr.addstr(h - 4, 0, f" WATCHING channel {wch}: " + format_stats(watcher)[:w - 25], curses.color_pair(4) if curses.has_colors() else 0)
            for i, m in enumerate(msgs[-2:]):
                if h > 3:
                    stdscr.addstr(h - 2 + i, 0, (" " + m)[:w - 1], curses.A_DIM)
            stdscr.refresh()
            k = stdscr.getch()
            if watcher and proc is not None and proc.poll() is not None:
                msgs.append("player closed"); stop_watch()
            if k in (ord("q"), 27):
                break
            elif k == curses.KEY_DOWN:
                sel = min(sel + 1, len(rows) - 1)
            elif k == curses.KEY_UP:
                sel = max(sel - 1, 0)
            elif k in (10, 13, curses.KEY_ENTER):
                show = not show
            elif k == ord("r"):
                mon.rescan.set(); [setattr(s, "probe_t", 0.0) for s in mon.stations.values()]
            elif k == ord("x") and watcher:
                stop_watch()
            elif k == ord("w") and rows and not watcher:
                st = rows[sel]
                if not st.psi or not st.psi["programs"] or not any(p["streams"] for p in st.psi["programs"].values()):
                    msgs.append(f"channel {st.ch}: no program table yet (probe first; it needs a lock)")
                    continue
                try:
                    prog = resolve_program(st.psi, None)
                    sink, proc = open_player(a.player, f"ATSC {st.ch}")
                except SystemExit as e:
                    msgs.append(str(e)); continue
                mon.paused.set()
                time.sleep(0.3)
                wch = st.ch
                watcher = Watcher(radio, pool, st.ch, sink, psi=st.psi, program=str(prog), chunk=a.chunk, gain=st.gain, workers=nworkers, delay=a.delay, log=lambda m: msgs.append(m)).start()
            if watcher and watcher.stop_flag.is_set():
                msgs.append(f"watch ended: {watcher.stats.state}"); stop_watch()
    finally:
        stop_watch(); mon.stop()


def cmd_monitor(a):
    radio = make_radio(a); pool, n = make_pool(a)
    try:
        if a.plain:
            mon = Monitor(radio, pool, a.channels or CHANNELS, probe_every=a.probe_every, log=lambda m: print(m, flush=True)); mon.start()
            t0 = time.time()
            try:
                while not a.seconds or time.time() - t0 < a.seconds:
                    time.sleep(1.0)
                    if mon.sweeps and int(time.time() - t0) % 10 == 0:
                        for s in mon.stations.values():
                            if s.is_station:
                                print(f"  ch {s.ch:2d} pilot {s.pilot:5.1f} dB  {s.status:9s} SNR {s.snr if s.snr is not None else float('nan'):5.1f}  RS ok {100*(s.rs_ok or 0):3.0f}%  {s.names()}", flush=True)
            except KeyboardInterrupt:
                pass
            mon.stop()
        else:
            curses.wrapper(tui, a, radio, pool, n)
    finally:
        pool.shutdown(cancel_futures=True, wait=False); radio.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    radio_io.add_radio_args(p)
    p.add_argument("--replay", metavar="CF64", help="use a capture file as the air instead of a radio (complex64, 20 MS/s)")
    p.add_argument("--replay-channel", type=int, default=35); p.add_argument("--replay-cf", type=float, default=604.0, help="the capture's centre [MHz]")
    p.add_argument("--replay-speed", type=float, default=1.0, help="x real time (0: as fast as it is asked for)")
    p.add_argument("--workers", type=int, help="decoder processes (default: cores - 2, at most 12)")
    p.add_argument("--channels", type=int, nargs="+", help="channels to sweep (default: VHF 7-13 and UHF 14-36)")
    p.add_argument("--player", default="ffplay", help="ffplay, vlc, stdout or file:PATH")
    p.add_argument("--chunk", type=float, default=2.0, help="seconds of air per decode chunk when watching")
    p.add_argument("--delay", type=float, default=2.0, help="extra seconds of buffer before the first packet goes to the player")
    p.add_argument("--probe-every", type=float, default=90.0, help="seconds between probes of a station in the monitor")
    p.add_argument("--all", action="store_true", help="monitor: show every channel, not only those with a pilot")
    p.add_argument("--plain", action="store_true", help="monitor: print lines instead of the curses screen")
    p.add_argument("--seconds", type=float, default=0, help="stop after this long (monitor --plain, watch)")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("scan", help="pilot level of every channel, once")
    s = sub.add_parser("list", help="decode channels and list their programs"); s.add_argument("channel", type=int, nargs="+"); s.add_argument("--seconds", type=float, default=2.5)
    s = sub.add_parser("watch", help="play a channel live"); s.add_argument("channel", type=int); s.add_argument("program", nargs="?", help="virtual channel N.M or program number (default: the first with video)")
    sub.add_parser("monitor", help="the live table (default)")
    a = p.parse_args(argv)
    {"scan": cmd_scan, "list": cmd_list, "watch": cmd_watch, None: cmd_monitor, "monitor": cmd_monitor}[a.cmd](a)


if __name__ == "__main__":
    main()
