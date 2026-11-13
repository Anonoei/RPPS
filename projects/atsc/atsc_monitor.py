"""Live ATSC monitor (8VSB TV): one HackRF (or a recording played back as if live) -> find the stations, list their programs, watch one in the browser in real time

    python projects/atsc/atsc_monitor.py                                    # then open http://127.0.0.1:8035/
    python projects/atsc/atsc_monitor.py --remote tcp://HOST:5555     # the HackRF of a pysdrlib server
    python projects/atsc/atsc_monitor.py --radio uhd                         # the local B210
    python projects/atsc/atsc_monitor.py --channels 19 20 35                # only sweep these
    python projects/atsc/atsc_monitor.py --replay data/atsc/atsc35_6s.cf64 --replay-channel 35 --replay-cf 604 --replay-speed 0.5    # a capture as the air

What it does. A sweep tunes each TV channel (lower edge + 8 MHz, 20 MS/s) and measures the pilot; channels with a pilot are probed: a 2.5 s capture goes through the streaming receiver (atsc_rx.py) and the
page shows each station's SNR (on the segment syncs), the share of Reed-Solomon codewords it corrects, and its virtual channels (PSIP names, streams from the PAT / PMT). "Watch" gives the radio to one channel:
the receiver runs in real time (three threads: front end in C, equalizer regions, trellis + Reed-Solomon; ~1.3x real time on a laptop), the packets of the chosen program go to ffmpeg, which deinterlaces and
re-encodes to H.264 / AAC in fragmented MP4, and the page plays that through Media Source Extensions (`/video`). Latency is ~3-5 s. While watching, the sweep is paused (one radio), and the page shows the live
receiver: lock state, SNR, Reed-Solomon yield, clock error, how far behind real time it is, per-thread load, bit rate, a spectrum of the channel and a history of SNR / yield.
The web server listens on 127.0.0.1 only (--host to change it); requests must carry a local Host header. Needs ffmpeg for video. Nothing is written to disk.
Meant to be upstreamed into RPPS.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3") # BLAS threads for the equalizer's normal equations (they release the GIL); set before numpy is imported
import argparse
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import atsc_live as live  # noqa: E402
import atsc_rx  # noqa: E402
import radio_io  # noqa: E402
import ts_info  # noqa: E402

FS = live.FS
BLOCK = 1 << 20


def pids_of(ts):
    return ((ts[:, 1].astype(np.int32) & 0x1F) << 8) | ts[:, 2]


class Psi:
    """Programs and names from the packets of the tables only (PAT, PMT, PSIP): kept for a few seconds and parsed now and then (parsing is a Python loop, so it must not see the whole stream)"""
    def __init__(self):
        self.chunks = deque(); self.pmts = set(); self.table = None; self.next = 0.0

    def add(self, ts, err, pid):
        m = (err >= 0) & ((pid == 0) | (pid == 0x1FFB) | np.isin(pid, list(self.pmts)))
        if m.any():
            self.chunks.append((time.time(), ts[m]))
        while self.chunks and time.time() - self.chunks[0][0] > 8:
            self.chunks.popleft()

    def refresh(self, force=False):
        if (not force and time.time() < self.next) or not self.chunks:
            return self.table
        self.next = time.time() + 1.5
        t = live.parse_psi(np.concatenate([c for _, c in self.chunks]))
        self.table = live.merge_psi(self.table, t)
        self.pmts = {p["pmt"] for p in self.table["programs"].values()}
        return self.table


class Fmp4:
    """ffmpeg: MPEG-2 / AC-3 transport stream in, H.264 / AAC fragmented MP4 out, split into the init segment (ftyp + moov) and fragments (moof + mdat) for any number of browsers"""
    def __init__(self, program_title, height=720, on_log=None, dump=None):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-fflags", "+nobuffer", "-flags", "low_delay", "-probesize", "300000", "-analyzeduration", "500000", "-f", "mpegts", "-i", "pipe:0",
               "-map", "0:v:0", "-map", "0:a:0?", "-vf", f"yadif=0:-1:0,scale=-2:'min(ih,{height})',format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-profile:v", "baseline", "-level", "3.1",
               "-g", "30", "-keyint_min", "30", "-sc_threshold", "0", "-b:v", "2500k", "-maxrate", "3000k", "-bufsize", "3000k", "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "48000", "-af", "aresample=async=1:first_pts=0",
               "-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof", "pipe:1"]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.init = None; self.clients = []; self.lock = threading.Lock(); self.q = queue.Queue(maxsize=400); self.alive = True; self.fragments = 0; self.log = deque(maxlen=20); self.bytes_in = 0; self.on_log = on_log; self.seen = {}; self.shed = 0; self.shed_logged = 0; self.dump = open(dump, "wb") if dump else None # dump: the exact bytes fed to ffmpeg, for debugging
        for f in (self._feed, self._read, self._err):
            threading.Thread(target=f, daemon=True).start()

    def write(self, data):
        try:
            self.q.put_nowait(data)
        except queue.Full:
            try:
                self.q.get_nowait(); self.q.put_nowait(data) # a stalled ffmpeg loses the oldest packets rather than stalling the receiver
                self.shed += 1
                if self.on_log and self.shed - self.shed_logged >= 50:
                    self.on_log("error", f"ffmpeg input queue full: {self.shed} chunks of packets thrown away so far (ffmpeg is slower than the stream)"); self.shed_logged = self.shed
            except (queue.Empty, queue.Full):
                pass

    def _feed(self):
        while self.alive:
            try:
                d = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.p.stdin.write(d); self.bytes_in += len(d)
                if self.dump:
                    self.dump.write(d); self.dump.flush()
            except (BrokenPipeError, OSError, ValueError):
                self.alive = False

    def _err(self):
        for line in self.p.stderr:
            line = line.decode("utf-8", "replace").strip()[:200]
            self.log.append(line)
            if self.on_log and line:
                kind = re.sub(r"0x[0-9a-f]+|\d+", "#", line) # the same complaint at another place counts as the same
                n, t = self.seen.get(kind, (0, 0.0))
                now = time.time()
                if now - t > 5.0: # one line per kind per 5 s, with how many came since
                    self.on_log("ffmpeg", line + (f"  (+{n} similar)" if n else ""))
                    self.seen[kind] = (0, now)
                else:
                    self.seen[kind] = (n + 1, t)

    def _read(self):
        buf = bytearray(); init = bytearray(); frag = bytearray()
        while self.alive:
            d = self.p.stdout.read(65536)
            if not d:
                break
            buf += d
            while len(buf) >= 8:
                size = int.from_bytes(buf[:4], "big"); kind = bytes(buf[4:8])
                if size < 8 or len(buf) < size:
                    break
                box = bytes(buf[:size]); del buf[:size]
                if self.init is None:
                    init += box
                    if kind == b"moov":
                        self.init = bytes(init)
                elif kind == b"moof":
                    frag = bytearray(box)
                else: # mdat completes the fragment
                    frag += box; self.fragments += 1; data = bytes(frag); frag = bytearray()
                    with self.lock:
                        for q in list(self.clients):
                            try:
                                q.put_nowait(data)
                            except queue.Full:
                                self.clients.remove(q)
        self.alive = False

    def subscribe(self):
        q = queue.Queue(maxsize=100)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def close(self):
        self.alive = False
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.terminate(); self.p.wait(timeout=3)
        except Exception:
            self.p.kill()


class Monitor:
    def __init__(self, radio, channels, probe_every=120.0, height=720, dump_ts=None, dump_dips=True):
        self.radio, self.channels, self.probe_every, self.height, self.dump_ts, self.dump_dips = radio, channels, probe_every, height, dump_ts, dump_dips
        self.stations = {ch: live.Station(ch) for ch in channels}
        self.events = deque(maxlen=200); self.lock = threading.RLock()
        self.mode = "scan"; self.current = None; self.sweeps = 0; self.stop_flag = threading.Event(); self.rescan = threading.Event(); self.rescan.set()
        self.probes = ThreadPoolExecutor(2); self.watch = None; self.hist = deque(maxlen=240)
        self.thread = threading.Thread(target=self._sweep_loop, daemon=True); self.thread.start()

    def log(self, kind, text):
        self.events.append((time.time(), kind, text))

    # ------------------------------------------------------------------ scan
    def _sweep_loop(self):
        while not self.stop_flag.is_set():
            try:
                if self.mode != "scan":
                    time.sleep(0.2); continue
                if self.sweeps and self.sweeps % 8 == 0 and not self.rescan.is_set():
                    self.rescan.set()
                full = self.rescan.is_set(); self.rescan.clear()
                todo = self.channels if full else [c for c in self.channels if self.stations[c].is_station]
                for ch in todo:
                    if self.stop_flag.is_set() or self.mode != "scan":
                        break
                    st = self.stations[ch]
                    with self.radio.lock:
                        if self.mode != "scan":
                            break
                        self.current = ch
                        st.pilot, st.clip = live.pilot_level(self.radio, ch, st.gain); st.seen = time.time()
                        if st.is_station and not st.probing and time.time() - st.probe_t > self.probe_every:
                            self._probe(st)
                self.current = None; self.sweeps += 1
                if not any(s.is_station for s in self.stations.values()):
                    self.rescan.set()
                time.sleep(0.1)
            except Exception as e:
                self.current = None
                if self.stop_flag.is_set():
                    return
                self.log("error", f"sweep: {type(e).__name__}: {str(e)[:100]}"); time.sleep(1.0)

    def _probe(self, st):
        st.gain = st.gain or live.pick_gain(self.radio, st.ch, lambda t: self.log("gain", t))
        cf = live.centre(st.ch)
        self.radio.tune(cf, st.gain)
        x = self.radio.grab(2.6).astype(np.complex64)
        if radio_io.clipped(x) > 2e-4:
            st.gain = None
        st.probing = True
        self.probes.submit(self._probe_run, st, x, cf)

    def _probe_run(self, st, x, cf):
        try:
            if self.mode != "scan":
                return # the radio is being watched: the CPU is for the receiver
            ts, err, stats = atsc_rx.decode_array(x, cf, st.lower*1e6)
            st.locked = stats["state"] == "locked" and len(ts) > 0 and (stats.get("snr") or 0) >= atsc_rx.MIN_LOCK_SNR
            st.snr, st.ppm = stats.get("snr"), stats.get("clock_ppm")
            st.rs_ok = float(np.mean(err >= 0)) if len(err) else None
            if len(ts):
                psi = live.parse_psi(ts[err >= 0])
                if psi["programs"] or psi["vct"]:
                    st.psi = live.merge_psi(st.psi, psi)
            st.note = ""
            self.log("station", f"channel {st.ch}: " + (f"locked, SNR {st.snr:.1f} dB, Reed-Solomon ok {100*(st.rs_ok or 0):.0f}%, {len(st.psi['programs']) if st.psi else 0} programs" if st.locked else f"pilot {st.pilot:.0f} dB but no lock ({stats['state']})"))
        except Exception as e:
            st.note = f"{type(e).__name__}: {e}"[:80]; self.log("error", f"probe of channel {st.ch}: {st.note}")
        finally:
            st.probing = False; st.probe_t = time.time()

    # ------------------------------------------------------------------ watch
    def start_watch(self, ch, program=None):
        with self.lock:
            if self.watch:
                self.stop_watch()
            st = self.stations.get(ch)
            if st is None:
                raise ValueError(f"channel {ch} is not in the sweep")
            self.mode = "watch"
        self.watch = Watch(self, ch, program, st)
        self.watch.start()

    def stop_watch(self):
        with self.lock:
            w, self.watch = self.watch, None
        if w:
            w.stop()
        self.mode = "scan"

    # ------------------------------------------------------------------ for the page
    def state(self):
        now = time.time()
        sts = []
        for st in sorted(self.stations.values(), key=lambda s: s.ch):
            if not (st.is_station or st.probing):
                continue
            vcs = []
            if st.psi:
                names = {p: (mm, nm) for mm, (nm, p, _) in st.psi["vct"].items()}
                for n, p in sorted(st.psi["programs"].items()):
                    mm, nm = names.get(n, (None, ""))
                    vcs.append(dict(program=n, vc=f"{mm[0]}.{mm[1]}" if mm else None, name=nm, streams=[live.ts_info.STREAM.get(t, f"{t:#x}") for t, _ in p["streams"]]))
                vcs.sort(key=lambda v: (v["vc"] is None, [int(a) for a in v["vc"].split(".")] if v["vc"] else [v["program"]]))
            sts.append(dict(ch=st.ch, lower=st.lower, pilot=None if st.pilot is None else round(st.pilot, 1), status=st.status, snr=None if st.snr is None else round(st.snr, 1), rs_ok=st.rs_ok,
                            ppm=None if st.ppm is None else round(st.ppm, 2), programs=vcs, age=round(now - st.seen) if st.seen else None, probing=st.probing, note=st.note))
        return dict(radio=self.radio.name, mode=self.mode, current=self.current, sweeps=self.sweeps, stations=sts,
                    watch=self.watch.state() if self.watch else None, events=[dict(age=round(now - t, 1), kind=k, text=x) for t, k, x in list(self.events)[-60:]][::-1],
                    channels=len(self.channels), ffmpeg=bool(shutil.which("ffmpeg")))

    def close(self):
        self.stop_flag.set()
        self.stop_watch()
        self.probes.shutdown(wait=False, cancel_futures=True)
        self.thread.join(timeout=5)


class Watch:
    """The radio for one channel: capture thread -> Receiver (3 threads) -> program filter -> ffmpeg -> fragments for the browsers"""
    def __init__(self, mon, ch, program, st):
        self.mon, self.ch, self.want, self.st = mon, ch, program, st
        self.radio = mon.radio; self.rx = None; self.ff = None; self.psi = Psi(); self.keep = None; self.program = None; self.vc = None
        self.stop_flag = threading.Event(); self.t0 = time.time(); self.packets = 0; self.bytes = 0; self.bytes_win = deque(maxlen=20); self.spec = None
        self.clip = None; self.ring = deque(); self.ring_n = 0; self.dip_at = None; self.dumps = 0; self.lost_seen = 0; self.msg = "starting"; self.dropped0 = 0; self.hist = deque(maxlen=240); self.last_hist = 0.0; self.thread = None; self.out_packets = 0

    def start(self):
        self.thread = threading.Thread(target=self._capture, daemon=True); self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=8)
        if self.rx:
            self.rx.stop()
        if self.ff:
            self.ff.close()

    def _capture(self):
        mon = self.mon
        try:
            with self.radio.lock:
                gain = self.st.gain or live.pick_gain(self.radio, self.ch, lambda t: mon.log("gain", t))
                self.radio.tune(live.centre(self.ch), gain)
                self.rx = atsc_rx.Receiver(live.centre(self.ch), live.lower_edge(self.ch)*1e6, FS, on_ts=self._on_ts).start()
                self.dropped0 = self.radio.dropped; mon.log("watch", f"channel {self.ch}: receiver started (gain {gain:.2f})")
                n = 0
                while not self.stop_flag.is_set():
                    b = self.radio.get_samples()
                    if not len(b):
                        time.sleep(0.003); continue
                    b = np.asarray(b, np.complex64)
                    self.rx.push(b); n += 1
                    self._ring_step(b)
                    if n % 8 == 0:
                        self._spectrum(b)
        except Exception as e:
            self.msg = f"{type(e).__name__}: {e}"[:100]; mon.log("error", f"watch: {self.msg}")
        finally:
            if self.rx:
                self.rx.stop()
            self.mon.mode = "scan" if self.mon.watch is self else self.mon.mode

    # ---- raw samples around a dip of the Reed-Solomon yield, to decode again offline (decoder fault or signal?)
    RING_S, AFTER_S, DIP, MAX_DUMPS = 4.0, 2.0, 0.6, 3

    def _ring_step(self, b):
        if not self.mon.dump_dips:
            return
        self.ring.append(b); self.ring_n += len(b)
        while self.ring_n - len(self.ring[0]) > self.RING_S*FS:
            self.ring_n -= len(self.ring.popleft())
        now = time.time(); rs = self.rx.stats.get("rs_ok")
        if self.dip_at is None:
            if rs is not None and rs < self.DIP and now - self.t0 > 15 and self.dumps < self.MAX_DUMPS:
                self.dip_at = now
        elif now - self.dip_at >= self.AFTER_S:
            self.dip_at = None; self.dumps += 1
            data = np.concatenate(list(self.ring)); name = os.path.join(os.path.dirname(HERE), "..", "data", "atsc", f"dip_ch{self.ch}_{time.strftime('%H%M%S')}.cf64")
            name = os.path.normpath(name)
            def save():
                data.tofile(name); self.mon.log("watch", f"yield dip: {len(data)/FS:.1f} s of raw samples saved to {name} (centre {live.centre(self.ch)/1e6:.3f} MHz, lower edge {live.lower_edge(self.ch)} MHz, {FS/1e6:.0f} MS/s)")
            threading.Thread(target=save, daemon=True).start()

    def _spectrum(self, b):
        self.clip = max(self.clip*0.9, radio_io.clipped(b)) if self.clip is not None else radio_io.clipped(b) # fraction of samples at full scale (decaying peak)
        N = 1 << 16
        if len(b) < N:
            return
        p = np.abs(np.fft.fftshift(np.fft.fft(b[:N]*np.hanning(N))))**2
        k = 64; p = p[:N//k*k].reshape(-1, k).mean(1)
        db = 10*np.log10(p + 1e-12); f0 = (live.centre(self.ch) - FS/2)/1e6; step = FS/N*k/1e6
        self.spec = dict(f0=f0, step=step, db=[round(float(v), 1) for v in db], lower=live.lower_edge(self.ch), upper=live.lower_edge(self.ch) + 6)

    def _on_ts(self, ts, err):
        """Called by the decode thread every ~0.1 s of air with 1248 packets"""
        pid = pids_of(ts)
        good = err >= 0
        self.packets += len(ts); self.out_packets += int(good.sum())
        self.psi.add(ts, err, pid)
        if self.rx and self.rx.stats.get("lost", 0) > self.lost_seen: # say why the receiver let go
            self.lost_seen = self.rx.stats["lost"]; s = self.rx.stats
            self.mon.log("error", f"channel {self.ch}: lock lost (#{self.lost_seen}); last regions (field-sync corr, sync offset, SNR dB, lag s): {s.get('loss_trace')}; radio dropped blocks {self.radio.dropped - self.dropped0}, overruns {s.get('overruns', 0)}")
        now = time.time()
        if self.keep is None:
            tbl = self.psi.refresh()
            if tbl and tbl["programs"] and any(p["streams"] for p in tbl["programs"].values()):
                try:
                    self.program = live.resolve_program(tbl, self.want)
                    if tbl["programs"][self.program]["streams"]:
                        self.keep = np.array(sorted(live.program_pids(tbl, self.program)))
                        names = {p: mm for mm, (nm, p, _) in tbl["vct"].items()}
                        self.vc = names.get(self.program)
                        self.ff = Fmp4(f"ATSC {self.ch}", self.mon.height, on_log=self.mon.log, dump=self.mon.dump_ts)
                        self.mon.log("watch", f"channel {self.ch}: program {self.program}" + (f" ({self.vc[0]}.{self.vc[1]})" if self.vc else "") + ", ffmpeg started")
                        if self.st.psi is None:
                            self.st.psi = tbl
                except (SystemExit, ValueError) as e:
                    self.msg = str(e)
            return
        if self.ff is not None and self.ff.alive:
            sel = good & np.isin(pid, self.keep)
            if sel.any():
                data = ts[sel].tobytes(); self.bytes += len(data); self.ff.write(data)
        self.psi.refresh()
        self.bytes_win.append((now, self.out_packets))
        if now - self.last_hist > 0.5 and self.rx:
            self.last_hist = now; s = self.rx.stats
            self.hist.append(dict(t=round(now - self.t0, 1), snr=s.get("snr"), rs=s.get("rs_ok"), lag=s.get("lag")))

    def state(self):
        s = dict(self.rx.stats) if self.rx else {}
        s.pop("load", None)
        load = dict(self.rx.stats.get("load", {})) if self.rx else {}
        rate = None
        if len(self.bytes_win) >= 2:
            (t0, p0), (t1, p1) = self.bytes_win[0], self.bytes_win[-1]
            rate = (p1 - p0)*188*8/max(t1 - t0, 1e-3)/1e6
        ff = self.ff
        return dict(ch=self.ch, program=self.program, vc=(f"{self.vc[0]}.{self.vc[1]}" if self.vc else None), state=s.get("state", self.msg), snr=s.get("snr"), rs_ok=s.get("rs_ok"), fields=s.get("fields"), lost=s.get("lost"),
                    clock_ppm=s.get("clock_ppm"), pilot_db=s.get("pilot_db"), lag=s.get("lag"), load=load, packets=s.get("packets"), bad=s.get("bad"), overruns=s.get("overruns", 0), clip=self.clip, mbps=None if rate is None else round(rate, 2),
                    dropped=self.radio.dropped - self.dropped0, video=dict(ready=bool(ff and ff.init), fragments=ff.fragments if ff else 0, clients=len(ff.clients) if ff else 0, shed=ff.shed if ff else 0, alive=bool(ff and ff.alive),
                    error=(list(ff.log)[-1] if ff and ff.log else None)), error=s.get("error"), spectrum=self.spec, history=list(self.hist)[-120:], uptime=round(time.time() - self.t0, 1))


# ---- web server --------------------------------------------------------------------------------------------------------------------------------------------------------------------

def make_handler(mon, page):
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
            if self.path == "/":
                self.reply(page(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.reply(json.dumps(mon.state()).encode(), "application/json")
            elif self.path == "/video":
                w = mon.watch
                if not w or not w.ff or not w.ff.init:
                    self.send_error(404, "not watching"); return
                ff = w.ff; q = ff.subscribe()
                self.send_response(200); self.send_header("Content-Type", "video/mp4"); self.send_header("Transfer-Encoding", "chunked"); self.send_header("Cache-Control", "no-store"); self.send_header("Connection", "close"); self.end_headers()
                try:
                    d = ff.init
                    self.wfile.write(b"%x\r\n" % len(d) + d + b"\r\n"); self.wfile.flush()
                    while ff.alive and mon.watch is w:
                        try:
                            d = q.get(timeout=1.0)
                        except queue.Empty:
                            continue
                        self.wfile.write(b"%x\r\n" % len(d) + d + b"\r\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    ff.unsubscribe(q); self.close_connection = True
            else:
                self.send_error(404)

        def do_POST(self):
            if not self.local():
                return
            if "application/json" not in self.headers.get("Content-Type", ""):
                self.send_error(400); return
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/watch":
                    mon.start_watch(int(d["ch"]), d.get("program"))
                elif self.path == "/api/log": # errors of the page itself, kept in the event list
                    mon.log("browser", str(d.get("text", ""))[:300])
                elif self.path == "/api/stop":
                    mon.stop_watch()
                elif self.path == "/api/rescan":
                    mon.rescan.set()
                    for s in mon.stations.values():
                        s.probe_t = 0.0
                else:
                    self.send_error(404); return
            except (ValueError, TypeError, KeyError) as e:
                self.reply(json.dumps(dict(error=str(e))).encode(), "application/json", 400); return
            self.reply(b"{}", "application/json")
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    live.radio_io.add_radio_args(ap)
    ap.add_argument("--replay", metavar="CF64", help="a capture file as the air instead of a radio (complex64, 20 MS/s)")
    ap.add_argument("--replay-channel", type=int, default=35); ap.add_argument("--replay-cf", type=float, default=604.0, help="the capture's centre [MHz]"); ap.add_argument("--replay-speed", type=float, default=1.0)
    ap.add_argument("--channels", type=int, nargs="+", help="channels to sweep (default: VHF 7-13 and UHF 14-36)")
    ap.add_argument("--height", type=int, default=720, help="video height after the re-encode")
    ap.add_argument("--dump-ts", metavar="FILE", help="write the transport packets fed to ffmpeg to FILE (debugging; overwritten at each watch)")
    ap.add_argument("--no-dip-dump", action="store_true", help="do not save raw samples (data/atsc/dip_*.cf64, 4 s before and 2 s after a dip of the Reed-Solomon yield; at most 3 per watch)")
    ap.add_argument("--probe-every", type=float, default=120.0)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8035)
    a = ap.parse_args(argv)
    radio = live.ReplayRadio(a.replay, a.replay_channel, a.replay_cf, a.replay_speed) if a.replay else live.Radio(a.remote, a.radio, a.channels, a.serial, a.label, "RPPS atsc_monitor")
    mon = Monitor(radio, a.channels or live.CHANNELS, a.probe_every, a.height, a.dump_ts, not a.no_dip_dump)
    page = lambda: open(os.path.join(HERE, "atsc_monitor.html"), "rb").read() # read per request: a reload picks up edits
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(mon, page)); srv.allow_remote = a.host not in ("127.0.0.1", "localhost", "::1"); srv.daemon_threads = True
    print(f"atsc monitor: http://{a.host}:{a.port}/  ({radio.name}, channels {len(mon.channels)})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        mon.close(); radio.close()


if __name__ == "__main__":
    main()
