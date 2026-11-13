"""Live P25 monitor (phase 1 voice and phase 2 signalling and voice): one HackRF (or a recording played back in real time) -> control-channel decoding, voice channels followed from the grants, unencrypted voice to the browser

    python projects/p25_dmr/p25_monitor.py --cf 409.0 --fs 8 --control 407.6125 --gain 0.7            # then open http://127.0.0.1:8025/
    python projects/p25_dmr/p25_monitor.py --cf 409.0 --fs 4 --control 407.6125 --remote tcp://HOST:5555    # the HackRF of a pysdrlib server
    python projects/p25_dmr/p25_monitor.py --file data/survey/w409.0.cf64 --cf 409.0 --fs 8 --control 407.6125   # replay a recording as if live

Pipeline (all streaming, nothing is read twice): the radio's samples go through one FFT channelizer (overlap-save; the channel is cut out of the spectrum, turned into 48 kS/s baseband with
phase continuity kept across blocks), then per channel the 7 kHz low-pass, FM discriminator and the same sync / NID / TSBK / LDU code as p25_sync.py, p25_voice.py and p25_audio.py. The control
channel gives the channel tables and voice grants (talkgroup -> frequency, with the service options' encrypted flag); a decoder is started on every granted voice frequency inside the capture, so a call
is caught from its first frame. Each call's LDU2 encryption algorithm ID is read (p25_audio's rule: unencrypted only when it says so) and only unencrypted calls reach the IMBE decoder (imbe.py);
encrypted calls are listed and never decoded, no key is used or attempted. Audio is 8 kS/s 16-bit PCM streamed over HTTP to the page, kept in memory only, never written to disk.
The web server listens on 127.0.0.1 only (--host to change it); requests must carry a local Host header. Phase 2 (TDMA) carriers are followed from the grants too: p25p2.py decodes both slots (calls, talkgroup, unit, encryption algorithm) and p25p2_voice.py the half-rate voice of unencrypted calls. Limits: the capture span limits which
voice channels are reachable, the 0x02 grant update is ignored (its layout is unverified), and the half-rate voice decoder has not been checked on speech (no clear phase 2 call recorded yet). `load` on the page is processing time per second of signal: above 1.0 the monitor drops samples (`overruns`).
Meant to be upstreamed into RPPS.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import Counter, deque
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from rpps.signal.filters import FIR, low_pass

HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import imbe  # noqa: E402
import p25_audio  # noqa: E402
import p25p2  # noqa: E402
import p25p2_voice  # noqa: E402
import p25_sync  # noqa: E402
import p25_voice  # noqa: E402

RATE = 48000


class Channelizer:
    """Overlap-save FFT channelizer: one big FFT of the capture stream shared by all channels; each channel takes its bins, windows them and inverse transforms to 48 kS/s baseband.
    The input rate need not be a multiple of 48 kHz (a rational ratio p/q is used); each block's phase is advanced by exp(-j 2 pi kc n_b / N) so the baseband is continuous across blocks."""
    def __init__(self, fs):
        fr = Fraction(int(round(fs)), RATE); self.p, self.q = fr.numerator, fr.denominator; self.fs = fs
        kk = max(4, 131072//self.p); self.N = self.p*kk; self.M = self.q*kk
        ov = -(-96//self.q); self.Mov = self.q*ov; self.Nov = self.p*ov; self.hop = self.N - self.Nov
        assert self.Mov < self.M//2, "capture rate not usable"
        self.rel = np.arange(-(self.M//2), self.M - self.M//2); f = np.abs(self.rel)*fs/self.N
        self.W = np.where(f <= 10e3, 1.0, np.where(f >= 17e3, 0.0, np.cos(0.5*np.pi*(f - 10e3)/7e3)**2)).astype(np.float32)
        self.buf = np.zeros(0, np.complex64); self.n0 = 0

    def bin(self, off_hz):
        return int(round(off_hz*self.N/self.fs))

    def feed(self, x):
        self.buf = np.concatenate((self.buf, x))
        while len(self.buf) >= self.N:
            yield self.n0, np.fft.fft(self.buf[:self.N])
            self.buf = self.buf[self.hop:]; self.n0 += self.hop

    def extract(self, X, n_b, kc):
        y = np.fft.ifft(np.fft.ifftshift(X[(kc + self.rel) % self.N]*self.W))*(self.M/self.N)
        return (y[self.Mov:]*np.exp(-2j*np.pi*((kc*n_b) % self.N)/self.N)).astype(np.complex64)


class Chan:
    """One channel: baseband -> low-pass -> discriminator -> rolling buffer; every 0.2 s the buffer is searched for frame syncs and each new frame handed to `on_frame`"""
    KEEP = 3*RATE

    def __init__(self, freq, kc, on_frame, role="voice"):
        self.freq = freq; self.kc = kc; self.on_frame = on_frame; self.role = role
        self.taps = low_pass(121, 7e3, fs=RATE).filt; self.zi = None; self.last = np.complex64(0); self.zb = None
        self.fi = np.zeros(0, np.float32); self.pw = np.zeros(0, np.float32); self.base = 0; self.since = 0
        self.done = []; self.level = -120.0; self.active = False; self.frames = 0; self.nac = None; self.last_frame = 0.0; self.born = time.monotonic()

    def push(self, y):
        z, self.zi = FIR.stream(y, self.taps, self.zi)
        zz = np.concatenate(([self.last], z)); self.last = z[-1]
        d = np.angle(zz[1:]*np.conj(zz[:-1]))*(RATE/(2*np.pi))
        f, self.zb = FIR.stream(d, np.ones(5)/5, self.zb)
        self.fi = np.concatenate((self.fi, f.astype(np.float32))); self.pw = np.concatenate((self.pw, (np.abs(z)**2).astype(np.float32)))
        if len(self.fi) > self.KEEP:
            cut = len(self.fi) - self.KEEP; self.fi = self.fi[cut:]; self.pw = self.pw[cut:]; self.base += cut
        self.since += len(y)
        if self.since >= RATE//5:
            self.since = 0; self.tick()

    def tick(self):
        n = min(len(self.fi), 2*RATE)
        if n < RATE//2:
            return
        W = self.fi[-n:]; P = self.pw[-n:]; base = self.base + len(self.fi) - n
        p90, p10 = np.percentile(P, 90), np.percentile(P, 10)
        self.level = float(10*np.log10(p90 + 1e-20)); self.active = bool(p90 > 8*p10)
        m = P > 0.1*p90
        if m.sum() < RATE//10:
            return
        f = W - np.median(W[m]); am = np.abs(f)*m
        ph = int(np.argmax([am[p::10].mean() for p in range(10)]))
        sn = f[ph::10]
        hits, _ = p25_sync.sync_hits(sn)
        now = time.monotonic()
        for i in hits:
            pos = i*10 + ph
            if pos + 8800 > len(W):
                continue
            a = base + pos
            if any(abs(a - q) < 40 for q in self.done):
                continue
            self.done.append(a)
            fr = p25_sync.refine((f, ph), i, n=864)
            if len(fr) < 864:
                continue
            v = p25_sync.nid(fr, 0)
            if not v:
                continue
            fr = fr/(np.mean(fr[:24]*p25_sync.SYNC)/9.0)
            self.frames += 1; self.nac = v[0]; self.last_frame = now
            self.on_frame(self, v[1], v[0], fr)
        self.done = [q for q in self.done if q > self.base]


class Call:
    """The state of one voice call on a channel (a run of LDUs without a gap)"""
    def __init__(self, tg):
        self.rec = None; self.tg = tg; self.algs = Counter(); self.pending = []; self.flag_enc = False; self.dec = None; self.last = time.monotonic(); self.state = "unknown"; self.env = 4000.0; self.audio = 0.0


OPS = {0x00: "voice grant", 0x02: "grant update", 0x03: "unit grant", 0x28: "affiliation", 0x2C: "registration", 0x20: "ack", 0x3D: "channel table", 0x34: "channel table (VU)", 0x33: "channel table (TDMA)",
       0x3B: "network status", 0x3A: "site status", 0x39: "secondary control channel", 0x3C: "adjacent site", 0x38: "status update", 0x24: "unit status"}

class Monitor:
    def __init__(self, fs, cf, ppm, control, gate_phase2=True):
        self.fs = fs; self.cf = cf; self.ppm = ppm; self.chz = Channelizer(fs)
        self.lock = threading.RLock(); self.chans = {}; self.calls = {}; self.tsbk = p25_sync.Tsbk(); self.tgs = {}; self.freq_tg = {}
        self.sel = {"mode": "off", "tg": None}; self.clients = []; self.scan_lock = (None, 0.0)
        self.stats = dict(load=0.0, overruns=0, samples=0, started=time.time(), blocks=0); self.out_of_span = set()
        self.control = control; self.events = deque(maxlen=300); self.ops = Counter(); self.blk = Counter(); self.recent = deque(maxlen=10); self.radio = {}; self.hk = 0.0
        self.heard = None; self.p2 = {}; self.session = Counter(); self.psd = None; self.psd_n = 0; self.pool = max(1, self.chz.N//1024); self.psd_start = (self.chz.N - 1024*self.pool)//2
        for f in control:
            self.add_channel(f, "control")

    def log(self, kind, text):
        self.events.append((time.time(), kind, text))

    def in_span(self, freq):
        return abs(freq*1e6*(1 + self.ppm*1e-6) - self.cf) < self.fs*0.42

    def add_channel(self, freq, role="voice", p2=False):
        key = round(freq, 5)
        with self.lock:
            if key in self.chans:
                return self.chans[key]
            if not self.in_span(freq):
                self.out_of_span.add(key); return None
            kc = self.chz.bin(freq*1e6*(1 + self.ppm*1e-6) - self.cf)
            if p2:
                self.chans[key] = p25p2.ChanP2(freq, kc, self.p2_params, self.on_p2)
            else:
                self.chans[key] = Chan(freq, kc, self.on_frame, role)
            return self.chans[key]

    # -- data from the DSP thread
    def process(self, blk):
        t0 = time.perf_counter()
        for n_b, X in self.chz.feed(blk):
            self.spectrum_update(X)
            with self.lock:
                chans = list(self.chans.values())
            for ch in chans:
                ch.push(self.chz.extract(X, n_b, ch.kc))
        self.housekeeping()
        clip = float(np.mean((np.abs(blk.real) > 0.98) | (np.abs(blk.imag) > 0.98))); rms = float(np.mean(blk.real**2 + blk.imag**2))
        self.stats["clip"] = 0.9*self.stats.get("clip", 0.0) + 0.1*clip; self.stats["rms_db"] = 10*np.log10(0.9*10**(self.stats.get("rms_db", -40.0)/10) + 0.1*rms + 1e-12)
        dt = time.perf_counter() - t0; self.stats["samples"] += len(blk)
        self.stats["load"] = 0.9*self.stats["load"] + 0.1*dt/(len(blk)/self.fs)

    def spectrum_update(self, X):
        self.psd_n += 1
        if self.psd_n % 3:
            return
        p = np.fft.fftshift(np.abs(X)**2)[self.psd_start:self.psd_start + 1024*self.pool].reshape(1024, self.pool).mean(1)
        self.psd = p if self.psd is None else 0.8*self.psd + 0.2*p

    def spectrum(self):
        if self.psd is None:
            return None
        c = self.chz; step = self.pool*c.fs/c.N
        f0 = self.cf + (self.psd_start + self.pool/2 - c.N//2)*c.fs/c.N
        return dict(f0=f0/1e6, step=step/1e6, db=[round(float(v), 1) for v in 10*np.log10(self.psd + 1e-9)])

    def housekeeping(self):
        mono = time.monotonic()
        if mono - self.hk < 0.5:
            return
        self.hk = mono
        for freq, call in list(self.calls.items()):
            if mono - call.last > 1.0:
                self.end_call(freq)

    def end_call(self, freq):
        call = self.calls.pop(freq, None)
        if call is None:
            return
        if call.state == "clear":
            self.log("call", f"call ended on {freq:.4f} MHz, talkgroup {call.tg if call.tg is not None else '?'}: {call.audio:.1f} s of audio")
        elif call.state == "encrypted":
            self.log("enc", f"call ended on {freq:.4f} MHz, talkgroup {call.tg if call.tg is not None else '?'}: encrypted, not decoded")
        else:
            self.log("call", f"call ended on {freq:.4f} MHz before its encryption state was known")

    def on_frame(self, ch, duid, nac, fr):
        if duid == 0x7:
            for by, good in p25_sync.tsbk_blocks(p25_sync.frame_bits(fr, 0)):
                self.blk["ok" if good else "bad"] += 1
                if good:
                    self.control_block(by)
        elif duid in (0x5, 0xA):
            self.voice(ch, duid, fr)
        elif duid in (0x3, 0xF):
            self.end_call(ch.freq)

    def control_block(self, by):
        with self.lock:
            self.tsbk.add(by); self.ops[OPS.get(by[0] & 0x3f, f"op {by[0] & 0x3f:#x}") if by[1] == 0 else f"manufacturer {by[1]:#x}"] += 1
            if by[1] == 0 and (by[0] & 0x3f) == 0x00:
                v = int.from_bytes(by[:10], "big") << 16; opts = (v >> 72) & 0xff; chn = (v >> 56) & 0xffff; tg = (v >> 40) & 0xffff
                freq = self.tsbk.freq(chn); slot = self.tsbk.slot(chn); t = self.tgs.setdefault(tg, dict(tg=tg, freq=None, slot=None, opts=0, grants=0, last_grant=0, last_audio=0, audio_s=0.0, calls=0, alg=None))
                fresh = time.time() - t["last_grant"] > 5 or t["freq"] != freq
                t.update(freq=freq, slot=slot, opts=opts, last_grant=time.time()); t["grants"] += 1
                if fresh:
                    where = f"{freq:.4f} MHz" if freq else f"channel {chn:#x} (table not seen yet)"
                    note = "phase 2, following" if slot is not None else ("encrypted flag set, not followed" if opts & 0x40 else ("clear flag, following" if freq and self.in_span(freq) else "outside the capture span"))
                    self.log("grant", f"control channel: talkgroup {tg} granted {where} ({note})")
                if freq is not None and slot is not None:
                    self.add_channel(freq, "voice", p2=True) # phase 2: both slots of the carrier are followed (signalling and encryption state)
                if freq is not None and slot is None:
                    self.freq_tg[round(freq, 5)] = tg
                    if not opts & 0x40: # a talkgroup flagged encrypted is listed from its grants and never followed
                        self.add_channel(freq)

    def p2_params(self):
        s = self.tsbk.sys; nac = None
        for c in self.chans.values():
            if c.role == "control" and c.nac is not None:
                nac = c.nac
        if not s or "wacn" not in s or nac is None:
            return None
        return (s["wacn"], s["sysid"], nac)

    def on_p2(self, ch, slots):
        """Slots decoded on a phase 2 carrier: track the calls of its two logical channels"""
        now = time.monotonic()
        for sl in slots:
            if sl.lch is None:
                continue
            key = (round(ch.freq, 5), sl.lch)
            st = self.p2.setdefault(key, dict(tg=None, source=None, alg=None, kid=None, last=0.0, start=0.0, voice=0, active=False))
            pdu = sl.pdu
            if pdu and pdu["ok"]:
                if pdu["name"] == "MAC_PTT":
                    st.update(tg=pdu["group"], source=pdu["source"], alg=pdu["algid"], kid=pdu["kid"], last=now)
                    self.p2_call(ch, sl.lch, st, "PTT")
                elif pdu["name"] == "MAC_END_PTT":
                    if st["active"]:
                        self.log("call" if st["alg"] == 0x80 else "enc", f"phase 2 {ch.freq:.4f} MHz slot {sl.lch}: talkgroup {st['tg']} call ended after {now - st['start']:.0f} s ({st['voice']} voice bursts)")
                    st.update(active=False, voice=0)
                else:
                    for m in pdu.get("msgs", []):
                        if m["type"] == "group voice user":
                            st.update(tg=m["group"], source=m["source"] or st["source"], last=now)
                            if m["group"] and st["tg"] is not None:
                                self.p2_call(ch, sl.lch, st, "user")
            if sl.ess_b is not None or sl.ess_a is not None or sl.voice:
                st["last"] = now; st["voice"] += 1
            if getattr(sl, "ess", None):
                st["alg"] = sl.ess["algid"]; st["kid"] = sl.ess["kid"]
            if sl.voice and st["alg"] == 0x80 and st["tg"] is not None:
                self.p2_audio(ch, st, sl)
            self.p2_tg(ch, st)

    def p2_audio(self, ch, st, sl):
        """A burst of an unencrypted phase 2 call: the half-rate vocoder (p25p2_voice.py) turns its voice frames into audio"""
        call = st.get("call")
        if call is None:
            call = st["call"] = Call(st["tg"]); call.state = "clear"; call.dec = p25p2_voice.HalfRate()
            call.rec = dict(tg=st["tg"], freq=ch.freq, t=time.time(), buf=bytearray()); self.recent.appendleft(call.rec); self.session["clear"] += 1
            with self.lock:
                t = self.tgs.get(st["tg"])
                if t is not None:
                    t["calls"] += 1
            self.log("call", f"phase 2 {ch.freq:.4f} MHz talkgroup {st['tg']}: unencrypted, decoding with the half-rate vocoder")
        call.tg = st["tg"]; call.last = time.monotonic()
        pcm = np.concatenate([call.dec.decode(c) for c in sl.voice])
        self.emit(ch.freq, call, pcm)

    def p2_call(self, ch, lch, st, why):
        if not st["active"]:
            st["call"] = None
            st["active"] = True; st["start"] = time.monotonic()
            alg = p25_voice.ALGO.get(st["alg"], hex(st["alg"])) if st["alg"] is not None else "algorithm not read yet"
            self.log("enc" if st["alg"] not in (None, 0x80) else "call", f"phase 2 {ch.freq:.4f} MHz slot {lch}: talkgroup {st['tg']}, unit {st['source']}: {alg}")

    def p2_tg(self, ch, st):
        with self.lock:
            t = self.tgs.get(st["tg"]) if st["tg"] is not None else None
            if t is not None and st["alg"] is not None:
                t["alg"] = st["alg"]

    def voice(self, ch, duid, fr):
        now = time.monotonic(); call = self.calls.get(ch.freq)
        if call is None or now - call.last > 0.6:
            if call is not None:
                self.end_call(ch.freq)
            call = self.calls[ch.freq] = Call(self.freq_tg.get(round(ch.freq, 5)))
            self.log("call", f"voice frames on {ch.freq:.4f} MHz (talkgroup {call.tg if call.tg is not None else 'not read yet'})")
        call.last = now
        hb = p25_voice.hexbits(p25_voice.frame_bits_full(fr))
        frames = p25_audio.voice_frames(fr)
        if duid == 0x5:
            lcf, mfid, svc, tg, src = p25_voice.link_control(hb)
            if lcf == 0 and mfid == 0:
                call.tg = tg; call.flag_enc = call.flag_enc or bool(svc & 0x40)
        else:
            call.algs[p25_voice.ess(hb)[1]] += 1
        with self.lock:
            t = self.tgs.get(call.tg) if call.tg is not None else None
            if t is not None and t["opts"] & 0x40:
                call.flag_enc = True
        n80 = call.algs.get(0x80, 0); others = [n for a, n in call.algs.items() if a != 0x80]
        was = call.state
        if call.state == "unknown":
            if others and max(others) >= 2 or (others and not n80):
                call.state = "encrypted"
            elif n80 >= 1 and not call.flag_enc:
                call.state = "clear"
                call.dec = imbe.Decoder()
                if t is not None:
                    t["calls"] += 1
            elif call.flag_enc and n80 + sum(others) >= 1:
                call.state = "encrypted"
        if was == "unknown" and call.state != "unknown":
            names = ", ".join(p25_voice.ALGO.get(a, hex(a)) for a in call.algs) or "service flag"
            self.log("call" if call.state == "clear" else "enc", f"{ch.freq:.4f} MHz talkgroup {call.tg if call.tg is not None else '?'}: {names} -> " + ("decoding" if call.state == "clear" else "encrypted, not decoded"))
            self.session["clear" if call.state == "clear" else "encrypted"] += 1
            if call.state == "clear":
                call.rec = dict(tg=call.tg, freq=ch.freq, t=time.time(), buf=bytearray()); self.recent.appendleft(call.rec)
        with self.lock:
            if call.tg is not None and call.tg in self.tgs and call.algs:
                alg = call.algs.most_common(1)[0][0]; self.tgs[call.tg]["alg"] = alg
        if call.state == "encrypted":
            call.pending = []; return
        call.pending.append(frames)
        if call.state != "clear":
            call.pending = call.pending[-3:]; return
        pcm = np.concatenate([call.dec.decode(b) for fs_ in call.pending for b in fs_]); call.pending = []
        self.emit(ch.freq, call, pcm)

    def emit(self, freq, call, pcm):
        peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
        call.env = max(call.env*0.97, peak, 1500.0)
        out = np.clip(pcm*min(10.0, 18000.0/call.env), -32000, 32000).astype(np.int16).tobytes()
        call.audio += len(pcm)/8000
        if call.rec is not None and len(call.rec["buf"]) < 8000*2*180:
            call.rec["buf"] += out
            call.rec["tg"] = call.tg
        with self.lock:
            t = self.tgs.get(call.tg)
            if t is not None:
                t["last_audio"] = time.time(); t["audio_s"] += len(pcm)/8000
            sel = self.sel; ok = sel["mode"] == "scan" or (sel["mode"] == "tg" and sel["tg"] == call.tg)
            if ok and sel["mode"] == "scan":
                lt, lat = self.scan_lock; now = time.monotonic()
                if lt not in (None, call.tg) and now - lat < 1.0:
                    ok = False
                else:
                    self.scan_lock = (call.tg, now)
            if ok:
                self.heard = dict(tg=call.tg, freq=freq, t=time.time(), since=(self.heard["since"] if self.heard and self.heard["freq"] == freq and time.time() - self.heard["t"] < 2 else time.time()))
                for q in list(self.clients):
                    try:
                        q.put_nowait(out)
                    except queue.Full:
                        pass

    # -- data for the web page
    def state(self):
        now = time.time(); mono = time.monotonic()
        with self.lock:
            tgs = []
            for t in self.tgs.values():
                enc = bool(t["opts"] & 0x40); alg = t["alg"]
                status = ("phase2" if alg is None and not enc else ("clear" if alg == 0x80 else "encrypted")) if t["slot"] is not None else ("encrypted" if enc or (alg is not None and alg != 0x80) else ("clear" if alg == 0x80 else "unverified"))
                tgs.append(dict(tg=t["tg"], freq=t["freq"], status=status, enc_flag=enc, p2=t["slot"] is not None, alg=p25_voice.ALGO.get(alg, hex(alg)) if alg is not None else None, emergency=bool(t["opts"] & 0x80),
                                grants=t["grants"], grant_age=now - t["last_grant"], audio_age=(now - t["last_audio"]) if t["last_audio"] else None, audio_s=round(t["audio_s"], 1), calls=t["calls"]))
            tgs.sort(key=lambda t: (t["grant_age"]))
            chans = [dict(kind=getattr(c, "kind", "p1"), lch=[dict(lch=k[1], tg=v["tg"], source=v["source"], alg=(p25_voice.ALGO.get(v["alg"], hex(v["alg"])) if v["alg"] is not None else None),
                                                                      age=round(mono - v["last"], 1) if v["last"] else None, active=bool(v["active"] and mono - v["last"] < 2))
                                                 for k, v in self.p2.items() if k[0] == round(c.freq, 5)] if getattr(c, "kind", "p1") == "p2" else [],
                          freq=c.freq, role=c.role, level=round(c.level, 1), active=bool(c.active or (c.frames and mono - c.last_frame < 2)), frames=c.frames, nac=c.nac, age=(mono - c.last_frame) if c.frames else None,
                          call=bool(self.calls.get(c.freq) and mono - self.calls[c.freq].last < 0.6), tg=(self.calls[c.freq].tg if self.calls.get(c.freq) else None),
                          cstate=(self.calls[c.freq].state if self.calls.get(c.freq) and mono - self.calls[c.freq].last < 0.6 else None)) for c in self.chans.values()]
            s = self.tsbk.sys
            return dict(cf=self.cf/1e6, fs=self.fs/1e6, load=round(self.stats["load"], 2), overruns=self.stats["overruns"], uptime=round(now - self.stats["started"]),
                        system=dict(wacn=s.get("wacn"), sysid=s.get("sysid")) if s else None, sel=self.sel, tgs=tgs, channels=sorted(chans, key=lambda c: (c["role"] != "control", c["freq"])),
                        out_of_span=sorted(self.out_of_span), radio=self.radio, p2_audio=True, input=dict(clip=round(self.stats.get("clip", 0.0), 6), rms_db=round(float(self.stats.get("rms_db", -99)), 1)),
                        playing=(dict(tg=self.heard["tg"], freq=self.heard["freq"], for_s=round(now - self.heard["since"], 1)) if self.heard and now - self.heard["t"] < 1.5 and self.sel["mode"] != "off" else None),
                        scan=dict(watching=sum(1 for c in self.chans.values() if c.role == "voice"), clear=self.session["clear"], encrypted=self.session["encrypted"], since=round(now - self.stats["started"])), control=dict(ok=self.blk["ok"], bad=self.blk["bad"], ops=dict(self.ops.most_common(12))), clients=len(self.clients),
                        events=[dict(age=round(now - t, 1), kind=k, text=x) for t, k, x in list(self.events)[-60:]][::-1],
                        recent=[dict(i=i, tg=r["tg"], freq=r["freq"], age=round(now - r["t"]), seconds=round(len(r["buf"])/16000, 1)) for i, r in enumerate(self.recent) if len(r["buf"]) > 4000])

    def listen(self, mode, tg=None):
        with self.lock:
            self.sel = {"mode": mode if mode in ("off", "scan", "tg") else "off", "tg": tg if mode == "tg" else None}


# ---- sources -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def radio_source(cf, fs, gain, stop, remote=None):
    import radio_io
    sdr = radio_io.open_radio(cf, fs, remote=remote); radio_io.set_gain(sdr, gain); sdr.start_rx(); sdr.get_samples()
    try:
        while not stop.is_set():
            b = sdr.get_samples()
            if len(b):
                yield np.asarray(b, np.complex64)
            else:
                time.sleep(0.002)
    finally:
        try:
            sdr.stop_rx(); sdr.close()  # releases the radio (for a remote one: the server's device)
        except Exception:
            pass


def file_source(path, fs, stop, speed=1.0, loop=True):
    x = np.memmap(path, np.complex64, "r"); n = int(fs*0.05); i = 0; t0 = time.monotonic(); k = 0
    while not stop.is_set():
        if i + n > len(x):
            if not loop:
                return
            i = 0
        yield np.array(x[i:i + n]); i += n; k += 1
        wait = t0 + k*0.05/speed - time.monotonic()
        if wait > 0:
            time.sleep(wait)


def run_dsp(mon, source, stop):
    q = queue.Queue(maxsize=60)

    def reader():
        for blk in source:
            try:
                q.put_nowait(blk)
            except queue.Full:
                mon.stats["overruns"] += 1
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()
    while not stop.is_set():
        try:
            blk = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if blk is None:
            break
        mon.process(blk)


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
                self.reply(page, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.reply(json.dumps(mon.state()).encode(), "application/json")
            elif self.path == "/api/spectrum":
                self.reply(json.dumps(mon.spectrum()).encode(), "application/json")
            elif self.path.startswith("/api/recent/"):
                try:
                    r = list(mon.recent)[int(self.path.rsplit("/", 1)[1])]; self.reply(bytes(r["buf"]), "application/octet-stream")
                except (ValueError, IndexError):
                    self.send_error(404)
            elif self.path == "/audio":
                q = queue.Queue(maxsize=200); mon.clients.append(q)
                self.send_response(200); self.send_header("Content-Type", "application/octet-stream"); self.send_header("Transfer-Encoding", "chunked"); self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close"); self.end_headers()
                try:
                    while True:
                        try:
                            d = q.get(timeout=1.0)
                        except queue.Empty:
                            continue
                        self.wfile.write(b"%x\r\n" % len(d) + d + b"\r\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    if q in mon.clients:
                        mon.clients.remove(q)
                    self.close_connection = True
            else:
                self.send_error(404)

        def do_POST(self):
            if not self.local():
                return
            if self.path != "/api/listen" or "application/json" not in self.headers.get("Content-Type", ""):
                self.send_error(400); return
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                mon.listen(d.get("mode", "off"), d.get("tg"))
            except (ValueError, TypeError):
                self.send_error(400); return
            self.reply(json.dumps(mon.state()["sel"]).encode(), "application/json")
    return H


def main(argv=None):
    import radio_io
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cf", type=float, required=True, help="centre of the capture [MHz]"); ap.add_argument("--fs", type=float, default=8.0, help="[MS/s]")
    ap.add_argument("--control", type=float, nargs="+", required=True, help="control channel(s) [MHz], inside the capture span")
    radio_io.add_gain_arg(ap, 0.7)
    ap.add_argument("--ppm", type=float, default=15.0, help="the radio's clock error: signals appear at f*(1+ppm/1e6) (see the HackRF clock note)")
    ap.add_argument("--remote", metavar="ADDRESS", help="use the HackRF of a pysdrlib server, e.g. tcp://HOST:5555 (default: the PYSDRLIB_REMOTE environment variable, else a local radio)")
    ap.add_argument("--file", help="play a .cf64 recording instead of the radio"); ap.add_argument("--speed", type=float, default=1.0, help="playback speed for --file")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8025)
    a = ap.parse_args(argv)
    fs = a.fs*1e6; mon = Monitor(fs, a.cf*1e6, a.ppm, a.control); stop = threading.Event()
    src = file_source(a.file, fs, stop, a.speed) if a.file else radio_source(a.cf*1e6, fs, a.gain, stop, a.remote)
    mon.radio = dict(source=("file " + os.path.basename(a.file)) if a.file else ("radio at " + (a.remote or os.environ.get("PYSDRLIB_REMOTE")) if (a.remote or os.environ.get("PYSDRLIB_REMOTE")) else "HackRF"), cf=a.cf, fs=a.fs, usable=0.84*a.fs, gain=a.gain, ppm=a.ppm)
    threading.Thread(target=run_dsp, args=(mon, src, stop), daemon=True).start()
    page = open(os.path.join(HERE, "p25_monitor.html"), "rb").read()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(mon, page)); srv.allow_remote = a.host not in ("127.0.0.1", "localhost", "::1"); srv.daemon_threads = True
    print(f"p25 monitor: http://{a.host}:{a.port}/  ({'file ' + a.file if a.file else 'radio'}, {a.cf:g} MHz, {a.fs:g} MS/s, control {a.control})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
