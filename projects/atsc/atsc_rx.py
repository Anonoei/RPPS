"""Streaming ATSC (8VSB) receiver: a 20 MS/s capture stream in, MPEG transport stream packets out, in real time

    rx = Receiver(cf, lower)            # tune centre [Hz], the channel's lower edge [Hz]
    rx.on_ts = lambda ts, err: ...      # ts (n, 188) uint8 packets, err (n,) Reed-Solomon errors (-1: uncorrectable), called every ~0.1 s of air
    rx.feed(block)                      # complex64 samples, any length; call it with the radio's blocks
    rx.stats                            # state, snr, rs_ok, fields, lost, clock ... (a dict, updated as it runs)
    ts, err, psi, stats = decode_array(capture, cf, lower)    # a finished capture in one call (what the monitor's probe uses)

The offline receiver (atsc_probe.recover + atsc_decode) estimates everything from the whole capture and costs ~6 CPU-seconds per second of air. This one estimates once and then tracks:
  acquire (first ~0.8 s): pilot (FFT) -> clock (segment-sync drift, as recover does) -> segment and field sync -> an equalizer trained on the field syncs and the segment syncs, then on its own decisions.
  track (every 4 fields, ~0.1 s): `Front` (rpps `FractionalPolyphase`) gives the channel at ONE complex sample per symbol; the pilot's phase (per 8192 symbols) turns it into the real 8VSB signal, a 64-tap complex equalizer at the symbol
  spacing removes the channel (the two real degrees of freedom per symbol of the T/2 real equalizer of atsc_probe), the segments go through the trellis (C), the byte de-interleaver (state carried), Reed-Solomon
  (atsc_decode.rs_decode: all codewords in one call, rpps._native) and the de-randomizer, field after field, with no chunk edges. The equalizer is re-solved every region from decision-directed least squares with a forgetting factor; the
  segment-sync position of the unequalized signal drives the symbol clock; a field sync that stops correlating drops back to acquire.
Meant to be upstreamed into RPPS (rp.symbol.VSB, rp.signal.filters.equalizer).
"""
import argparse
import queue
import sys
import threading
import time

from collections import deque

import numpy as np
from rpps.signal.filters import convolve

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import atsc_decode as ad
import atsc_probe as ap
import rpps as rp

FS = 20e6
FRONT_CUTOFF = 3.7e6 # [Hz] flat over the 6 MHz channel (+-3 MHz), 60 dB down by 7.7 MHz (the first alias of the 10.76 MS/s output)
SEG, FIELD = ap.SEG, ap.FIELD
FIELD_SYM = SEG*FIELD # 260 416 symbols
SR = ap.SYMBOL_RATE
PILOT_REL = 3e6 - ap.PILOT_OFFSET # the pilot sits this far below the channel centre [Hz]
L, C = 64, 24 # equalizer taps (complex, one per symbol) and how many of them look back
K = 4 # fields per region (~97 ms)
ACQ_SECONDS = 0.9
WIN_APP, WIN_EST = 1024, 8192 # pilot phase: applied per 1024 symbols, measured over 8192
PILOT_MIN = 10.0 # dB, as in atsc_live
CLOCK_GAIN = float(os.environ.get("ATSC_CLOCK_GAIN", "0.25")) # fraction of the measured sync drift removed from the symbol clock per region
FORGET = 0.2 # weight kept of the previous regions' normal equations
DD_SEGS, DD_STRIDE = 80, 2 # decision-directed rows per region: segments x every DD_STRIDE-th symbol
SYNC = np.array([5.0, -5.0, -5.0, 5.0])
KEY = ad.randomizer_bytes(ad.FIELD_DATA*187).reshape(ad.FIELD_DATA, 187)


def carrier(z, n0, w_p):
    """Phasors that make the pilot real and positive: exp(-j (w_p n + phi_n)) for symbols n0 .. n0 + len(z) - 1. phi_n is the pilot's phase, measured over 8192 symbols and applied per 1024.
    Returns (phasors complex64, residual frequency [rad/symbol] to add to w_p)"""
    n = len(z); nb = -(-n//WIN_APP)
    zz = np.zeros(nb*WIN_APP, np.complex64); zz[:n] = z
    b = np.arange(nb)
    base = np.exp(-1j*w_p*(n0 + b*WIN_APP)).astype(np.complex64)
    rot = np.exp(-1j*w_p*np.arange(WIN_APP)).astype(np.complex64)
    ph = base[:, None]*rot[None, :] # (nb, 1024)
    d = zz.reshape(nb, WIN_APP)*ph
    m1 = d.sum(axis=1)
    if nb >= 8:
        cs = np.convolve(m1, np.ones(WIN_EST//WIN_APP), "same")
    else:
        cs = np.full(nb, m1.sum())
    phi = np.unwrap(np.angle(cs))
    slope = float(np.polyfit(b, phi, 1)[0])/WIN_APP if nb >= 8 else 0.0
    ph = ph*np.exp(-1j*phi).astype(np.complex64)[:, None]
    return ph.reshape(-1)[:n], slope


def fold_peak(r, nseg):
    """Segment-sync correlation of the real signal r folded on 832 symbols: (profile, nseg rows used)"""
    s = r[:-3] - r[1:-2] - r[2:-1] + r[3:]
    n = len(s)//SEG
    return s[:n*SEG].reshape(n, SEG)


def fit_levels(soft, iters=3):
    """atsc_probe.fit_levels without polyfit: gain and offset of the eight levels from the slicer's decisions on a 1/31 subsample (closed-form least squares)"""
    d = np.array(soft, np.float32)
    for _ in range(iters):
        v = d[:, 4:].ravel()[::31].astype(np.float64)
        lv = ap.slice_levels(v)
        lm, vm = lv.mean(), v.mean()
        g = float(np.dot(lv - lm, v - vm)/max(np.dot(lv - lm, lv - lm), 1e-9)); o = vm - g*lm
        d = (d - np.float32(o))/np.float32(g)
    return d


def parabola(prof, k):
    a, b, c = prof[(k - 1) % SEG], prof[k % SEG], prof[(k + 1) % SEG]
    den = a - 2*b + c
    return k + (0.5*(a - c)/den if den else 0.0)



class Front:
    """The channel centred and resampled to the symbol rate, one complex sample per symbol: `rp.signal.sample.FractionalPolyphase` with the channel's offset fused in

        f = Front(cf, center_hz, step)           # step: input samples per symbol (20e6 / 10.7622e6 = 1.8582, trimmed to the clock error)
        z = f.process(block)                     # complex64, continuous across blocks (oscillator phase, filter history and the fractional read position are carried)
        f.step = ...                             # the clock trim: takes effect from the next block

    A 6 MHz channel centred on 0 is a complex signal 6 MHz wide, so one complex sample per symbol (10.76 MS/s) holds all of it with no aliasing, and the real T/2 signal
    the offline receiver (atsc_probe) builds at 21.5 MS/s is not needed: a complex equalizer at one sample per symbol has the same two real degrees of freedom per symbol (see
    the module docstring). Z[0] is input sample 0 (no startup skip).
    """
    def __init__(self, cf, center_hz, step, fs=FS):
        self.cf, self.fs, self.stream = cf, fs, None
        self.stage = rp.signal.sample.FractionalPolyphase(1.0/step, cutoff=FRONT_CUTOFF, shift=cf - center_hz)

    @property
    def step(self):
        return 1.0/self.stage.ratio

    @step.setter
    def step(self, v):
        self.stage.ratio = 1.0/float(v)

    def process(self, x):
        d = rp.Data(np.ascontiguousarray(x, np.complex64), Fs=self.fs, sps=1, cf=self.cf, stream=self.stream)
        self.stream = d.stream
        return np.asarray((d + self.stage).data)

class Equalizer:
    """Complex symbol-spaced FIR + bias: out_n = Re(phasor_n * sum_m g_m z[n + m - C]) + bias, trained by (decision-directed) least squares on normal equations kept with a forgetting factor"""
    def __init__(self):
        self.g = np.zeros(L, np.complex64); self.g[C] = 1.0; self.bias = 0.0
        self.A = np.zeros((2*L + 1, 2*L + 1)); self.b = np.zeros(2*L + 1)

    def apply(self, zext, ph):
        """zext: the symbols with C before and L - C after the n of interest; returns the equalized real signal for the len(ph) symbols between"""
        n = len(ph)
        y = convolve(zext, self.g[::-1], mode="full")[L - 1:L - 1 + n]
        return (y*ph).real.astype(np.float32) + np.float32(self.bias)

    @staticmethod
    def normal(zext, ph, rows, target, weight=None):
        sw = np.lib.stride_tricks.sliding_window_view(zext, L)
        X = sw[rows]*ph[rows, None]
        F = np.concatenate((X.real, -X.imag, np.ones((len(rows), 1), np.float32)), axis=1).astype(np.float32)
        if weight is not None:
            Fw = F*weight[:, None].astype(np.float32)
        else:
            Fw = F
        return (Fw.T @ F).astype(np.float64), (Fw.T @ np.asarray(target, np.float32)).astype(np.float64)

    def solve(self, A=None, b=None):
        A = self.A if A is None else A; b = self.b if b is None else b
        u = np.linalg.solve(A + 1e-7*np.trace(A)/len(A)*np.eye(len(A)), b)
        self.g = (u[:L] + 1j*u[L:2*L]).astype(np.complex64); self.bias = float(u[2*L])

    def update(self, A, b, forget=None):
        forget = FORGET if forget is None else forget
        self.A = forget*self.A + A; self.b = forget*self.b + b
        self.solve()


class TsBuilder:
    """Trellis output bytes (whole fields) -> transport packets: the convolutional byte de-interleaver with its history carried, Reed-Solomon, de-randomizer (continuous, restarts at a reset)"""
    def __init__(self):
        self.hist = np.zeros(ad.DELAY, np.uint8); self.skip = ad.DELAY; self.out = np.zeros(0, np.uint8); self.cw = 0

    def push(self, y):
        ycat = np.concatenate((self.hist, y))
        i = np.arange(len(y))
        z = ycat[ad.DELAY + i - ad.BRANCHES*ad.BRANCH_MULT*(ad.BRANCHES - 1 - i % ad.BRANCHES)]
        self.hist = ycat[-ad.DELAY:]
        if self.skip:
            t = min(self.skip, len(z)); z = z[t:]; self.skip -= t
        self.out = np.concatenate((self.out, z))
        n = len(self.out)//207
        cw = self.out[:n*207].reshape(n, 207); self.out = self.out[n*207:]
        data, err = ad.rs_decode(cw)
        plain = data ^ KEY[(self.cw + np.arange(n)) % ad.FIELD_DATA]
        self.cw += n
        return ad.transport_stream(plain, err), err


class Receiver:
    def __init__(self, cf, lower, fs=FS, on_ts=None):
        self.cf, self.lower, self.fs, self.on_ts = cf, lower, fs, on_ts
        self.stats = dict(state="acquire", snr=None, rs_ok=None, fields=0, lost=0, clock_ppm=None, pilot_db=None, packets=0, bad=0, lag=0.0, acquired=0, mer=None)
        self.sync_trace = deque(maxlen=6); self.step_hint = None; self.acq = []; self.acq_n = 0; self.threaded = False; self.gen = 0; self.rlock = threading.RLock(); self.lost_flag = False; self.tail_error = None
        self._reset()

    def _reset(self):
        self.state = "acquire"; self.front = None; self.zbuf = np.zeros(0, np.complex64); self.zoff = 0; self.next_field = None
        self.eq = Equalizer(); self.ts = TsBuilder(); self.w_p = 0.0; self.p_prev = None; self.bad_sync = 0; self.stats["state"] = "acquire"

    def reset(self):
        self.acq = []; self.acq_n = 0; self._reset()

    # ------------------------------------------------------------------ streaming
    def feed(self, x):
        x = np.asarray(x, np.complex64)
        if self.state != "track":
            self.acq.append(x); self.acq_n += len(x)
            if self.acq_n < ACQ_SECONDS*self.fs:
                return
            buf = np.concatenate(self.acq); self.acq = []; self.acq_n = 0
            if not self._acquire(buf):
                return
        else:
            z = self.front.process(x)
            self.zbuf = np.concatenate((self.zbuf, z))
        self._advance()

    def _advance(self):
        while self.state == "track" and self.zoff + len(self.zbuf) >= self.next_field + K*FIELD_SYM + (L - C):
            self._region()

    # ------------------------------------------------------------------ acquisition
    def _chain(self, x, center, step):
        """Front + carrier + raw real signal for a capture; returns (front, z, raw real, w_p, residual slope)"""
        fr = Front(self.cf, center, step, self.fs)
        z = fr.process(x)
        w_p = -2*np.pi*PILOT_REL*step/self.fs
        ph, slope = carrier(z, 0, w_p)
        return fr, z, (z*ph).real.astype(np.float32), w_p, slope

    def _clock(self, raw):
        """Symbol clock error [ppm] from the drift of the segment-sync position over the capture (64-segment chunks, regression), and the folded profile"""
        rows = fold_peak(raw, 0)
        prof = rows.mean(0)
        pos = []
        for i in range(0, len(rows) - 63, 64):
            f = rows[i:i + 64].mean(0); pos.append(parabola(f, int(np.argmax(f))))
        pos = np.unwrap(np.array(pos), period=SEG)
        slope = np.polyfit(np.arange(len(pos)), pos, 1)[0]/64 # symbols per segment
        return slope/SEG*1e6, prof

    def _acquire(self, x):
        st = self.stats
        try:
            fp, over = ap.find_pilot(x, self.fs, self.cf, self.lower)
        except Exception:
            return False
        st["pilot_db"] = float(over)
        if over < PILOT_MIN:
            st["state"] = "no signal"; return False
        center = fp + PILOT_REL
        nominal = self.lower + ap.PILOT_OFFSET
        ppm0 = (fp - nominal)/nominal
        base = self.fs/SR
        if self.step_hint is not None: # a re-acquisition: the clock is where it was a moment ago
            step, iters = self.step_hint, 2
        else:
            cands = [base*(1 + ppm0), base*(1 - ppm0)]
            short = x[:int(0.3*self.fs)]
            score = []
            for s in cands:
                raw = self._chain(short, center, s)[2]
                prof = fold_peak(raw, 0).mean(0)
                score.append(np.abs(prof).max()/prof.std())
            step, iters = cands[int(np.argmax(score))], 6
        for it in range(iters): # recover's loop: clock from the sync drift, again with it removed
            raw = self._chain(x, center, step)[2]
            ppm, prof = self._clock(raw)
            if abs(ppm) < 0.004:
                break
            step *= 1 + ppm*1e-6
        fr, z, raw, w_p, slope = self._chain(x, center, step)
        w_p += slope
        prof = fold_peak(raw, 0).mean(0)
        if prof.max() < -prof.min():
            st["state"] = "bad polarity"; return False
        start = int(np.argmax(prof))
        nseg = (len(raw) - start)//SEG
        if nseg < 2*FIELD + 20:
            return False
        segs = raw[start:start + nseg*SEG].reshape(nseg, SEG)
        data = segs[:, 4:]
        A_, B_ = data[:-FIELD], data[FIELD:]
        rho = np.sum(A_*B_, 1)/np.sqrt(np.sum(A_*A_, 1)*np.sum(B_*B_, 1) + 1e-12)
        hits = np.where(rho > 0.5)[0]
        if len(hits) < 3:
            st["state"] = "no field sync"; return False
        fs_seg = int(np.bincount(hits % FIELD).argmax())
        fsegs = [k for k in range(nseg) if k % FIELD == fs_seg]
        npn = 4 + 511
        avg = np.mean([segs[k][:npn] for k in fsegs], axis=0)
        self.pn = np.sign(avg - np.mean(avg))*5.0
        # equalizer: first on the known symbols (field syncs and every segment sync), then on its decisions
        zext = np.concatenate((np.zeros(C, np.complex64), z, np.zeros(L - C, np.complex64)))
        ph, _ = carrier(z, 0, w_p)
        eq = Equalizer()
        rows_k = np.concatenate([start + k*SEG + np.arange(npn) for k in fsegs]); tgt_k = np.tile(self.pn, len(fsegs))
        rows_s = np.concatenate([start + k*SEG + np.arange(4) for k in range(nseg) if k % FIELD != fs_seg]); tgt_s = np.tile(SYNC, len(rows_s)//4)
        ok = lambda r: (r >= 0) & (r < len(z))
        rows = np.concatenate((rows_k, rows_s)); tgt = np.concatenate((tgt_k, tgt_s)); wgt = np.concatenate((np.ones(len(rows_k)), np.full(len(rows_s), 3.0)))
        m = ok(rows)
        A, b = Equalizer.normal(zext, ph, rows[m], tgt[m], wgt[m]); eq.solve(A, b)
        for it in range(2):
            out = eq.apply(zext, ph)
            soft = ap.fit_levels(out[start:start + nseg*SEG].reshape(nseg, SEG))
            dec = ap.slice_levels(soft[:, 4:])
            dk = np.array([k for k in range(nseg) if k % FIELD != fs_seg])
            sel = dk[::max(1, len(dk)//200)] # segments spread over the capture
            r2 = (start + sel[:, None]*SEG + 4 + np.arange(828)[None, :])[:, ::3].ravel(); t2 = dec[sel][:, ::3].ravel()
            A2, b2 = Equalizer.normal(zext, ph, r2, t2)
            A3, b3 = Equalizer.normal(zext, ph, rows[m], tgt[m], wgt[m]*2.0)
            eq.solve(A2 + A3, b2 + b3)
        eq.A, eq.b = A2 + A3, b2 + b3
        self.eq = eq; self.front = fr; self.step = step; self.w_p = w_p
        self.zbuf = z; self.zoff = 0
        self.next_field = start + fs_seg*SEG
        while self.next_field < C:
            self.next_field += FIELD_SYM
        self.ts = TsBuilder(); self.p_prev = None; self.bad_sync = 0
        self.step_hint = step
        self.state = "track"; st["state"] = "locked"; st["acquired"] += 1; st["clock_ppm"] = float((step/(self.fs/SR) - 1)*1e6)
        return True

    def _emit(self, soft):
        if self.threaded:
            self.dec_q.put((self.gen, soft))
        else:
            self._decode(soft)

    def _decode(self, soft):
        st = self.stats
        fields = ad.decode_fields(soft, 0)
        y = np.concatenate([bb for _, bb in fields if len(bb) == ad.FIELD_DATA*207]) if fields else np.zeros(0, np.uint8)
        if len(y) == K*ad.FIELD_DATA*207:
            ts, err = self.ts.push(y)
            if len(ts):
                good = err >= 0
                st["rs_ok"] = float(good.mean()) if st["rs_ok"] is None else 0.8*st["rs_ok"] + 0.2*float(good.mean())
                st["packets"] += len(ts); st["bad"] += int((~good).sum()); st["fields"] += K
                if self.on_ts:
                    self.on_ts(ts, err)

    # ------------------------------------------------------------------ threads: capture blocks -> front end -> equalizer regions -> trellis / Reed-Solomon / packets
    def start(self, backlog_seconds=6.0):
        """Run as three threads (front end, regions, decode): push() the radio's blocks, they never wait for the DSP. Everything heavy is in C or BLAS, which release the GIL."""
        self.threaded = True; self.stop_flag = threading.Event(); self.overruns = 0; self.busy = dict(front=0.0, region=0.0, decode=0.0); self.air = dict(front=0.0, region=0.0, decode=0.0)
        self.in_q = queue.Queue(maxsize=max(4, int(backlog_seconds*self.fs/(1 << 20)))); self._pend = []; self._pend_n = 0; self.z_q = queue.Queue(maxsize=4); # bounded: the front end may run only ~0.2 s ahead of the regions, so a clock correction reaches samples that have not been processed yet
        self.dec_q = queue.Queue() # unbounded: a skipped region would break the de-interleaver's history; `lag` shows a backlog
        self.threads = [threading.Thread(target=f, daemon=True, name=f.__name__) for f in (self._front_loop, self._region_loop, self._decode_loop)]
        for t in self.threads:
            t.start()
        return self

    def push(self, x):
        """Queue the radio's samples (blocks of any size; gathered into ~1M-sample pieces). A full queue drops a piece (`overruns`): the receiver notices the gap and re-acquires"""
        x = np.asarray(x, np.complex64)
        self._pend.append(x); self._pend_n += len(x)
        if self._pend_n < (1 << 20):
            return
        piece = np.concatenate(self._pend) if len(self._pend) > 1 else self._pend[0]
        self._pend = []; self._pend_n = 0
        try:
            self.in_q.put_nowait(piece)
        except queue.Full:
            self.overruns += 1; self.stats["overruns"] = self.overruns

    def stop(self):
        self.stop_flag.set()
        for t in self.threads:
            t.join(timeout=5)

    def _front_loop(self):
        while not self.stop_flag.is_set():
            try:
                x = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            t_b = time.perf_counter(); self.air["front"] += len(x)/self.fs
            self.stats["lag"] = (self.in_q.qsize()*(1 << 20)/self.fs) + self.z_q.qsize()*0.05 + self.dec_q.qsize()*K*FIELD_SYM/SR
            try:
                if self.lost_flag:
                    with self.rlock:
                        self.lost_flag = False; self.gen += 1; self.reset(); self.stats["state"] = "lost"
                if self.state != "track":
                    self.acq.append(x); self.acq_n += len(x)
                    if self.acq_n >= ACQ_SECONDS*self.fs:
                        buf = np.concatenate(self.acq); self.acq = []; self.acq_n = 0
                        with self.rlock:
                            self.gen += 1
                        if self._acquire(buf):
                            z0 = self.zbuf; self.zbuf = np.zeros(0, np.complex64)
                            self._put_z((self.gen, z0))
                else:
                    self._put_z((self.gen, self.front.process(x)))
            except Exception as e: # a bad block must not end the receiver
                self.tail_error = f"{type(e).__name__}: {e}"; self.stats["error"] = self.tail_error
            self.busy["front"] += time.perf_counter() - t_b; self._load()

    def _load(self):
        """Seconds of CPU per second of air of each thread so far (1.0 = the limit of real time)"""
        self.stats["load"] = {k: round(self.busy[k]/max(self.air[k], 1e-3), 2) for k in self.busy}

    def _put_z(self, item):
        while not self.stop_flag.is_set():
            try:
                self.z_q.put(item, timeout=0.2); return
            except queue.Full:
                continue

    def _region_loop(self):
        while not self.stop_flag.is_set():
            try:
                gen, z = self.z_q.get(timeout=0.2)
            except queue.Empty:
                continue
            t_b = time.perf_counter(); self.air["region"] += (K*FIELD_SYM/SR)*0 + len(z)/SR
            try:
                with self.rlock:
                    if gen != self.gen or self.state != "track":
                        continue
                    self.zbuf = np.concatenate((self.zbuf, z))
                    self._advance()
            except Exception as e:
                self.tail_error = f"{type(e).__name__}: {e}"; self.stats["error"] = self.tail_error
            self.busy["region"] += time.perf_counter() - t_b

    def _decode_loop(self):
        while not self.stop_flag.is_set():
            try:
                gen, soft = self.dec_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self.gen:
                continue
            t_b = time.perf_counter()
            try:
                self._decode(soft)
            except Exception as e:
                self.tail_error = f"{type(e).__name__}: {e}"; self.stats["error"] = self.tail_error
            self.busy["decode"] += time.perf_counter() - t_b; self.air["decode"] += K*FIELD_SYM/SR

    # ------------------------------------------------------------------ tracking
    def _region(self):
        st = self.stats
        a = self.next_field - self.zoff; b = a + K*FIELD_SYM
        core = self.zbuf[a:b]
        zext = self.zbuf[a - C:b + (L - C)]
        ph, slope = carrier(core, self.next_field, self.w_p)
        self.w_p += 0.5*slope
        raw = (core*ph).real
        # clock: where the unequalized segment sync is, folded over the region; expected at 0
        prof = fold_peak(raw, 0).mean(0)
        win = np.arange(-12, 13) % SEG
        k = int(win[np.argmax(prof[win])]); p = float(parabola(prof.astype(np.float64), k))
        p = p - SEG if p > SEG/2 else p
        if self.p_prev is not None:
            rate = (p - self.p_prev)/(K*FIELD) # symbols per segment
            self.front.step = float(self.front.step*(1.0 + CLOCK_GAIN*rate/SEG))
            self.step = self.front.step
        self.p_prev = p
        shift = int(round(p)) if abs(p) > 2.0 else 0 # the sync has walked away from where we cut: cut later / earlier (the taps stay valid)
        # equalize with the taps so far, train on this region's own decisions (and the known sync symbols), equalize again with the result: the offline receiver's block-wise training, causal in the taps' memory
        dk = np.arange(K*FIELD)[np.arange(K*FIELD) % FIELD != 0]
        sel = dk[::max(1, len(dk)//DD_SEGS)]
        rs_ = (dk[:, None]*SEG + np.arange(4)[None, :]).ravel()[::2]; ts_ = np.tile(SYNC, len(dk))[::2]
        for it in range(2):
            out = self.eq.apply(zext, ph)
            soft = fit_levels(out.reshape(K*FIELD, SEG), iters=3)
            if it == 1:
                break
            r2 = (sel[:, None]*SEG + 4 + np.arange(828)[None, :])[:, ::DD_STRIDE].ravel(); t2 = ap.slice_levels(soft[sel][:, 4:])[:, ::DD_STRIDE].ravel()
            rows = np.concatenate((r2, rs_)); tgt = np.concatenate((t2, ts_)); wgt = np.concatenate((np.ones(len(r2)), np.full(len(rs_), 6.0)))
            A, b = Equalizer.normal(zext, ph, rows, tgt, wgt)
            self.eq.update(A, b)
        keep = np.arange(len(soft)) % FIELD != 0
        nz = float(np.mean((soft[keep][:, :4] - SYNC)**2)); st["snr"] = float(10*np.log10(25/nz)) if nz > 0 else None
        st["p"] = p
        fsv = soft[::FIELD, :4 + 511]
        corr = float(np.mean([np.dot(np.sign(f[:len(self.pn)] - f.mean()), np.sign(self.pn))/len(self.pn) for f in fsv])) if hasattr(self, "pn") else 1.0
        self.bad_sync = self.bad_sync + 1 if abs(corr) < 0.4 else 0
        self.sync_trace.append((round(corr, 2), round(p, 1), round(st.get("snr") or 0, 1), round(st.get("lag") or 0, 2))) # the last regions, for the report when lock is lost
        self._emit(soft)
        # advance
        self.next_field += K*FIELD_SYM + shift
        cut = self.next_field - C - self.zoff
        if cut > 0:
            self.zbuf = self.zbuf[cut:]; self.zoff += cut
        if self.bad_sync >= 3:
            st["lost"] += 1; st["state"] = "lost"
            st["loss_trace"] = list(self.sync_trace)
            if self.threaded:
                self.lost_flag = True; self.state = "lost"
            else:
                self.reset(); st["state"] = "lost"


def decode_array(x, cf, lower, fs=FS):
    """A finished capture through the streaming receiver: (ts packets (n, 188), errors (n,), stats). A capture shorter than ~1.3 s gives nothing (acquisition takes 0.9 s, a region 0.1 s)"""
    outs = []
    rx = Receiver(cf, lower, fs, on_ts=lambda ts, err: outs.append((ts, err)))
    for i in range(0, len(x), 1 << 20):
        rx.feed(x[i:i + (1 << 20)])
    if not outs:
        return np.zeros((0, 188), np.uint8), np.zeros(0, int), rx.stats
    return np.concatenate([t for t, _ in outs]), np.concatenate([e for _, e in outs]), rx.stats


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap_.add_argument("path"); ap_.add_argument("--cf", type=float, required=True, help="capture centre [MHz]"); ap_.add_argument("--lower", type=float, required=True, help="channel lower edge [MHz]")
    ap_.add_argument("--seconds", type=float, default=0); ap_.add_argument("--out", help="write the transport stream here")
    a = ap_.parse_args(argv)
    x = np.memmap(a.path, np.complex64, "r")
    if a.seconds:
        x = x[:int(a.seconds*FS)]
    t = time.time()
    ts, err, st = decode_array(np.asarray(x), a.cf*1e6, a.lower*1e6)
    dt = time.time() - t
    print(f"{len(x)/FS:.1f} s of air in {dt:.1f} s ({len(x)/FS/dt:.2f}x real time): {len(ts)} packets, Reed-Solomon ok {100*np.mean(err >= 0) if len(err) else 0:.1f}%, SNR {st['snr']}, clock {st['clock_ppm']} ppm, state {st['state']}")
    if a.out:
        ts[err >= 0].tofile(a.out)


if __name__ == "__main__":
    main()
