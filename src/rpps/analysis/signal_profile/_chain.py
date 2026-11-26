"""One signal through the chain: channel, Acquire, timing and carrier, the code (rpps.analysis.signal_profile)"""
import traceback

import numpy as np

from ._framed import framed
from ._measures import CARRIER_SHARE, ESN0_GAP, NEAR, PM_OFFSET, RATE_WIDTH, SYMS, _strong, _waveform_detail, carrier_share, esn0_gap
from ._result import FOUND, GUESS, MIN_PAYLOAD, SignalProfile, Step, UNSOLVED, _code_steps, _guess_steps, weights

class _Chain:
    """A signal's chain over the blocks of its capture, and what it keeps for the profile"""
    def __init__(self, sig, Fs, blind, baud=None, ch=None, codes=None):
        """ch: the channel an rp.signal.FFTBank brings this signal to baseband in (its samples are given to `step` at
        ch.Fs); None: the chain channelizes the capture's samples itself (sig.channel())"""
        from ... import sync
        from ...base import Stream
        self.sig, self.Fs, self.baud, self.banked = sig, Fs, baud, ch is not None
        self.errors = []
        self.acq, self.mat, self.ff = sync.Acquire(baud=baud), sync.time.Mat(), sync.phase.FFCarrier()
        self.fec = sync.AcquireFEC(blind=blind, names=codes)
        self.raw, self.nsym, self.dead = [], 0, False
        self.near, self.nnear = [], 0 # The channel's first NEAR samples, for rpps.analysis.spectrum.waveform
        self.syms, self.nsyms = [], 0 # The first SYMS carrier-tracked symbols, for rpps.analysis.framing.frames
        if getattr(sig, "holds", 0): # A raised floor under other signals: they are profiled, it isn't one
            self.ch, self.dead = None, True
            return
        try:
            self.ch = ch if ch is not None else sig.channel()
            self.stream = Stream(self.ch.Fs if ch is not None else Fs)
        except Exception as e:
            self.errors.append(("channel", e, traceback.format_exc(limit=4)))
            self.dead = True

    def carriers(self, depth=4.0, q=95, bins=400, min_frac=1/40, ripple=4.0, edge=15.0):
        """The carriers in this chain's box that a fine spectrum of its own channel samples shows apart: the box's
        `q`th-percentile spectrum over time (bursts and TDD carriers at their full level, not averaged down) at `bins`
        bins across it, cut at each valley `depth` dB under the lower of the peaks either side (`ripple` times the
        spectrum's own ripple when that is more: a noisy estimate's dips aren't notches), within where it is less than
        `edge` dB under its top (the channel filter's skirts aren't valleys); each piece the span of its bins over
        halfway from its top to the valleys bounding it, pieces under `min_frac` of the box dropped. Signals of the
        capture, two or more, else [] (the survey's own spectrum is too coarse and too short to see notches between
        carriers a few tens of kHz apart). Under 8 rows over the percentile (a short capture, bursts): [] (a spectrum
        that rough has notches on its noise). On a hand-reviewed pool: 65 of 73 boxes reviewers said held several signals
        split, 6 of 325 they said held one"""
        import copy
        if not self.near or not self.sig.width:
            return []
        x = np.concatenate(self.near)
        Fs = self.ch.Fs
        n = 1 << int(np.ceil(np.log2(max(16, bins*Fs/self.sig.width))))
        if len(x) < 8*n:
            return []
        seg = x[:len(x)//n*n].reshape(-1, n)[:256]
        if len(seg)*(1 - q/100) < 8: # (the percentile from a row or two: a bin's spread is +-5 dB, notches on noise)
            return []
        P = np.abs(np.fft.fftshift(np.fft.fft(seg*np.hanning(n), axis=1), axes=1))**2
        f = np.fft.fftshift(np.fft.fftfreq(n, 1/Fs))
        inb = np.abs(f) <= self.sig.width/2 # (the channel is centred on the box)
        raw = 10*np.log10(np.percentile(P[:, inb], q, axis=0) + 1e-30)
        fb = f[inb]
        k = max(1, len(raw)//80)
        p = np.convolve(raw, np.ones(k)/k, mode="same")
        core = np.flatnonzero(p > np.percentile(p, 95) - edge)
        if len(core) < 8:
            return []
        i0 = core[0]
        v = p[i0:core[-1] + 1]
        wide = np.convolve(raw, np.ones(4*k)/(4*k), mode="same")[i0:core[-1] + 1]
        need = max(depth, ripple*1.4826*np.median(np.abs(v - wide)))
        cuts = []
        for i in range(1, len(v) - 1):
            if not (v[i] <= v[i - 1] and v[i] < v[i + 1]):
                continue
            l = i # Out each way while no lower: the highest passed is that side's peak
            while l > 0 and v[l - 1] >= v[i]:
                l -= 1
            r = i
            while r < len(v) - 1 and v[r + 1] >= v[i]:
                r += 1
            if min(v[l:i + 1].max(), v[i:r + 1].max()) - v[i] >= need:
                cuts.append(i)
        bounds = [0] + cuts + [len(v) - 1]
        least, df = max(3, int(len(p)*min_frac)), fb[1] - fb[0]
        out = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < least:
                continue
            t = 0.5*(v[a:b + 1].max() + max(v[a], v[b]))
            on = a + np.flatnonzero(v[a:b + 1] >= t)
            if on[-1] - on[0] + 1 < least: # (a spike: a tone or a spur on the slope, not a carrier)
                continue
            lo, hi = fb[i0 + on[0]] - df/2, fb[i0 + on[-1]] + df/2
            s = copy.copy(self.sig)
            s.mid, s.width = self.sig.mid + 0.5*(lo + hi), hi - lo
            s.lo, s.hi = s.mid - s.width/2, s.mid + s.width/2
            s.refit = None # (the box was fitted, not each carrier in it)
            out.append(s)
        return out if len(out) >= 2 else []

    def split(self, baud, beta=0.35, fill=0.6):
        """When the cyclic spectrum's rate needs well under the box (its (1 + beta) x baud under `fill` of the width: carriers
        side by side the survey took for one, or one carrier in a box wider than it): the carriers in it, from a survey of
        this chain's own channel samples, as Signals of the capture (each at most (1 + beta) x baud wide); [] otherwise,
        and when more pieces than carriers at this rate fit in the box are found"""
        import copy
        if not self.near or (1 + beta)*baud >= fill*self.sig.width:
            return []
        from ..spectrum.signal import survey
        from ...source import Memory
        try:
            subs = survey(Memory(np.concatenate(self.near), Fs=self.ch.Fs), baud=False)
        except Exception:
            return []
        # Each piece of the finer survey wide enough to carry this rate is a carrier candidate, but one wider than the box
        # (the box itself, found again), and one centred inside a wider piece kept already (a lump of that carrier's
        # spectrum; a piece spanning most of the box hides nothing, it can be the box around carriers side by side)
        keep = []
        for g in sorted(subs, key=lambda g: -g.width):
            lo, hi = g.mid - g.width/2, g.mid + g.width/2
            if abs(g.mid) > self.sig.width/2 or g.width < 0.6*baud or g.width > self.sig.width:
                continue
            if any(abs(g.mid - 0.5*(a + b)) < (b - a)/4 for a, b in keep if b - a < 0.8*self.sig.width):
                continue
            keep.append((lo, hi))
        if len(keep) > max(1, int(self.sig.width/((1 + beta)*baud*0.8))):
            return [] # More pieces than carriers at this rate fit in the box: not carriers side by side
        out = []
        for lo, hi in keep:
            s = copy.copy(self.sig)
            s.mid, s.width = self.sig.mid + 0.5*(lo + hi), min(hi - lo, (1 + beta)*baud)
            s.lo, s.hi = s.mid - s.width/2, s.mid + s.width/2
            out.append(s)
        return out

    def wants(self, symbols):
        """Whether more of the capture could still tell this chain something: its code not decided yet, and fewer than
        `symbols` symbols seen"""
        if self.dead:
            return False
        fec = self.stream.est.get("fec") if self.stream.det is not None else None
        return not (fec is not None and fec.complete) and self.nsym < symbols

    def step(self, samps, last=False):
        """One block of the capture through the chain; `last` ends the stream (the stages decide on what they hold)"""
        if self.dead:
            return
        from ...base import Data
        stage = "channel"
        try:
            if self.banked: # Already at baseband, at the channel's rate
                near = Data(samps, Fs=self.ch.Fs, stream=self.stream, flush=last)
            else:
                near = Data(samps, Fs=self.Fs, stream=self.stream, flush=last) + self.ch.front
            if self.nnear < NEAR and len(near):
                self.near.append(near.data[:NEAR - self.nnear].copy())
                self.nnear += len(self.near[-1])
            stage = "acquire / timing"
            mat = near + self.acq + self.mat
            if sum(len(r) for r in self.raw) < 6000 + 400000:
                self.raw.append(mat.data.copy())
            stage = "carrier"
            syms = mat + self.ff
            self.nsym += len(syms)
            if self.nsyms < SYMS and len(syms):
                self.syms.append(np.asarray(syms.data)[:SYMS - self.nsyms].copy())
                self.nsyms += len(self.syms[-1])
            stage = "fec"
            syms + self.fec
        except Exception as e:
            self.errors.append((stage, e, traceback.format_exc(limit=4)))
            self.dead = True

    def cyclic_rate(self, p):
        """When the chain found no rate, or a doubtful one: the symbol rate rpps.analysis.spectrum.cyclic sees in |x|^2 (a
        calibrated line, at any rate in band: Acquire's search starts at Fs/16), to run the chain again at; else None"""
        if self.baud or not self.near or p.solved or p.profile_name:
            return None
        acq = next((st for st in p.steps if st.kind == "acquire"), None)
        det = self.stream.det
        wide = det is not None and 1.35*det.baud < 0.6*self.sig.width # (a rate in a box well wider than it needs)
        if acq is None or not (acq.status == UNSOLVED or "rate doubtful" in (acq.note or "") or
                               "fit doubtful" in (acq.note or "") or wide):
            return None
        from ..spectrum import cyclic
        try:
            f = cyclic.features(np.concatenate(self.near)[:2**18], self.ch.Fs)
        except Exception:
            return None
        if not f.linear:
            return None
        if det is not None and abs(f.baud - det.baud) < 0.01*det.baud and 1.35*f.baud >= 0.6*self.sig.width:
            return None # (the rate the chain had, in a box it fills: nothing new; in a wider box, the carrier is narrower)
        if f.baud > 1.2*self.sig.width: # (not a symbol rate: wider than the band the signal takes up)
            return None
        return f.baud

    def profile(self, name):
        p = self._linear(name)
        if p.solved or p.profile_name or not self.near:
            return p
        return self._telemetry(p) or self._waveform(p) or p

    def _telemetry(self, p):
        """A residual carrier under the signal: PM telemetry (rpps.analysis.framing.telemetry), its subcarriers demodulated; or
        None (no carrier line, or the analysis failed: the chain's profile stands)"""
        from ..framing import telemetry
        x = np.concatenate(self.near)
        if carrier_share(x) < CARRIER_SHARE:
            return None
        try:
            t = telemetry.analyze(x, self.ch.Fs)
        except Exception as e:
            p.add("telemetry", "residual-carrier PM", UNSOLVED, note=f"{type(e).__name__}: {e}")
            return None
        if t is None or t.pm.share > 0.95: # No carrier, or nothing but one (the waveform step names tones)
            return None
        m = t.pm
        width = getattr(self.sig, "width", None) or self.ch.Fs
        sym, side = telemetry.symmetry(x, self.ch.Fs, m.carrier, band=width, snr_db=getattr(self.sig, "snr", None))
        if not any(s.ok for s in t.subcarriers) and (sym < telemetry.SYMMETRIC or side < telemetry.SIDEBANDS
                                                     or abs(m.carrier) > PM_OFFSET*width):
            return None # Nothing demodulated, and the spectrum isn't phase modulation's about that line: a tone in noise or
                        # beside another signal (the waveform step names tones)
        q = SignalProfile(p.name, p.signal)
        q.symbols, q.det = p.symbols, p.det
        q.steps.append(p.steps[0])
        q.add("waveform", f"residual carrier {m.carrier:+.1f} Hz off with {100*m.share:.0f}% of the power: phase modulation, "
                          f"index {m.index:.2f} rad",
              detail={"carrier": m.carrier, "share": m.share, "index": m.index, "phase_rms": m.phase_rms, "symmetry": sym,
                      "sidebands": side},
              note=f"the linear chain's reading, not used: {p.steps[1].what}" if len(p.steps) > 1 else None)
        q.add("demodulate", "the phase around the carrier (a narrow loop), data on it and on subcarriers",
              detail={"subcarriers": [[f, h] for f, h in m.subcarriers], "tones": list(m.tones or [])},
              note=("tones in the phase: " + ", ".join(f"{f/1e3:.3f} kHz" for f in m.tones)) if m.tones else None)
        if not t.subcarriers:
            q.add("subcarrier", "data in the phase", UNSOLVED, note="neither data on the carrier nor a subcarrier stood out")
        for s in t.subcarriers:
            d = {"freq": s.freq, "baud": s.baud, "quality": s.quality, "esn0": s.esn0, "manchester": s.manchester,
                 "differential": s.differential, "rate": s.rate, "idle": s.idle}
            if not s.ok:
                q.add("subcarrier", f"{s.freq/1e3:.3f} kHz", UNSOLVED, detail=d, note=s.why)
                continue
            q.add("subcarrier", s.head(), detail=d)
            if s.idle:
                q.add("frame", f"idle: {s.idle}")
            elif s.frame is not None:
                f = s.frame
                q.add("frame", str(f), GUESS, detail={"period": f.period, "fixed": f.fixed, "frames": f.frames,
                                                       "counters": [list(c) for c in f.counters]},
                      note="blind (rpps.analysis.framing.telemetry): no known frame sync; the fields are the frame format's")
            else:
                q.add("frame", f"{len(s.bits)} data bits", UNSOLVED,
                      note="no frame period in them: a longer capture may show one")
        return q

    def _waveform(self, p):
        """When the linear chain didn't solve it: the profile by what kind of waveform it is (tones, OFDM, a constant
        envelope, bursts: rpps.analysis.spectrum.waveform), or None when nothing there says more than the chain did"""
        from ..spectrum.waveform import classify
        try:
            w = classify(np.concatenate(self.near), self.ch.Fs)
        except Exception as e: # The chain's own profile stands
            p.add("waveform", "classify", UNSOLVED, note=f"{type(e).__name__}: {e}")
            return None
        kind = w.kind
        det = self.stream.det
        lost = det is None or p.steps[1].status == UNSOLVED or "doubtful" in (p.steps[1].note or "")
        if kind is None and w.bursts is None:
            return None
        if kind is None and w.bursts is not None and not p.solved and (w.bursts.duty or 1.0) < 0.9:
            # Bursts the continuous chain could not lock to: each demodulated on its own preamble, the code in its payload
            q = SignalProfile(p.name, p.signal)
            q.symbols, q.det = p.symbols, p.det
            q.steps.append(p.steps[0])
            q.add("waveform", str(w), FOUND, detail=_waveform_detail(w),
                  note=f"the linear chain's reading, not used: {p.steps[1].what}" if len(p.steps) > 1 else None)
            self._bursts(q, np.concatenate(self.near))
            return q
        if kind is None or kind == "constant envelope" and not lost:
            # Only bursts, or a constant envelope the PSK chain locked onto anyway (a carrier is one PSK point): noted
            p.steps.insert(2 if len(p.steps) > 1 else 1, Step("waveform", str(w), FOUND, detail=_waveform_detail(w)))
            return p
        q = SignalProfile(p.name, p.signal)
        q.symbols, q.det = p.symbols, p.det
        q.steps.append(p.steps[0])
        q.add("waveform", str(w), FOUND, detail=_waveform_detail(w),
              note=f"the linear chain's reading, not used: {p.steps[1].what}" if len(p.steps) > 1 else None)
        if kind == "tones":
            q.add("demodulate", "nothing: unmodulated carrier" + ("s" if len(w.tones.lines) > 1 else ""))
        elif kind == "ofdm":
            sy = w.ofdm.systems[0][0] if w.ofdm.systems else None
            hint = (" (projects/lte: cell search, PBCH; projects/nr/nr_cell.py: NR SSB)" if sy and ("LTE" in sy or "NR" in sy)
                    else "")
            if w.ofdm.comb:
                note = (f"a pilot comb, every {w.ofdm.comb} subcarriers: FFT {w.ofdm.comb*w.ofdm.tu*self.ch.Fs:.0f} at this "
                        "rate; the cell search tools name the cell" + (f"; {sy}{hint}" if sy else ""))
            else:
                note = (f"no OFDM chain in rp: FFT {w.ofdm.tu*self.ch.Fs:.0f} at this rate, prefix "
                        f"{w.ofdm.tcp*self.ch.Fs:.0f} samples" + (f"; {sy}{hint}" if sy else ""))
            q.add("demodulate", "OFDM receiver", UNSOLVED, note=note)
        else:
            self._fsk(q, np.concatenate(self.near))
        return q

    def _bursts(self, q, x):
        """TDMA bursts (rpps.analysis.burst.demodulate: each acquired on its kind's preamble) into q, and for each
        kind the code laid out from where its payload starts (linear.frame_units on every bit map: one rotation is
        common to all bursts and unknown), else what the payload bits are like"""
        from .. import burst
        from ..code.block.linear import frame_units, solve_block
        from ..framing.payload import verdict
        from ..identify import bit_streams, _distinct
        from ... import symbol
        try:
            r = burst.demodulate(x, self.ch.Fs)
        except Exception as e:
            q.add("demodulate", "bursts, each on its preamble", UNSOLVED, note=f"{type(e).__name__}: {e}")
            return
        ok = [b for b in r.bursts if b.ok and b.symbols is not None]
        q.add("demodulate", str(r), FOUND if ok else UNSOLVED, detail={"mod": r.mod, "baud": r.baud, "bursts": len(r.bursts),
              "demodulated": len(ok), "kinds": len(r.preambles or [r.preamble])},
              note="blind (rpps.analysis.burst): each burst acquired on its own preamble")
        c = symbol.load_or_generate(r.mod)
        kinds = r.preambles or [r.preamble]
        looked = False
        for kind, pre in enumerate(kinds):
            group = [b for b in ok if (b.kind or 0) == kind and len(b.payload)]
            if len(group) < 4:
                continue
            looked = True
            pays = [np.asarray(b.payload, dtype=np.complex64) for b in group] # (blind bursts: each its own rotation)
            label = f"kind {kind} ({len(pre)}-symbol preamble, {len(group)} bursts)" if len(kinds) > 1 else f"{len(group)} bursts"
            u = burst.units([len(y) for y in pays]) # code blocks a burst holds, from the bursts' lengths
            if u is not None:
                q.add("frame", f"{label}: {u.unit} (+-{u.err:.1f}) symbol units, {u.overhead} symbols besides; "
                      f"{', '.join(f'{c} of {n}' for n, c in sorted(u.counts.items()))} units", GUESS,
                      detail={"unit": u.unit, "err": u.err, "overhead": u.overhead, "counts": u.counts, "fit": u.fit},
                      note="rpps.analysis.burst.units: payload lengths as overhead + n units")
            best = None
            for key in _distinct(bit_streams(pays[0], c, differential=False), c):
                rows = [bit_streams(y, c, differential=False, keys=[key])[key] for y in pays]
                n, d = frame_units(rows)
                if n is not None and (best is None or (n, -d) < (best[0], -best[1])):
                    best = (n, d, key, rows)
            if best is None: # no code laid out: the first map's bits
                key = next(iter(_distinct(bit_streams(pays[0], c, differential=False), c)))
                rows = [bit_streams(y, c, differential=False, keys=[key])[key] for y in pays]
            self._header(q, label, [[bit_streams(y, c, differential=False, keys=[k])[k] for y in pays]
                                    for k in _distinct(bit_streams(pays[0], c, differential=False), c)])
            if best is None:
                bits = np.concatenate(rows)
                v = verdict(bits)
                q.add("fec", f"the payload of {label}", UNSOLVED, note=f"no block code laid out from its start; the bits are {v.kind}: {v.why}")
                continue
            n, d, key, rows = best
            units = np.concatenate([r_[o:o + n] for r_ in rows for o in range(0, len(r_) - n + 1, n)])
            found = solve_block(units, n=n, offset=0)
            if found:
                for h, g in found:
                    q.add("fec", f"{h.name} in the payload of {label}", GUESS, detail={"z": h.z, "map": str(key), "n": n},
                          note=f"blind, on the bursts' payloads laid end to end from their starts: {g!r}" if g is not None else None)
            else:
                q.add("fec", f"a length-{n} code in the payload of {label}", GUESS, detail={"n": n, "deficiency": d, "map": str(key)},
                      note=f"{d} checks a word on the bursts' payload words; not identified further")
        if ok and not looked: # Demodulated, but no kind with bursts enough to look for a code in: not solved
            q.add("fec", "the bursts' payload", UNSOLVED,
                  note=f"{len(ok)} of {len(r.bursts)} bursts demodulated, under 4 of any one kind: too few to look for a code in")

    def _header(self, q, label, maps, bits=256, min_frames=8):
        """What the first `bits` of the bursts' payloads hold (rpps.analysis.framing.fields: constants, counters, enums,
        lengths), noted when something besides random bits is there. maps: the payloads' bit rows under each of the
        constellation's distinct bit maps (the bursts share one rotation, but which is unknown: a counter's bits read
        swapped and inverted under another): the map whose fields explain the most bits"""
        from ..framing import fields
        m = None
        for rows in maps:
            rows = [np.asarray(r, dtype=np.uint8) for r in rows if len(r) >= 16]
            if len(rows) < min_frames:
                continue
            try:
                mm = fields.map(rows, length=min(bits, min(len(r) for r in rows)))
            except ValueError:
                continue
            explained = sum(f.width*(2 if f.kind in ("counter", "length", "id") else 1) for f in mm.fields
                            if f.kind not in ("random", "varying"))
            if m is None or explained > m[0]:
                m = (explained, mm)
        if m is None:
            return
        m = m[1]
        found = [f for f in m.fields if f.kind not in ("random", "varying")]
        if not found or (len(found) == 1 and found[0].kind == "constant" and found[0].width < 8):
            return
        q.add("header", f"the start of {label}'s payloads: " + "; ".join(repr(f) for f in found[:6])
              + (f"; {len(found) - 6} more" if len(found) > 6 else ""), GUESS,
              detail={"fields": [{"start": f.start, "width": f.width, "kind": f.kind} for f in m.fields]},
              note="rpps.analysis.framing.fields.map on the bursts' first payload bits (one map, one rotation for all)")

    def _fsk(self, q, x):
        """A constant envelope: FSK / MSK / GMSK by the instantaneous frequency (rpps.analysis.modulation.fsk), else unsolved"""
        from ..modulation import fsk
        try:
            r = fsk.analyze(x, self.ch.Fs)
        except Exception as e:
            r = None
            why = f"{type(e).__name__}: {e}"
        else:
            why = r.why
        if r is None or r.M is None:
            q.add("demodulate", "FM, FSK or CPM", UNSOLVED,
                  note=f"a constant envelope: not a filtered PSK, which is all the linear chain tries; FSK: {why}")
            return
        q.add("demodulate", str(r), GUESS,
              detail={"M": r.M, "baud": r.baud, "h": r.h, "spacing": r.spacing, "deviation": r.deviation,
                      "centre": r.centre, "isi": r.isi, "name": r.name},
              note="blind (rpps.analysis.modulation.fsk): fsk.demodulate gives the symbols")
        q.add("fec", "the code on the FSK symbols", UNSOLVED, note="not searched: the FEC tools take PSK symbols or bits")

    def _framing(self, p, det):
        """Locked symbols no code fits: a frame structure (rpps.analysis.framing.frames) into p, and a block code on the
        frame-aligned payload: one codeword a frame when there are frames enough for one as long, else words laid end to
        end from each payload's start. True when a code was found (not framing, fill or a recurrence)"""
        from ..framing import frames
        from ..code.block.linear import solve_block, frame_units
        y = np.concatenate(self.syms)[2000:] # Past the carrier loop's start
        try:
            fr = frames.find(y, det.mod)
        except Exception as e:
            p.add("frame", "symbol framing", UNSOLVED, note=f"{type(e).__name__}: {e}")
            return
        if fr is None:
            return
        cut = frames.cut(y, fr)
        bits = frames.payload_bits(cut, fr, det.mod)
        rep, same = frames.repeats(bits, agree=0.95)
        p.add("frame", str(fr), GUESS, detail={"period": fr.period, "uw_start": fr.uw_start, "uw_length": fr.uw_length,
                                               "coherence": fr.coherence, "payload_bits": int(bits.shape[1]),
                                               "repeats": rep},
              note=f"blind (rpps.analysis.framing.frames): {bits.shape[1]} payload bits a frame" +
                   (f"; frames repeat every {rep} ({same*100:.0f}%): fill or a superframe" if rep else ""))
        n = bits.shape[1]
        # The rank method: a row per codeword, more rows than its length. A frame's payload as one codeword needs n + 48
        # frames; shorter words laid end to end from the payload's start (rows of a product code, a frame of several
        # codewords) need only enough of them across the frames (linear.frame_units)
        whole = len(bits) >= n + 48
        if not whole and len(bits) < 8:
            return
        pay = cut[:400, fr.uw_length:].ravel()
        if len(pay) < MIN_PAYLOAD: # A unique word with next to nothing after it (a short period the comb found): no code
            p.add("fec", "block code on the frame-aligned payload: not tried", UNSOLVED,
                  note=f"{len(pay)} payload symbols in the frames found: too few to judge the decisions by, or a code in")
            return
        from ... import symbol
        from ..modulation.quality import gate
        g = gate(pay, symbol.load_or_generate(det.mod).points)
        if not g.ok: # Bit errors the noise doesn't explain fill the rank: a full rank then says nothing
            p.add("fec", "block code on the frame-aligned payload: not tried", UNSOLVED,
                  note=f"the hard decisions can't be trusted ({g.why}): fix the symbols (a pilot-aided sync, "
                       "rpps.analysis.framing.frames.pilot_frames) before a rank test")
            return
        if whole:
            found, where = solve_block(bits.ravel(), n=n, offset=0), "the frame-aligned payload"
        else:
            # The length from the payload after the unique word, on the map payload_bits uses. Then the start and the map
            # (one rotation and IQ sense is common to all frames and unknown: the wrong one makes the code affine, a check
            # short) where words of that length are shortest of rank: the payload's start can be up to 3 symbols into
            # the word (a payload's first symbols the same in every frame, fixed bits or a code's offset, read as part
            # of it), and words off their start by a few bits are still short of rank, by less
            from ..identify import bit_streams, _distinct
            u, d = frame_units(list(bits), max_len=min(n, 1024))
            if u is None:
                return
            c = symbol.load_or_generate(det.mod)
            best = (d, 0, None, bits)
            for back in range(min(4, fr.uw_length)):
                pay = cut[:, fr.uw_length - back:]
                maps = _distinct(bit_streams(pay.ravel(), c, differential=False), c)
                for key, b in maps.items():
                    r_ = np.asarray(b, np.uint8).reshape(len(pay), -1)
                    _, d_ = frame_units(list(r_), min_len=u, max_len=u)
                    if d_ > best[0]:
                        best = (d_, back, key, r_)
            _, back, key, rows = best
            m = rows.shape[1]
            units = np.concatenate([r[o:o + u] for r in rows for o in range(0, m - u + 1, u)])
            into = (f", {back} symbol{'s'*(back > 1)} into the unique word" if back else "") + (f", bit map {key}" if key else "")
            found, where = solve_block(units, n=u, offset=0), f"words of {u} bits laid end to end from each frame's payload start{into}"
        for h, g in found:
            p.add("fec", h.name, GUESS, detail={"z": h.z, **{k: str(v) for k, v in h.params.items()}},
                  note=f"blind, on {where}: {g!r}" if g is not None else f"blind, on {where}")
        return any(h.params.get("kind") not in ("frame", "recurrence", "fill") for h, _ in found)

    def _tcm(self, p, det):
        """No code on the bits: a trellis code on the labels (rpps.analysis.code.conv.tcm: Ungerboeck, pragmatic), when one holds"""
        from ..code.conv import tcm
        from ... import symbol
        y = np.concatenate(self.syms)[2000:2000 + 30000]
        M = len(symbol.load_or_generate(det.mod).points)
        try:
            found = tcm.search(y, M, max_nu=5)
        except Exception as e:
            p.add("fec", "trellis-coded modulation", UNSOLVED, note=f"{type(e).__name__}: {e}")
            return
        # A check on z^0 alone (h^1.. all zero) ties no coded bit in: that bit's own recurrence (a period, a wrong rate),
        # not a trellis code
        found = [c for c in found if any(c.h[1:])]
        if found:
            c = found[0]
            p.add("fec", f"trellis code: {c}", GUESS, detail={"h": list(c.octal), "nu": c.nu, "labels": c.labels,
                                                              "rotation": c.rotation, "conj": c.conj, "z": c.z},
                  note="blind (rpps.analysis.code.conv.tcm): parity checks on the labels; a Viterbi on this trellis decodes it")

    def _cyclic(self, p, det):
        """The rate is doubtful: the cyclic spectrum's own verdict (rpps.analysis.spectrum.cyclic), with a calibrated false-alarm
        probability, as a note: its rate when it differs, or that no rate line is there at all"""
        from ..spectrum import cyclic
        try:
            f = cyclic.features(np.concatenate(self.near)[:2**18], self.ch.Fs)
        except Exception as e:
            p.add("rate", "cyclic spectrum", UNSOLVED, note=f"{type(e).__name__}: {e}")
            return
        if f.baud is None:
            what, st = "no symbol-rate line at all: not a linear modulation at a rate in band (or too short)", UNSOLVED
        elif abs(f.baud - det.baud) > 0.01*det.baud:
            what, st = f"{f.baud/1e3:.6g} kBd from {f.baud_from}, not the {det.baud/1e3:.6g} kBd acquired", GUESS
        else:
            return
        p.add("rate", f"cyclic spectrum: {what}", st, detail={"baud": f.baud, "from": f.baud_from, "order": f.order,
                                                             "envelope": f.envelope}, note=str(f))

    def _linear(self, name):
        from ... import symbol
        sig = self.sig
        p = SignalProfile(name, sig)
        p.symbols = self.nsym
        ch = getattr(self, "ch", None)
        snr = f", SNR {sig.snr:.1f} dB in band" if getattr(sig, "snr", None) is not None else ""
        if ch is None and getattr(sig, "holds", 0):
            p.add("band", f"{sig.mid/1e6:+.3f} MHz, {sig.width/1e3:.0f} kHz wide, holding {sig.holds} other signals")
            p.add("demodulate", "nothing: a raised noise floor (a satellite transponder's noise and intermodulation) under "
                                "the signals it holds, each profiled on its own")
            return p
        if ch is None:
            p.add("channel", f"{sig.mid/1e6:+.3f} MHz, {sig.width/1e3:.0f} kHz wide", UNSOLVED,
                  note=f"{self.errors[0][0]}: {self.errors[0][1]}")
            return p
        p.add("channel", f"shift {-sig.mid/1e6:+.4f} MHz, filter to {sig.width/1e3:.0f} kHz{snr}, decimate /{ch.decimation} "
                         f"-> {ch.Fs/1e6:.4g} MS/s")
        det = self.stream.det
        p.det = det
        failed = {stage: (e, tb) for stage, e, tb in self.errors}
        if det is None:
            why = f"{type(failed['acquire / timing'][0]).__name__}: {failed['acquire / timing'][0]}" if "acquire / timing" in failed \
                else "no symbol-rate line stood out: noise, analog, a burst too short, or not a linear modulation"
            p.add("acquire", "symbol rate and modulation", UNSOLVED, note=why)
            return p
        scores = sorted((det.scores or {}).items(), key=lambda t: -t[1])
        doubt = det.margin < 0.1
        # The rate itself: Acquire takes one after max_wait even when no rate line stood out, and a signal can't be
        # narrower than its symbol rate (the survey's width runs a little under it at small roll-offs)
        rate_doubt = []
        if det.line is not None and det.line < self.acq.lock:
            rate_doubt.append(f"no symbol-rate line stood out (the strongest {det.line:.1f}x its floor, {self.acq.lock:g}x "
                              "needed): the rate was taken anyway")
        if sig.width < RATE_WIDTH*det.baud:
            rate_doubt.append(f"the signal is {sig.width/1e3:.0f} kHz wide, narrower than this rate")
        # The fit: symbols far noisier than the band is strong weren't locked onto this constellation at this rate
        want, gap = esn0_gap(sig, det)
        fit_doubt = []
        if gap is not None and gap > ESN0_GAP:
            fit_doubt.append(f"Es/N0 {det.esn0:.1f} dB is {gap:.1f} dB under the {want:.1f} dB the band's SNR gives at this "
                             "rate: the symbols don't sit on this constellation (not linear, a wrong rate, one not tried "
                             "such as APSK, carriers on top of each other, bursts, or OFDM)")
        q = "" if det.esn0 is None else f", Es/N0 {det.esn0:.1f} dB"
        p.add("acquire", f"{det.mod.upper()} at {det.baud/1e6:.6g} MBd (sps {det.sps:.4f}), roll-off {det.beta}, "
                         f"CFO {det.cfo/1e3:+.2f} kHz{q}",
              detail={"mod": det.mod, "baud": det.baud, "sps": det.sps, "beta": det.beta, "cfo": det.cfo, "esn0": det.esn0,
                      "esn0_expected": want, "esn0_gap": gap, "scores": dict(scores), "margin": det.margin},
              note="; ".join((["rate doubtful: " + ", ".join(rate_doubt)] if rate_doubt else []) +
                             (["fit doubtful: " + ", ".join(fit_doubt)] if fit_doubt else []) +
                             [("doubtful: " if doubt else "") + "log-likelihood per symbol " +
                              ", ".join(f"{m} {s:.3f}" for m, s in scores)]))
        rate_doubt = rate_doubt + fit_doubt # Either way a weak code fitting these symbols is no evidence
        for stage in ("carrier", "fec"):
            if stage in failed:
                p.add(stage, "failed", UNSOLVED, note=f"{type(failed[stage][0]).__name__}: {failed[stage][0]}")
                return p
        p.add("timing", f"matrix matched filter, RRC {det.beta}, symbol instants tracked block by block")
        nth = {1: "1st", 2: "2nd", 3: "3rd"}.get(det.M, f"{det.M}th")
        p.add("carrier", f"feed-forward: {det.cfo/1e3:+.2f} kHz off, phase by the {nth} power over 64 symbols")
        fec = self.stream.est.get("fec")
        p.fec = fec
        c = symbol.load_or_generate(det.mod)
        weak = fec is not None and fec.found and rate_doubt and not _strong(fec)
        if fec is not None and fec.found and not weak:
            p.profile_name, p.profile_params = fec.profile_name, fec.profile_params
            _code_steps(p, fec, c)
            # Inner finds the steps don't hold: blind ones, and a configured RS code with nothing to align it by
            extra = [h for h in fec.inner if h.kind != "rs" or "pattern" not in h.params]
            if extra:
                _guess_steps(p, extra)
            elif not fec.complete:
                p.add("payload", "what the code carries", UNSOLVED,
                      note=f"the capture ended before enough symbols to look inside the code ({fec.n} used)")
            elif not fec.inner and not fec.profile_name:
                v = getattr(fec, "payload", None)
                note = ("no RS code (configured or blind), binary block code, scrambler or HDLC framing found inside"
                        if self.fec.blind else "nothing configured inside; AcquireFEC(blind=True) looks further")
                if v is not None and v.kind == "random-looking":
                    note = f"nothing inside: the bits it carries are random-looking ({v.why})"
                p.add("payload", "what the code carries", UNSOLVED, note=note)
            return p
        if fec is None or (not fec.found and not fec.complete and fec.report is None):
            p.add("fec", "still acquiring", UNSOLVED, note=f"{fec.n if fec else self.nsym} symbols: fewer than it needs")
            return p
        try:
            fr = framed(np.concatenate(self.raw), det) if self.raw else []
        except Exception as e:
            fr = []
            p.add("frame", "framed profiles", UNSOLVED, note=f"{type(e).__name__}: {e}")
        if fr:
            name, heads, n, inverted, params, ids = fr[0]
            p.profile_name, p.profile_params = name, params
            p.add("spectrum", "inverted: the symbols conjugated before the profile" if inverted else "normal",
                  note=f"the frame preamble turns {'backwards' if inverted else 'forwards'}, symbol to symbol")
            p.add("frame", f"{name}: {n} headers decoded")
            for hid, (sets, total) in heads.items():
                if sets[0][1] == 1: # Every one different (a counter, a length): how many, and one of them
                    p.add("header", f"{hid}: {total} decoded, all different, e.g. {sets[0][0]}")
                else:
                    p.add("header", f"{hid}: " + "; ".join(f"{n}x {fields}" for fields, n in sets))
            if ids and not any(h in heads for h in ids[-1]): # The frames were found and read, what they carry wasn't decoded
                p.add("payload", f"{name}: no {' or '.join(ids[-1])} decoded", UNSOLVED,
                      note=f"the frames' {', '.join(h for g in ids for h in g if h in heads)} headers read, but not one frame's payload "
                           "decoded: a mode the profile doesn't cover (the headers say which), or symbols too poor for its code")
            return p
        if weak:
            # At a wrong rate (or off the constellation) the symbols repeat or smear, and a weak code (short constraint,
            # low rate) can still fit them. Framed profiles were tried first: their headers are evidence of their own
            what = "the symbol rate" if any("wide" in d or "line" in d for d in rate_doubt) else "the symbol fit"
            p.add("fec", f"a code fits, but {what} is doubtful", UNSOLVED,
                  candidates=[(f"{fec.kind} {fec.name} on {fec.stream}" + (f" ({fec.mismatch*100:.2f}% mismatch, a wrong code "
                               f"{fec.baseline*100:.1f}%)" if fec.mismatch is not None else ""), float(fec.z or 0.0), 1.0)],
                  note=f"re-encoding is not far enough below a wrong code's to confirm the symbols: {'; '.join(rate_doubt)}")
            return p
        if fec.inner:
            _guess_steps(p, fec.inner)
            return p
        coded = False
        if not rate_doubt and not doubt and self.syms:
            coded = self._framing(p, det)
            if det.mod in ("8psk", "16qam", "16apsk") and self.syms and not coded:
                self._tcm(p, det)
        if rate_doubt and self.near:
            self._cyclic(p, det)
        more ="" if fec.complete else f"; the capture ended before the full window ({fec.n} symbols): more may still show it"
        cand = weights(fec.report.hyps if fec.report is not None else [])
        if not cand:
            why = "no hypothesis above chance"
        elif max(z for _, z, _ in cand) > fec.report.z_crit:
            why = (f"best of {fec.report.trials} hypotheses; none taken: below the threshold z>{fec.report.z_crit:.1f}, or "
                   "holding at every alignment (constant or repeated bits, as a carrier or a symbol rate locked too high gives, "
                   "are not a code)")
        else:
            why = f"best of {fec.report.trials} hypotheses, all below the threshold z>{fec.report.z_crit:.1f}"
        if fec.repeat is not None: # The same symbols again and again: more of them is no more evidence
            why = f"the symbols repeat every {fec.repeat} (fill, a test pattern or one frame sent again); {why}"
        if coded: # (the blind block code on the frames is the fec step: a guess, which the configured scans had none of)
            return p
        p.add("fec", "no definition fits", UNSOLVED, candidates=cand,
              note=why + more + ("; the symbol rate is doubtful" if rate_doubt else "") +
                   ("; the modulation itself is doubtful" if doubt else ""))
        return p
