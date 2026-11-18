"""A signal's profile: every step it takes to process it, as far as analysis can tell, and what it might be where it can't

    profiles = rp.analysis.profile(rp.source.File("ci32t", "capture.cs16", Fs=100e6))   # every signal the survey finds
    profiles = rp.analysis.profile(src, signals=[0, 3])                                   # or some of them
    for p in profiles:
        print(p)                       # numbered steps; an unsolved one ends the list with its top candidates
    p.solved, p.to_dict(), p.rx()      # whether every step is known, a JSON-able form, the receiver (when solved)

Each signal goes through the chain once: its channel (shift and decimation), Acquire (rate, modulation, offset), the
matrix matched filter and feed-forward carrier, then AcquireFEC (the code, its bit map and alignment, with the blind tools
when `blind`). When no code is found, shipped profiles that find their own frames (a preamble and a header: DVB-S2) are run
on the timed symbols. A step is "found" (a definition covers it), "guess" (a blind find: a definition once saved, see
rpps.analysis.fec.Guess) or "unsolved", which ends the profile with what it might be: hypotheses and their weight, a
softmax over their z-scores (a clear leader near 100%, close calls shared).
"""
import json
import traceback
from collections import Counter

import numpy as np

FOUND, GUESS, UNSOLVED = "found", "guess", "unsolved"

class Step:
    """One processing step. kind: what it does (channel, acquire, timing, carrier, map, fec, align, deinterleave,
    descramble, frame, ...). what: how, in words. candidates: [(name, z, weight)] for an unsolved step"""
    __slots__ = ("kind", "what", "status", "detail", "candidates", "note")

    def __init__(self, kind, what, status=FOUND, detail=None, candidates=None, note=None):
        self.kind, self.what, self.status = kind, what, status
        self.detail = detail or {}
        self.candidates = candidates or []
        self.note = note

    def to_dict(self):
        return {"kind": self.kind, "what": self.what, "status": self.status, "detail": self.detail,
                "candidates": [{"name": n, "z": z, "weight": w} for n, z, w in self.candidates], "note": self.note}

def _short(h):
    """A hypothesis in a line: what it is, then where and how well"""
    p = h.params
    bits = []
    if "g" in p:
        bits.append("generators " + " ".join(f"{int(''.join(map(str, g)), 2):o}" for g in p["g"].values()))
    for k in ("n", "k", "K", "offset", "mismatch", "baseline", "words_ok", "polarity", "interleaver"):
        if k in p and p[k] is not None and not (k == "n" and "K" in p):
            v = p[k]
            bits.append(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}")
    where = f" on {p['stream']}" if "stream" in p else ""
    return f"{h.kind} {h.name}{where}" + (f" ({', '.join(bits)})" if bits else "")

def weights(hyps, top=5):
    """[(name, z, weight)] of the best hypotheses, one per code (the same code on several bit maps is one candidate, at its
    best): weight is a softmax over their z"""
    best = {}
    for h in sorted(hyps, key=lambda h: -h.z):
        p = h.params
        key = (h.kind, h.name, str(p.get("g")), str(p.get("n")), str(h.coding))
        best.setdefault(key, h)
    hyps = list(best.values())[:top]
    if not hyps:
        return []
    z = np.array([h.z for h in hyps])
    w = np.exp(z - z.max())
    w = w/w.sum()
    return [(_short(h), float(h.z), float(x)) for h, x in zip(hyps, w)]

class SignalProfile:
    """The steps that process one signal (see the module). solved: every step found or guessed, none unsolved"""
    __slots__ = ("name", "signal", "steps", "fec", "det", "profile_name", "symbols")

    def __init__(self, name="", signal=None):
        self.name, self.signal = name, signal
        self.steps, self.fec, self.det, self.profile_name, self.symbols = [], None, None, None, 0

    def add(self, *a, **kw):
        self.steps.append(Step(*a, **kw))
        return self.steps[-1]

    @property
    def solved(self):
        return bool(self.steps) and all(s.status != UNSOLVED for s in self.steps)

    @property
    def unsolved(self):
        """The first unsolved step, or None"""
        return next((s for s in self.steps if s.status == UNSOLVED), None)

    def rx(self):
        """The receiver from the symbols rp.sync.AcquireFEC gives (a profile runner), when a definition decodes it"""
        if self.fec is None or not self.fec.found:
            raise ValueError(f"no definition decodes this signal: {self.unsolved.what if self.unsolved else 'framed profile'}")
        return self.fec.rx()

    def to_dict(self):
        return {"name": self.name, "solved": self.solved, "profile": self.profile_name, "symbols": self.symbols,
                "steps": [s.to_dict() for s in self.steps]}

    def __str__(self):
        mark = {FOUND: " ", GUESS: "?", UNSOLVED: "!"}
        head = self.name + (f"  -> {'profile ' + self.profile_name if self.profile_name else 'solved' if self.solved else 'UNSOLVED'}")
        out = [head]
        w = max([len(s.kind) for s in self.steps] + [4])
        for i, s in enumerate(self.steps, 1):
            out.append(f"{i:3d}.{mark[s.status]} {s.kind:<{w}}  {s.what}")
            if s.note:
                out.append(f"      {'':<{w}}  ({s.note})")
            for name, z, wt in s.candidates:
                out.append(f"      {'':<{w}}  {wt*100:5.1f}%  z={z:6.1f}  {name}")
        return "\n".join(out)

    __repr__ = __str__

# --- Describing the found steps ---------------------------------------------------------------------------------------

def _describe(spec, fec, c):
    """(kind, words) for one profile step spec, past the timing and carrier"""
    if "symbol" in spec:
        if spec["symbol"] == "constellation":
            return "demap", f"{spec['name']} map {spec.get('mapping') or c.maps[0].comment} -> LLRs"
        if spec["symbol"] == "differential":
            return "differential", "phase change from the symbol before"
        return spec["symbol"], json.dumps({k: v for k, v in spec.items() if k != "symbol"})
    if "sync" in spec:
        kind = spec["sync"]
        if kind == "puncture":
            return "puncture", f"align on {spec['fec']}'s puncture period (found at bit {fec.offset} of {fec.period})"
        if kind == "align":
            pat = " ".join(f"{v:02X}" if isinstance(v, int) else str(v) for v in spec.get("pattern", spec.get("markers", [])))
            return "align", f"{spec.get('name', 'align')}: every {spec['period']} bytes starts with {pat}, polarity resolved"
        return kind, json.dumps({k: v for k, v in spec.items() if k != "sync"})
    if "code" in spec:
        if spec["code"] == "fec":
            kinds = {"conv": "fec", "block": "fec", "interleave": "deinterleave"}
            return kinds.get(spec["type"], spec["type"]), f"{spec['type']} {spec['name']}"
        if spec["code"] == "scramble":
            return "descramble", spec["name"]
        return spec["code"], json.dumps({k: v for k, v in spec.items() if k != "code"})
    if "link" in spec:
        return spec["link"], spec.get("name", "")
    if "wave" in spec:
        return "wave", spec["wave"]
    return "step", json.dumps(spec)

def _code_steps(p, fec, c):
    """The code's steps (bit map, demapping, decoding...) into p"""
    th, flip = fec.correction or (0.0, False)
    p.add("map", f"turn {np.degrees(th):.0f} deg{', conjugate' if flip else ''} to {fec.target.comment}"
          + (f" (or {len(fec.equivalent)} equivalent: inverted bits, left to what follows)" if fec.equivalent else ""))
    for spec in fec.steps:
        if "sync" in spec and spec["sync"] in ("timing", "carrier"):
            continue
        if "wave" in spec:
            continue
        kind, what = _describe(spec, fec, c)
        detail = {}
        if spec.get("code") == "fec" and spec.get("type") == "conv" and fec.mismatch is not None:
            detail = {"mismatch": fec.mismatch, "baseline": fec.baseline, "z": fec.z}
            what += f"  (re-encodes with {fec.mismatch*100:.2f}% mismatch, a wrong code {fec.baseline*100:.1f}%)"
        p.add(kind, what, detail=detail)

def _guess_steps(p, hyps):
    """Finds the decoder doesn't hold into p: blind ones with their Guess (decodes once saved), and configured ones it
    could not place (an RS code with no bytes to align it by)"""
    for h in hyps:
        g = h.guess
        kind = {"conv-punctured": "fec", "block-blind": "fec", "rs-blind": "fec", "rs": "fec", "scrambler": "descramble",
                "hdlc": "frame"}.get(h.kind, h.kind)
        p.add(kind, h.name, GUESS if g is not None else FOUND, detail={"z": h.z, **{k: str(v) for k, v in h.params.items()}},
              note=f"blind: {g!r}" if g is not None else ("found, but nothing places it in the decoder" if h.kind == "rs" else None))

# --- Framed profiles (a preamble and headers: DVB-S2) ----------------------------------------------------------------

def framed(raw, det, skip=6000, max_symbols=400000, min_headers=3):
    """Shipped profiles with a frame sync step, run on timed symbols (the frequency offset taken off), conjugated first
    when the preamble turns the wrong way (`rotation`: a spectrum inverted on the way down, in which the coherent frame
    sync finds nothing): [(name,
    {header id: ([(fields, count)] commonest first, how many)}, headers, inverted)] of those whose headers decode, best
    first"""
    from .. import profile, symbol
    from ..base import Data
    from ..base.types import Rep
    out = []
    z = np.asarray(raw)[skip:skip + max_symbols]
    if len(z) < 20000:
        return out
    z = z*np.exp(-2j*np.pi*det.cfo/det.baud*np.arange(len(z)))
    z = (z/np.sqrt(np.mean(np.abs(z)**2))).astype(np.complex64)
    for fam, modes in profile.ls().items():
        for mode in modes:
            name = f"{fam}/{mode}"
            obj = profile.load(name).obj
            if not any(s.get("sync") == "frame" for s in obj["steps"]):
                continue
            header = next(s["header"] for s in obj["steps"] if s.get("sync") == "frame")
            inverted = rotation(z, symbol.load_header(header).preamble)
            if inverted is None: # The preamble isn't there either way round
                continue
            try:
                res = profile.load(name).rx(start=Rep.SYMBOLS)(Data(np.conj(z) if inverted else z, sps=1, flush=True))
            except Exception: # A profile that can't run on these symbols isn't this signal
                continue
            ctl = res.stream.control
            if ctl is None:
                continue
            heads = {}
            for step, fields, at in ctl.fields():
                heads.setdefault(step, []).append(json.dumps(fields, sort_keys=True, default=str))
            if max((len(v) for v in heads.values()), default=0) >= min_headers:
                summary = {k: ([(json.loads(f), n) for f, n in Counter(v).most_common(4)], len(v)) for k, v in heads.items()}
                out.append((name, summary, max(len(v) for v in heads.values()), inverted))
    return sorted(out, key=lambda t: -t[2])

def rotation(z, preamble, threshold=0.6, top=8):
    """Whether the spectrum is inverted, from the way a known preamble turns symbol to symbol: True (inverted), False, or
    None (no preamble found)

    The phase step between symbols, with the preamble's own steps taken off, is the same either way round in size and
    opposite in sign: conjugating turns every step backwards. A preamble that turns as it goes (pi/2-BPSK: +90 degrees a
    symbol, DVB-S2's SOF) so shows which way round the spectrum is, which a constellation symmetric under conjugation
    (every PSK) and an unknown carrier offset can't. Differential, so a carrier offset or phase doesn't matter. Peaks are
    where the size of the correlation passes `threshold`; the sign of the `top` strongest decides."""
    pre = np.asarray(preamble, dtype=complex)
    ref = pre[1:]*np.conj(pre[:-1])
    if np.allclose(ref.imag, 0): # Steps of 0 or 180 degrees only (BPSK): the same conjugated, nothing to tell by
        return None
    d = np.asarray(z[1:])*np.conj(np.asarray(z[:-1]))
    d = d/(np.abs(d) + 1e-12)
    # np.correlate conjugates its second argument: up = sum d*conj(ref) is real and near 1 where the steps turn as the
    # preamble's do, down = sum d*ref where they turn the other way (each its size, and the sign is what tells them apart)
    up = np.correlate(d, ref, "valid")/len(ref)
    down = np.correlate(d, np.conj(ref), "valid")/len(ref)
    size = np.maximum(np.abs(up), np.abs(down)) # (the same for steps of +-90 degrees, where conj(ref) = -ref)
    peaks = np.nonzero(size > threshold)[0]
    if not len(peaks):
        return None
    best = peaks[np.argsort(-size[peaks])[:top]]
    return bool(np.sum(down[best].real) > np.sum(up[best].real))

# --- One signal's chain ----------------------------------------------------------------------------------------------

RATE_WIDTH = 0.85 # A signal narrower than this times its symbol rate wasn't timed at its rate
STRONG = 0.1      # A code re-encoding under this times a wrong code's mismatch confirms the rate it was found at

def _strong(fec):
    """Whether a found code is evidence enough for the symbol rate too: one with more than a re-encoding (RS, HDLC, a
    block code, a matching shipped profile), or a convolutional code fitting far better than a wrong one"""
    if fec.kind != "conv" or fec.profile_name or fec.inner:
        return True
    return fec.mismatch is not None and fec.baseline and fec.mismatch < STRONG*fec.baseline

class _Chain:
    """A signal's chain over the blocks of its capture, and what it keeps for the profile"""
    def __init__(self, sig, Fs, blind):
        from .. import sync
        from ..base import Stream
        self.sig, self.Fs = sig, Fs
        self.errors = []
        self.acq, self.mat, self.ff = sync.Acquire(), sync.time.Mat(), sync.phase.FFCarrier()
        self.fec = sync.AcquireFEC(blind=blind)
        self.raw, self.nsym, self.dead = [], 0, False
        try:
            self.ch = sig.channel()
            self.stream = Stream(Fs)
        except Exception as e:
            self.errors.append(("channel", e, traceback.format_exc(limit=4)))
            self.dead = True

    def step(self, samps):
        if self.dead:
            return
        from ..base import Data
        stage = "channel"
        try:
            near = Data(samps, Fs=self.Fs, stream=self.stream) + self.ch.front
            stage = "acquire / timing"
            mat = near + self.acq + self.mat
            if sum(len(r) for r in self.raw) < 6000 + 400000:
                self.raw.append(mat.data.copy())
            stage = "carrier"
            syms = mat + self.ff
            self.nsym += len(syms)
            stage = "fec"
            syms + self.fec
        except Exception as e:
            self.errors.append((stage, e, traceback.format_exc(limit=4)))
            self.dead = True

    def profile(self, name):
        from .. import symbol
        sig = self.sig
        p = SignalProfile(name, sig)
        p.symbols = self.nsym
        ch = getattr(self, "ch", None)
        snr = f", SNR {sig.snr:.1f} dB in band" if getattr(sig, "snr", None) is not None else ""
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
        q = "" if det.esn0 is None else f", Es/N0 {det.esn0:.1f} dB"
        p.add("acquire", f"{det.mod.upper()} at {det.baud/1e6:.6g} MBd (sps {det.sps:.4f}), roll-off {det.beta}, "
                         f"CFO {det.cfo/1e3:+.2f} kHz{q}",
              detail={"mod": det.mod, "baud": det.baud, "sps": det.sps, "beta": det.beta, "cfo": det.cfo, "esn0": det.esn0,
                      "scores": dict(scores), "margin": det.margin},
              note="; ".join(["rate doubtful: " + ", ".join(rate_doubt)] if rate_doubt else []) +
                   ("; " if rate_doubt else "") + ("doubtful: " if doubt else "") +
                   "log-likelihood per symbol " + ", ".join(f"{m} {s:.3f}" for m, s in scores))
        for stage in ("carrier", "fec"):
            if stage in failed:
                p.add(stage, "failed", UNSOLVED, note=f"{type(failed[stage][0]).__name__}: {failed[stage][0]}")
                return p
        p.add("timing", f"matrix matched filter, RRC {det.beta}, symbol instants tracked block by block")
        nth = {1: "1st", 2: "2nd", 3: "3rd"}.get(det.M, f"{det.M}th")
        p.add("carrier", f"feed-forward: {det.cfo/1e3:+.2f} kHz off, phase by the {nth} power over 64 symbols")
        fec = self.stream.est.get("fec")
        p.fec = fec
        c = symbol.load(det.mod)
        if fec is not None and fec.found and rate_doubt and not _strong(fec):
            # At a wrong rate the symbols repeat or smear, and a weak code (short constraint, low rate) can still fit them
            p.add("fec", "a code fits, but the symbol rate is doubtful", UNSOLVED,
                  candidates=[(f"{fec.kind} {fec.name} on {fec.stream}" + (f" ({fec.mismatch*100:.2f}% mismatch, a wrong code "
                               f"{fec.baseline*100:.1f}%)" if fec.mismatch is not None else ""), float(fec.z or 0.0), 1.0)],
                  note=f"re-encoding is not far enough below a wrong code's to confirm the rate: {'; '.join(rate_doubt)}")
            return p
        if fec is not None and fec.found:
            p.profile_name = fec.profile_name
            _code_steps(p, fec, c)
            # Inner finds the steps don't hold: blind ones, and a configured RS code with nothing to align it by
            extra = [h for h in fec.inner if h.kind != "rs" or "pattern" not in h.params]
            if extra:
                _guess_steps(p, extra)
            elif not fec.complete:
                p.add("payload", "what the code carries", UNSOLVED,
                      note=f"the capture ended before enough symbols to look inside the code ({fec.n} used)")
            elif not fec.inner and not fec.profile_name:
                p.add("payload", "what the code carries", UNSOLVED,
                      note="no RS code (configured or blind), binary block code, scrambler or HDLC framing found inside"
                      if self.fec.blind else "nothing configured inside; AcquireFEC(blind=True) looks further")
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
            name, heads, n, inverted = fr[0]
            p.profile_name = name
            if inverted:
                p.add("spectrum", "inverted: conjugate the symbols before the profile",
                      note="the frame preamble turns backwards, symbol to symbol")
            p.add("frame", f"{name}: {n} headers decoded")
            for hid, (sets, total) in heads.items():
                if sets[0][1] == 1: # Every one different (a counter, a length): how many, and one of them
                    p.add("header", f"{hid}: {total} decoded, all different, e.g. {sets[0][0]}")
                else:
                    p.add("header", f"{hid}: " + "; ".join(f"{n}x {fields}" for fields, n in sets))
            return p
        if fec.inner:
            _guess_steps(p, fec.inner)
            return p
        more = "" if fec.complete else f"; the capture ended before the full window ({fec.n} symbols): more may still show it"
        cand = weights(fec.report.hyps if fec.report is not None else [])
        if not cand:
            why = "no hypothesis above chance"
        elif max(z for _, z, _ in cand) > fec.report.z_crit:
            why = (f"best of {fec.report.trials} hypotheses; none taken: below the threshold z>{fec.report.z_crit:.1f}, or "
                   "holding at every alignment (constant or repeated bits, as a carrier or a symbol rate locked too high gives, "
                   "are not a code)")
        else:
            why = f"best of {fec.report.trials} hypotheses, all below the threshold z>{fec.report.z_crit:.1f}"
        p.add("fec", "no definition fits", UNSOLVED, candidates=cand,
              note=why + more + ("; the symbol rate is doubtful" if rate_doubt else "") +
                   ("; the modulation itself is doubtful" if doubt else ""))
        return p

def profile(source, signals=None, freq=None, blocks=450, block=16384, blind=True, survey=None):
    """Profiles of signals in a capture: [SignalProfile], in the order chosen

    source: an rp.source Source (File, Memory: a live device is best captured into Memory first, as the chains run slower
        than it does). It is read once, `blocks` blocks of `block` samples, every chosen signal processed along the way.
    signals: which ones: None (every one the survey finds), an index or indices into the survey, or Signal objects
    freq: instead, the one nearest this frequency (Hz from the capture's centre)
    survey: the rp.analysis.survey of `source` already made (the indices refer to it; it isn't surveyed again)
    blind: run the blind tools where no definition fits (tens of seconds a signal)"""
    from .signal import survey as _survey
    sv = survey if survey is not None else None
    if signals is not None and not isinstance(signals, (int, np.integer)) and all(hasattr(s, "channel") for s in signals):
        picked = list(signals)
        sv = sv if sv is not None else picked
    else:
        sv = sv if sv is not None else list(_survey(source))
        if freq is not None:
            picked = [min(sv, key=lambda s: abs(s.mid - freq))] if len(sv) else []
        elif signals is None:
            picked = list(sv)
        else:
            idx = [signals] if isinstance(signals, (int, np.integer)) else list(signals)
            picked = [sv[i] for i in idx]
    index = {id(s): i for i, s in enumerate(sv)}
    chains = [_Chain(s, source.Fs, blind) for s in picked]
    for i, blk in enumerate(source(block)):
        if i >= blocks:
            break
        for ch in chains:
            ch.step(blk.data)
    out = []
    for ch in chains:
        s = ch.sig
        j = index.get(id(s))
        name = f"signal {j if j is not None else '?'}: {s.mid/1e6:+.3f} MHz, {s.width/1e3:.0f} kHz wide"
        try:
            out.append(ch.profile(name))
        except Exception as e:
            p = SignalProfile(name, s)
            p.add("profile", "building the profile", UNSOLVED, note=f"{type(e).__name__}: {e}")
            out.append(p)
    return out
