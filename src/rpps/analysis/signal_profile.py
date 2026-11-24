"""A signal's profile: every step it takes to process it, as far as analysis can tell, and what it might be where it can't

    profiles = rp.analysis.profile(rp.source.File("ci16t", "capture.cs16", Fs=100e6))   # every signal the survey finds
    profiles = rp.analysis.profile(src, signals=[0, 3])                                   # or some of them
    for p in profiles:
        print(p)                       # numbered steps; an unsolved one ends the list with its top candidates
    p.solved, p.to_dict(), p.rx()      # whether every step is known, a JSON-able form, the receiver (when solved)

Each signal goes through the chain once: its channel (every signal's from one rp.signal.FFTBank: one FFT of the
capture a hop for all of them), Acquire (rate, modulation, offset), the
matrix matched filter and feed-forward carrier, then AcquireFEC (the code, its bit map and alignment, with the blind tools
when `blind`). When no code is found, shipped profiles that find their own frames (a preamble and a header: DVB-S2) are run
on the timed symbols. A step is "found" (a definition covers it), "guess" (a blind find: a definition once saved, see
rpps.analysis.identify.Guess) or "unsolved", which ends the profile with what it might be: hypotheses and their weight, a
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
    __slots__ = ("name", "signal", "steps", "fec", "det", "profile_name", "profile_params", "symbols")

    def __init__(self, name="", signal=None):
        self.name, self.signal = name, signal
        self.steps, self.fec, self.det, self.profile_name, self.symbols = [], None, None, None, 0
        self.profile_params = None # The values of the profile's params that make it (a variant), {} for its defaults

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
        return {"name": self.name, "solved": self.solved, "profile": self.profile_name, "params": self.profile_params,
                "symbols": self.symbols,
                "steps": [s.to_dict() for s in self.steps]}

    def __str__(self):
        mark = {FOUND: " ", GUESS: "?", UNSOLVED: "!"}
        from .identify import _named
        prof = _named(self.profile_name, self.profile_params) if self.profile_name else None
        head = self.name + (f"  -> {'profile ' + prof if prof else 'solved' if self.solved else 'UNSOLVED'}")
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
            return "demap", f"{spec.get('name') or spec.get('shape')} map {spec.get('mapping') or c.maps[0].comment} -> LLRs"
        if spec["symbol"] == "differential":
            return "differential", "phase change from the symbol before"
        return spec["symbol"], json.dumps({k: v for k, v in spec.items() if k != "symbol"})
    if "sync" in spec:
        kind = spec["sync"]
        if kind == "puncture":
            return "puncture", f"align on {spec['conv']}'s puncture period (found at bit {fec.offset} of {fec.period})"
        if kind == "align":
            pat = " ".join(f"{v:02X}" if isinstance(v, int) else str(v) for v in spec.get("pattern", spec.get("markers", [])))
            return "align", f"{spec.get('name', 'align')}: every {spec['period']} bytes starts with {pat}, polarity resolved"
        return kind, json.dumps({k: v for k, v in spec.items() if k != "sync"})
    if "code" in spec:
        if spec["code"] in ("conv", "block", "interleave"):
            return {"interleave": "deinterleave"}.get(spec["code"], "fec"), f"{spec['code']} {spec['name']}"
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
        if spec.get("code") == "conv" and fec.mismatch is not None:
            detail = {"mismatch": fec.mismatch, "baseline": fec.baseline, "z": fec.z}
            what += f"  (re-encodes with {fec.mismatch*100:.2f}% mismatch, a wrong code {fec.baseline*100:.1f}%)"
        p.add(kind, what, detail=detail)

def _guess_steps(p, hyps):
    """Finds the decoder doesn't hold into p: blind ones with their Guess (decodes once saved), and configured ones it
    could not place (an RS code with no bytes to align it by)"""
    for h in hyps:
        g = h.guess
        kind = {"conv": "fec", "conv-punctured": "fec", "block-blind": "fec", "rs-blind": "fec", "rs": "fec", "scrambler": "descramble",
                "scrambler-additive": "descramble", "hdlc": "frame", "frame-blind": "frame"}.get(h.kind, h.kind)
        p.add(kind, h.name, GUESS if g is not None or h.kind == "frame-blind" else FOUND,
              detail={"z": h.z, **{k: str(v) for k, v in h.params.items()}},
              note=f"blind: {g!r}" if g is not None else ("found, but nothing places it in the decoder" if h.kind == "rs" else None))

# --- Framed profiles (a preamble and headers: DVB-S2) ----------------------------------------------------------------

def framed(raw, det, skip=6000, max_symbols=400000, min_headers=3):
    """Shipped profiles with a frame sync step, run on timed symbols (the frequency offset taken off), conjugated first
    when the preamble turns the wrong way (`rotation`: a spectrum inverted on the way down, in which the coherent frame
    sync finds nothing): [(name,
    {header id: ([(fields, count)] commonest first, how many)}, headers, inverted, params, ids)] of those whose headers decode,
    ids: the profile's header step ids in order, a list each (the headers of a branch point's branches together): one of
    the last decoded says the frames' payload decoded too,
    those whose payload decoded first, then most headers first; each variant a profile's choices make
    (rp.profile.variants) is tried, `params` the one that decoded, of the profiles whose "requires" admits the symbol rate
    (rp.profile.candidates)"""
    from .. import profile, symbol
    from ..base import Data
    from ..base.types import Rep
    out = []
    z = np.asarray(raw)[skip:skip + max_symbols]
    if len(z) < 20000:
        return out
    z = z*np.exp(-2j*np.pi*det.cfo/det.baud*np.arange(len(z)))
    z = (z/np.sqrt(np.mean(np.abs(z)**2))).astype(np.complex64)
    # Only the profiles a signal at this rate can be ("requires"): one whose rate is set elsewhere is never built. Not by
    # length: min_samples is the longest frame's, and short frames show their headers in less
    for name, params in profile.candidates(baud=det.baud, input="baseband"):
        prof = profile.load(name, **params)
        header = next((s.get("header") for s in prof.obj["steps"] if s.get("sync") == "frame"), None)
        if header is None: # No frame sync, or one by a word (no header to read)
            continue
        pre = symbol.load_header(header).preamble
        inverted = rotation(z, pre)
        if inverted is None: # The preamble isn't there either way round
            continue
        zz = np.conj(z) if inverted else z
        # The chain's carrier offset is an M-th power's, which a constellation not M-fold symmetric (APSK) leaves off by
        # a good part of the rate: the rest from the preamble's own phase steps
        w = preamble_offset(zz, pre)
        if w:
            zz = (zz*np.exp(-1j*w*np.arange(len(zz)))).astype(np.complex64)
        data = Data(zz, sps=1, flush=True)
        try:
            res = prof.rx(start=Rep.SYMBOLS)(data)
        except Exception: # A profile that can't run on these symbols isn't this signal
            continue
        # A profile ending in branches gives {branch: output}: its trunk publishes in the input's stream, each
        # branch in its own
        streams = [data.stream] + [o.stream for o in (res.values() if isinstance(res, dict) else [res])]
        heads = {}
        for st in {id(s): s for s in streams}.values():
            for step, fields, at in (st.control.fields() if st.control is not None else []):
                heads.setdefault(step, []).append(json.dumps(fields, sort_keys=True, default=str))
        if max((len(v) for v in heads.values()), default=0) >= min_headers:
            summary = {k: ([(json.loads(f), n) for f, n in Counter(v).most_common(4)], len(v)) for k, v in heads.items()}
            ids = _header_ids(prof.obj["steps"])
            out.append((name, summary, max(len(v) for v in heads.values()), inverted, params, ids))
    # A profile whose payload decoded before one whose headers only read (DVB-S2 normal and mis share the PL header)
    return sorted(out, key=lambda t: (not (t[5] and any(h in t[1] for h in t[5][-1])), -t[2]))

def preamble_offset(z, preamble, threshold=0.6, top=16, sigmas=3.0):
    """The carrier offset left on symbols z, in radians a symbol (within +-pi), from where a known preamble is: the phase
    steps symbol to symbol, the preamble's own taken off, are the offset's (differential: the carrier phase doesn't
    matter). For symbols the preamble was found in (`rotation`); 0.0 when nothing correlates past `threshold`, and when
    the offset is under `sigmas` standard errors of the preambles' own estimates (no offset to see: a noise estimate,
    applied, drifts a long frame's phase by radians)"""
    pre = np.asarray(preamble, dtype=complex)
    ref = pre[1:]*np.conj(pre[:-1])
    d = np.asarray(z[1:])*np.conj(np.asarray(z[:-1]))
    d = d/(np.abs(d) + 1e-12)
    up = np.correlate(d, ref, "valid")/len(ref)
    size = np.abs(up)
    if not len(size) or size.max() <= threshold:
        return 0.0
    # (the preambles: near the strongest; data matches the preamble's steps by chance less well, and at any angle)
    peaks = np.nonzero(size > max(threshold, 0.75*size.max()))[0]
    best = peaks[np.argsort(-size[peaks])[:top]]
    w = float(np.angle(np.sum(up[best])))
    if len(best) > 1:
        spread = np.angle(up[best]*np.exp(-1j*w))
        if abs(w) < sigmas*np.sqrt(np.sum(spread**2)/(len(best) - 1)/len(best)):
            return 0.0
    return w

def _header_ids(steps):
    """A profile's header step ids (symbol or link headers) in order: [[id], ...], the headers of a branch point's
    branches one list"""
    out = []
    for s in steps:
        if s.get("id") and (s.get("symbol") == "header" or s.get("link") == "header"):
            out.append([s["id"]])
        br = s.get("branches")
        if br:
            inner = [h for b in (br.values() if isinstance(br, dict) else br)
                     for g in _header_ids(b.get("steps", []) if isinstance(b, dict) else b) for h in g]
            if inner:
                out.append(inner)
    return out

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
ESN0_GAP = 1.5 # dB: a linear carrier, locked, measures the Es/N0 its band's SNR gives to within about 1 dB (carriers on air
               # that decode cleanly come in 0.5-1 dB under; this leaves the band estimate a little room). Further under it,
               # the symbols don't sit on this constellation: not linear, a wrong rate, or carriers on top of each other

def esn0_gap(sig, det):
    """(expected Es/N0, gap) in dB: what the survey's in-band SNR (over the noise beside the signal) gives at the acquired
    rate (Signal.esn0_db), and how far the measured one falls under it; (None, None) when either is unknown"""
    if det.esn0 is None or getattr(sig, "snr", None) is None or not getattr(sig, "width", None):
        return None, None
    want = sig.esn0_db(det.baud)
    return want, want - det.esn0

def _strong(fec):
    """Whether a found code is evidence enough for the symbol rate too: one with more than a re-encoding (RS, HDLC, a
    block code, a matching shipped profile), or a convolutional code fitting far better than a wrong one"""
    if fec.kind != "conv" or fec.profile_name or fec.inner:
        return True
    return fec.mismatch is not None and fec.baseline and fec.mismatch < STRONG*fec.baseline

NEAR = 2**20 # Channel samples kept for rpps.analysis.spectrum.waveform
PM_OFFSET = 0.25 # A residual carrier this far (of the signal's width) from its centre isn't one: PM is symmetric about it
CARRIER_SHARE = 0.01 # Power share of the strongest spectral bin (of 8192) worth a residual-carrier analysis

def carrier_share(x, nfft=8192):
    """The strongest bin's share of the power in an averaged spectrum: a carrier line holds far more than noise-like
    modulation's 1/nfft or so (a quick screen before rpps.analysis.framing.telemetry)"""
    x = np.asarray(x)
    k = len(x)//nfft
    if k < 1:
        return 0.0
    P = np.mean(np.abs(np.fft.fft(x[:k*nfft].reshape(k, nfft), axis=1))**2, axis=0)
    return float(P.max()/P.sum())
SYMS = 2**18 # Tracked symbols kept for rpps.analysis.framing.frames

def _waveform_detail(w):
    d = {"envelope": w.envelope}
    if w.tones is not None:
        d["tones"] = [[f, sh] for f, sh in w.tones.lines]
    if w.ofdm is not None:
        d["ofdm"] = {"tu": w.ofdm.tu, "tcp": w.ofdm.tcp, "strength": w.ofdm.strength, "systems": [n for n, _ in w.ofdm.systems[:3]]}
    if w.bursts is not None:
        d["bursts"] = {"duty": w.bursts.duty, "period": w.bursts.period}
    return d

class _Chain:
    """A signal's chain over the blocks of its capture, and what it keeps for the profile"""
    def __init__(self, sig, Fs, blind, baud=None, ch=None, codes=None):
        """ch: the channel an rp.signal.FFTBank brings this signal to baseband in (its samples are given to `step` at
        ch.Fs); None: the chain channelizes the capture's samples itself (sig.channel())"""
        from .. import sync
        from ..base import Stream
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
        carriers a few tens of kHz apart). On a hand-reviewed pool: 65 of 73 boxes reviewers said held several signals
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
        from .spectrum.signal import survey
        from ..source import Memory
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
        from ..base import Data
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
        from .spectrum import cyclic
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
        from .framing import telemetry
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
        from .spectrum.waveform import classify
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
        """TDMA bursts (rpps.analysis.spectrum.burst.demodulate: each acquired on its kind's preamble) into q, and for each
        kind the code laid out from where its payload starts (linear.frame_units on every bit map: one rotation is
        common to all bursts and unknown), else what the payload bits are like"""
        from .spectrum import burst
        from .code.block.linear import frame_units, solve_block
        from .framing.payload import verdict
        from .identify import bit_streams, _distinct
        from .. import symbol
        try:
            r = burst.demodulate(x, self.ch.Fs)
        except Exception as e:
            q.add("demodulate", "bursts, each on its preamble", UNSOLVED, note=f"{type(e).__name__}: {e}")
            return
        ok = [b for b in r.bursts if b.ok and b.symbols is not None]
        q.add("demodulate", str(r), FOUND if ok else UNSOLVED, detail={"mod": r.mod, "baud": r.baud, "bursts": len(r.bursts),
              "demodulated": len(ok), "kinds": len(r.preambles or [r.preamble])},
              note="blind (rpps.analysis.spectrum.burst): each burst acquired on its own preamble")
        c = symbol.load_or_generate(r.mod)
        kinds = r.preambles or [r.preamble]
        for kind, pre in enumerate(kinds):
            group = [b for b in ok if (b.kind or 0) == kind and len(b.payload)]
            if len(group) < 4:
                continue
            pays = [np.asarray(b.payload, dtype=np.complex64) for b in group] # (blind bursts: each its own rotation)
            best = None
            for key in _distinct(bit_streams(pays[0], c, differential=False), c):
                rows = [bit_streams(y, c, differential=False, keys=[key])[key] for y in pays]
                n, d = frame_units(rows)
                if n is not None and (best is None or (n, -d) < (best[0], -best[1])):
                    best = (n, d, key, rows)
            label = f"kind {kind} ({len(pre)}-symbol preamble, {len(group)} bursts)" if len(kinds) > 1 else f"{len(group)} bursts"
            if best is None:
                key = next(iter(_distinct(bit_streams(pays[0], c, differential=False), c)))
                bits = np.concatenate([bit_streams(y, c, differential=False, keys=[key])[key] for y in pays])
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

    def _fsk(self, q, x):
        """A constant envelope: FSK / MSK / GMSK by the instantaneous frequency (rpps.analysis.modulation.fsk), else unsolved"""
        from .modulation import fsk
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
        from .framing import frames
        from .code.block.linear import solve_block, frame_units
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
        from .. import symbol
        from .modulation.quality import gate
        g = gate(cut[:400, fr.uw_length:].ravel(), symbol.load_or_generate(det.mod).points)
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
            from .identify import bit_streams, _distinct
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
        from .code.conv import tcm
        from .. import symbol
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
        from .spectrum import cyclic
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
        from .. import symbol
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
            if inverted:
                p.add("spectrum", "inverted: conjugate the symbols before the profile",
                      note="the frame preamble turns backwards, symbol to symbol")
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

def profile(source, signals=None, freq=None, blocks=450, block=16384, blind=True, survey=None, symbols=2**18,
            max_blocks=None, progress=None, jobs=1, hint=None):
    """Profiles of signals in a capture: [SignalProfile], in the order chosen

    source: an rp.source Source (File, Memory: a live device is best captured into Memory first, as the chains run slower
        than it does). It is read once, `blocks` blocks of `block` samples, every chosen signal processed along the way.
    symbols, max_blocks: a signal whose code is not decided yet and that has fewer than `symbols` symbols keeps reading
        past `blocks`, up to `max_blocks` (default 8 x blocks): a narrow signal gets symbols enough for the code scans
        (and the layers inside a code) instead of the few thousand a fixed stretch of a wide capture holds
    signals: which ones: None (every one the survey finds), an index or indices into the survey, or Signal objects
    freq: instead, the one nearest this frequency (Hz from the capture's centre)
    survey: the rp.analysis.survey of `source` already made (the indices refer to it; it isn't surveyed again)
    blind: run the blind tools where no definition fits (tens of seconds a signal)
    progress: called as progress(stage, done, total, left) as the work goes: stage "read" (blocks read of the stretch
        planned: `blocks`, then `max_blocks` once some chains read on; `left` the chains still reading), "build" (profiles
        built of the signals), and "again" / "build again" for a second pass at a rate the cyclic spectrum gives
    jobs: profile the signals in this many processes (forked, each with its share of the signals and its own reading of
        the source from where this one starts: a File seeks back, Memory starts over; Linux and macOS). With jobs > 1
        `progress` is called as ("parallel", groups done, jobs, groups left)
    hint: hint(signal) -> None or {"baud": the symbol rate, "codes": the convolutional definitions to try (names in
        config/fec/conv), "blind": False to skip the blind tools}: what an earlier look at the same signal (another
        capture of it) found, so it is not searched for again"""
    from .spectrum.signal import survey as _survey
    tell = progress or (lambda *a: None)
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
    if jobs > 1 and len(picked) > 1:
        return _parallel(source, picked, sv, jobs, tell, dict(blocks=blocks, block=block, blind=blind, symbols=symbols,
                                                               max_blocks=max_blocks, hint=hint))
    refit = _refit(source, picked) # (each process its own: the survey's boxes are what it is given)
    for old, new in zip(picked, refit):
        if new is not old:
            index[id(new)] = index.get(id(old))
    picked = refit

    cap = max(blocks, max_blocks if max_blocks is not None else 8*blocks)

    def chains_for(sigs, bauds):
        """Chains for these signals, their channels from one rp.signal.FFTBank (one FFT of the capture a hop for all
        of them), and which bank channel each reads"""
        from ..signal import FFTBank
        from .spectrum.signal import Channel
        live = [s for s in sigs if not getattr(s, "holds", 0)]
        bank = FFTBank(source.Fs, [(s.mid, s.width) for s in live]) if live else None
        slot, out = {}, []
        for s, b in zip(sigs, bauds):
            ch = None
            if bank is not None and not getattr(s, "holds", 0):
                j = len(slot)
                ch = Channel(None, bank.Fs_out[j], bank.D[j], 0.0, 0)
            h = (hint(s) if hint is not None else None) or {}
            c = _Chain(s, source.Fs, blind and h.get("blind", True), baud=b or h.get("baud"), ch=ch, codes=h.get("codes"))
            if ch is not None:
                slot[id(c)] = j
            out.append(c)
        return out, bank, slot

    def run(chains, bank, slot, stage="read"):
        fed = [ch for ch in chains if not ch.dead and id(ch) in slot]
        if bank is None:
            return
        def give(ch, y, last=False): # A chain's channel samples as the bank gives them (its stages don't mind the cut)
            if len(y) or last:
                ch.step(y, last=last)
        for i, blk in enumerate(source(block)):
            if i >= cap:
                break
            if i >= blocks: # Past the common stretch: only the chains still short of a decided code
                for ch in [ch for ch in fed if not ch.wants(symbols)]:
                    give(ch, np.zeros(0, np.complex64), last=True)
                    fed.remove(ch)
                if not fed:
                    return
            outs = bank.step(blk.data, active=[slot[id(ch)] for ch in fed])
            for ch in fed:
                give(ch, outs[slot[id(ch)]])
            tell(stage, i + 1, blocks if i < blocks else cap, len(fed))
        outs = bank.flush(active=[slot[id(ch)] for ch in fed]) # The end of the stream for those still reading
        for ch in fed:
            give(ch, outs[slot[id(ch)]], last=True)

    def build(ch):
        s = ch.sig
        j = index.get(id(s))
        name = f"signal {j if j is not None else '?'}: {s.mid/1e6:+.3f} MHz, {s.width/1e3:.0f} kHz wide"
        try:
            p = ch.profile(name)
        except Exception as e:
            p = SignalProfile(name, s)
            p.add("profile", "building the profile", UNSOLVED, note=f"{type(e).__name__}: {e}")
            return p
        if getattr(s, "refit", None) and p.steps:
            m0, w0 = s.refit
            p.steps.insert(1, Step("box", f"fitted to the signal: centre {(s.mid - m0)/1e3:+.1f} kHz, {s.width/1e3:.0f} kHz wide "
                                          f"(the survey's box {w0/1e3:.0f} kHz)", GUESS,
                                   note="a fine spectrum of the capture (its 95th percentile over time), edges 10 dB under the top"))
        return p

    chains, bank, slot = chains_for(picked, [None]*len(picked))
    run(chains, bank, slot)
    out = []
    for k, ch in enumerate(chains):
        out.append(build(ch))
        tell("build", k + 1, len(chains), len(chains) - k - 1)
    # Where the chain found no rate (or a doubtful one) and the cyclic spectrum sees one: a second pass at that rate
    again, parts, extra = {}, {}, []
    notched = set() # Boxes split at notches a fine spectrum shows: carriers, each listed
    for k, (ch, p) in enumerate(zip(chains, out)):
        subs = ch.carriers() if not ch.dead else []
        if subs:
            parts[k] = [(s, None) for s in subs]
            notched.add(k)
            continue
        baud = ch.cyclic_rate(p) if not p.solved else None
        if baud:
            subs = ch.split(baud) # A box holding carriers side by side: each of them on its own (its rate acquired
            if len(subs) > 1 or (subs and subs[0].width < 0.7*ch.sig.width): # alone in its channel: the box's line
                parts[k] = [(s, None) for s in subs] # can be a beat between them), or the one carrier a too-wide box holds
            elif not (ch.stream.det is not None and abs(baud - ch.stream.det.baud) < 0.01*baud): # (not the same again)
                again[k] = (ch.sig, baud)
    todo = list(again.values()) + [sb for v in parts.values() for sb in v]
    if todo:
        redo, bank2, slot2 = chains_for([sb[0] for sb in todo], [sb[1] for sb in todo])
        run(redo, bank2, slot2, stage="again")
        split_chains = redo[len(again):]
        again = dict(zip(again, redo[:len(again)]))
        i = 0
        for k, subs in parts.items():
            mine = split_chains[i:i + len(subs)]
            i += len(subs)
            built = [build(c) for c in mine]
            what = (f"one of {len(subs)} carriers the survey's box held side by side" if len(subs) > 1 else
                    "the one carrier in a survey box wider than it")
            why = ("a fine spectrum of the box shows notches between them" if k in notched else
                   "its cyclic spectrum's rate needs well under the box")
            for q, c in zip(built, mine):
                q.steps.insert(1, Step("split", f"{what} ({why})", GUESS))
            box = out[k].name.split(":")[0] # "signal 16": each carrier in it is "signal 16.1", "signal 16.2"...
            for j, q in enumerate(built):
                q.name = f"{box}.{j + 1}: {q.signal.mid/1e6:+.3f} MHz, {q.signal.width/1e3:.0f} kHz wide"
            best = max(range(len(built)), key=lambda j: _reach(built[j]))
            first = out[k]
            # A box the notches split is not a signal itself, unless it solved as one and no piece did (a dip in one
            # carrier's spectrum)
            cut = k in notched and not (first.solved and not any(q.solved for q in built))
            if cut or _further(built[best], first):
                out[k] = built[best]
            # The rest, where they solved or went further than the box did (a piece that got nowhere is a lump of a
            # carrier's spectrum, or one the box's own profile already said as much of)
            # (carriers a notch separates are each a signal: all listed)
            extra += [q for j, q in enumerate(built) if out[k] is not q and (cut or q.solved or _further(q, first))]
        for j, (k, ch) in enumerate(again.items()):
            q = build(ch)
            tell("build again", j + 1, len(again), len(again) - j - 1)
            if _further(q, out[k]):
                q.steps.insert(1, Step("rate", f"{ch.baud/1e3:.6g} kBd from the cyclic spectrum (rpps.analysis.spectrum.cyclic)",
                                       GUESS, note="the first pass found " + (out[k].steps[1].what if len(out[k].steps) > 1
                                                                              else "nothing")))
                out[k] = q
    return out + extra

_JOB = None # What a forked worker of `_parallel` profiles (inherited, not pickled)

def _refit(source, sigs, samples=2**21, nfft=2**18, drop=10.0, q=95, move=0.08, widths=(0.8, 1.3)):
    """The survey's boxes fitted to their signals in a fine spectrum of the capture: `samples` read once (source.peek),
    the `q`th percentile over time of an `nfft`-bin spectrum (bursts and TDD carriers at their full level), each box's
    edges where it falls `drop` dB under the box's top (at most halfway down to the floor around it, a weak carrier's),
    walking out from its strongest point; a box under 6 dB over that floor keeps the survey's. The survey's own spectrum is
    short and coarse: it places some boxes off their signal (wider, narrower, beside it). A fit that runs out of the
    span looked at (1.5 box widths each side: the signal is part of something wider) or changes the width more than 3x
    keeps the survey's box; so does one that moves the centre under `move` of the width with the width within `widths`
    times (the two measure a carrier differently: occupied power, edges). Returns the boxes, refitted ones as copies
    (`refit` set). On a hand-reviewed pool: 17 of 24 boxes reviewers placed off their one signal moved the way they said
    (2 the other way), 3 of 325 they said were right moved far, and box width over symbol rate went from 0.98-1.35
    (10-90%) to 1.15-1.19: a carrier's -10 dB width, whatever share of its power the survey's box held"""
    import copy
    if not sigs or not hasattr(source, "peek"):
        return list(sigs)
    try:
        x = np.asarray(source.peek(samples))
    except Exception:
        return list(sigs)
    n = 1 << int(np.log2(max(16, min(nfft, len(x)//4))))
    if len(x) < 4*n or n < 4096:
        return list(sigs)
    Fs = float(source.Fs)
    seg = x[:len(x)//n*n].reshape(-1, n)[:16]
    w = np.hanning(n).astype(np.float32)
    P = np.empty((len(seg), n), np.float32)
    for j, v in enumerate(seg):
        P[j] = np.fft.fftshift(np.abs(np.fft.fft(v*w))**2)
    pk = 10*np.log10(np.percentile(P, q, axis=0) + 1e-30)
    del P
    f = np.fft.fftshift(np.fft.fftfreq(n, 1/Fs))
    out = []
    for s in sigs:
        if getattr(s, "holds", 0) or not getattr(s, "width", None):
            out.append(s)
            continue
        span = np.flatnonzero(np.abs(f - s.mid) <= 1.5*s.width)
        inb = np.flatnonzero(np.abs(f - s.mid) <= s.width/2)
        if len(inb) < 8:
            out.append(s)
            continue
        k = max(1, len(span)//200)
        p = np.convolve(pk[span], np.ones(k)/k, mode="same")
        fs = f[span]
        box = (np.abs(fs - s.mid) <= s.width/2)
        top_i = int(np.flatnonzero(box)[np.argmax(p[box])])
        # Down `drop` dB, but not into the floor (the span's lowest): a weak carrier's edges are halfway down to it
        hi90, floor = np.percentile(p[box], 90), np.percentile(p, 10)
        if hi90 - floor < 6.0: # (too weak over the floor to place its edges)
            out.append(s)
            continue
        thr = max(hi90 - drop, floor + 0.5*(hi90 - floor))
        a = top_i
        while a > 0 and p[a - 1] > thr:
            a -= 1
        b = top_i
        while b < len(p) - 1 and p[b + 1] > thr:
            b += 1
        mid, width = 0.5*(fs[a] + fs[b]), fs[b] - fs[a]
        rest = box.copy()
        rest[a:b + 1] = False
        # More in the box than this signal (carriers side by side: the notch split's to cut, not this to drop)
        others = np.count_nonzero(p[rest] > thr) > 0.1*np.count_nonzero(box)
        wide = width > 2*s.width and abs(mid - s.mid) > 0.2*s.width # (widened far and moved: into a neighbour)
        if others or wide or a == 0 or b == len(p) - 1 or not (s.width/3 <= width <= 3*s.width) or \
           (abs(mid - s.mid) < move*s.width and widths[0] <= width/s.width <= widths[1]):
            out.append(s)
            continue
        r = copy.copy(s)
        r.mid, r.width = float(mid), float(width)
        r.lo, r.hi = r.mid - r.width/2, r.mid + r.width/2
        r.refit = (s.mid, s.width)
        out.append(r)
    return out

def _parallel(source, picked, sv, jobs, tell, kw):
    """profile() of `picked` in `jobs` forked processes: the signals dealt round robin, results in their order"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    global _JOB
    jobs = min(jobs, len(picked))
    groups = [picked[k::jobs] for k in range(jobs)]
    _JOB = (source, getattr(source, "cur_samp", None), groups, sv, kw)
    out = {}
    try:
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
            futs = {ex.submit(_parallel_part, k): k for k in range(jobs)}
            for done, f in enumerate(as_completed(futs), 1):
                for s, p in zip(groups[futs[f]], f.result()):
                    out[id(s)] = p
                tell("parallel", done, jobs, jobs - done)
    finally:
        _JOB = None
    return [out[id(s)] for s in picked]

def _parallel_part(k):
    source, start, groups, sv, kw = _JOB
    if start is not None and hasattr(source, "seek"):
        source.seek(start)
    return profile(source, signals=groups[k], survey=sv, **kw)

def _reach(x):
    """How far a profile got: solved, a profile, steps found before the first unsolved one"""
    n = next((i for i, st in enumerate(x.steps) if st.status == UNSOLVED), len(x.steps))
    return (x.solved, bool(x.profile_name), n)

def _further(q, p):
    """Whether profile q got further than p (`_reach`)"""
    return _reach(q) > _reach(p)
