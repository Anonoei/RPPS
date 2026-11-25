"""A signal profile's steps and how sure each is (rpps.analysis.signal_profile)"""
import json

import numpy as np

FOUND, GUESS, UNSOLVED = "found", "guess", "unsolved"
MIN_PAYLOAD = 64 # Payload symbols (frames together) under which a frame's content is not looked into

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
        from ..identify import _named
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
