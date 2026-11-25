"""What identify found: CodeDetection (rpps.analysis.identify)"""
import numpy as np

class CodeDetection:
    """Result of `identify`: the code a symbol stream carries, and how to decode it. Stored in stream.est["fec"]

    found: a code was identified (complete: and nothing more would come of more symbols)
    kind, name: the outer code ("conv", "block", "hdlc") and its config name; inner: hypotheses found inside it (RS: in
        `steps` when the bytes that start its codewords were found too, which Align needs)
    stream: the bit stream that decodes (`bit_streams` key); equivalent: others that decode as well (a code that can't
        tell inverted bits apart leaves that to what's inside it: Align's polarity)
    offset, period: where in the window the code's structure starts (bits) and how often it repeats
    steps: profile steps from the timed symbols on; profile_name: the shipped profile that has the same ones, and
        profile_params the values of its params that make them (a variant: rp.profile.variants)
    target: the map the decoder uses; correction: (angle, conj) that turns the symbols to it (rp.sync.AcquireFEC applies it)
    mismatch, baseline: re-encoding mismatch of a convolutional code, and what a wrong hypothesis gets
    slips: carrier turns and symbol slips rp.sync.AcquireFEC caught and put right; latency: seconds it held the symbols for
    report: the ranked hypotheses of the tier that decided (none did: of every tier)
    repeat: the period (symbols) the stream repeats with (`repeat_period`), when it does
    guesses: with blind, the `Guess`es for what no definition covers (a code, an RS code or a scrambler inside a convolutional
        code, see `solve`), outermost first; they are in `inner` but never in `steps`: save one to decode with it
    """
    __slots__ = ("found", "complete", "mod", "kind", "name", "stream", "equivalent", "offset", "period", "inner",
                 "steps", "profile_name", "profile_params", "target", "correction", "z", "mismatch", "baseline", "report", "guesses", "n",
                 "slips", "latency", "payload", "repeat")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.found = bool(self.found)
        self.complete = bool(self.complete)
        self.inner = self.inner or []
        self.equivalent = self.equivalent or []
        self.slips = self.slips or 0
        self.guesses = self.guesses or []

    @property
    def differential(self):
        return self.stream is not None and self.stream[0] == "diff"

    def correct(self, y):
        """Symbols turned to the decoder's map"""
        if self.correction is None:
            return y
        th, flip = self.correction
        y = np.conj(y) if flip else y
        return y if th == 0 else (y*np.exp(1j*th)).astype(np.asarray(y).dtype, copy=False)

    def profile(self):
        """The profile that decodes it: the shipped one with the same steps, else one built from `steps`"""
        from ... import profile
        if not self.found:
            raise ValueError(f"no code found: {self}")
        if self.profile_name:
            return profile.load(self.profile_name, **(self.profile_params or {}))
        return profile.Profile("detected", self.definition())

    def rx(self):
        """Receive runner from the timed, carrier-corrected symbols rp.sync.AcquireFEC gives"""
        from ...base.types import Rep
        return self.profile().rx(start=Rep.SYMBOLS, applied=("timing", "carrier"))

    def definition(self):
        return {"base": "profile", "comment": f"detected by rp.analysis.identify.identify: {self}", "steps": self.steps}

    def save(self, name, path=None):
        """Write config/profile/guess/<name>.json (or <path>/guess/<name>.json). Returns the file"""
        import json
        from pathlib import Path
        from ... import _config
        base = Path(path) if path else _config.find_config("profile")
        folder = base / "guess"
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / f"{name.lower()}.json"
        f.write_text(json.dumps(self.definition(), indent=1))
        return f

    def __str__(self):
        if not self.found:
            g = f", blind guesses {self.guesses}" if self.guesses else ""
            return f"no known code ({self.n} symbols{g})"
        inner = "".join(f" + {h.params['interleaver'] + ' + ' if h.params.get('interleaver') else ''}{h.name}" for h in self.inner)
        prof = f" = profile {_named(self.profile_name, self.profile_params)}" if self.profile_name else ""
        mm = f", mismatch {self.mismatch:.4f}" if self.mismatch is not None else ""
        return f"{self.kind} {self.name}{inner} on {self.mod} {self.stream}{prof} (z={self.z:.1f}{mm})"

    __repr__ = __str__


def _named(name, params):
    """A profile and the params of its variant, as `rpps rx -p` takes them: "demo/variants (modulation=16qam)" """
    return name + (f" ({', '.join(f'{k}={v}' for k, v in params.items())})" if params else "")
