"""What the blind code searches return: a Hypothesis (a structure and how far its test sits from chance), a Report
of all of them, a Guess (a definition to save), and `pick`, the one that stands out at one alignment only"""
import numpy as np

from ..math.prob import z_critical

class Hypothesis:
    __slots__ = ("kind", "name", "z", "params", "coding", "guess")
    def __init__(self, kind, name, z, params=None, coding=None, guess=None):
        self.kind = kind
        self.name = name
        self.z = z
        self.params = params or {}
        self.coding = coding # (folder, config name) when it is in config/fec
        self.guess = guess # The Guess (definition to save) a blind find comes with

    def __repr__(self):
        c = f" [{self.coding[0]}/{self.coding[1]}]" if self.coding else ""
        return f"<{self.kind} {self.name} z={self.z:.1f}{c} {self.params}>"

class Report:
    """Ranked hypotheses and a verdict"""
    __slots__ = ("hyps", "trials", "z_crit", "facts")
    def __init__(self, hyps, trials, facts):
        self.hyps = sorted(hyps, key=lambda h: -h.z)
        self.trials = trials
        self.z_crit = z_critical(trials)
        self.facts = facts

    @property
    def best(self):
        return self.hyps[0] if self.hyps else None

    @property
    def verdict(self):
        """The best hypothesis if it stands out from chance, else None"""
        return self.best if self.best and self.best.z > self.z_crit else None

    def __str__(self):
        out = [f"{self.trials} hypotheses tried, significance threshold z>{self.z_crit:.1f}"]
        out += [f"  {h}" for h in self.hyps[:5]]
        out.append(f"verdict: {self.verdict if self.verdict else 'no FEC structure found'}")
        return "\n".join(out)

class Guess:
    """A guessed definition (a code, a scrambler, a CRC). Decoding only happens from a saved config: `save` it, then load
    it (rp.code.load("block/guess/<name>"), rp.code.scram.load("feedthrough/guess/<name>"), rp.link.load("crc/guess/<name>"))"""
    __slots__ = ("kind", "definition", "evidence", "plan", "scores")
    FOLDER = {"block": ("fec", "block"), "rs": ("fec", "block"), "conv": ("fec", "conv"),
              "scramble": ("scramble", "feedthrough"), "crc": ("crc", ""),
              "header": ("header", "link"), "plan": ("plan", "")} # kind: (config domain, folder in it)

    def __init__(self, kind, definition, evidence=None):
        self.kind = kind
        self.definition = definition
        self.evidence = evidence or {}
        self.plan = None # (a plan's guess: the Plan itself)
        self.scores = None # (a plan's guess: the curves it was chosen on, rp.viz.plan_evidence)

    def __repr__(self):
        d = self.definition
        if self.kind == "block":
            if d.get("type") == "product":
                return f"<Guess product ({d['rows']['n']},{d['rows']['k']}) x ({d['cols']['n']},{d['cols']['k']}) {self.evidence}>"
            if "checks" in d:
                return f"<Guess {d.get('type', 'block')} n={d['n']} k={d['k']} {len(d['checks'])} checks {self.evidence}>"
            extra = {k: v for k, v in d.items() if k not in ("base", "type", "n", "k", "comment")}
            return f"<Guess {d['type']} ({d['n']},{d['k']}) {extra} {self.evidence}>"
        if self.kind == "rs":
            return f"<Guess RS({d['n']},{d['k']}) poly {d['poly']:#x} fcr {d['fcr']} prim {d['prim']} {self.evidence}>"
        if self.kind == "header":
            return f"<Guess header {[(f['name'], f['bits']) for f in d['fields']]} {self.evidence}>"
        if self.kind == "crc":
            return f"<Guess CRC-{d['width']} poly {d['poly']} init {d['init']} xorout {d['xorout']} {self.evidence}>"
        if self.kind == "plan":
            return f"<Guess burst time plan {self.evidence}>"
        if self.kind == "scramble":
            return f"<Guess feedthrough scrambler taps {d['poly']} {self.evidence}>"
        # A convolutional guess: a mother code it extends (its puncture its own), or the generators themselves
        what = f"extends {d['extends']}" if d.get("extends") else \
            f"K={d.get('constraint')} generators {[''.join(map(str, g)) for g in d.get('generator', [])]}"
        if d.get("puncture") is not None:
            pat = np.asarray(d["puncture"])
            what += f" punctured {pat.tolist()} (rate {pat.shape[1]}/{int(pat.sum())})"
        return f"<Guess conv {what} {self.evidence}>"

    def save(self, name, path=None):
        """Write config/<domain>/<folder>/guess/<name>.json (or <path>/<folder>/guess/<name>.json). Returns the file"""
        import json
        from pathlib import Path
        from .. import _config
        domain, sub = self.FOLDER[self.kind]
        base = Path(path) if path else _config.find_config(domain)
        folder = base / sub / "guess"
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / f"{name.lower()}.json"
        f.write_text(json.dumps({**self.definition, "guess": self.evidence}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else list(o)))
        return f

def pick(hyps, z_crit, key, fits):
    """The best hypothesis that stands out, if the code fits these bits at its alignment only

    A code fits its own bits at one alignment (puncture phase, codeword start). Bits that fit at several did not come
    from it: constant bits (an unmodulated carrier) are a codeword of every linear code at every alignment, and bits
    repeated by a symbol rate locked too high fit a repetition code at all but a few. fits(h, other): other is as good"""
    good = [h for h in hyps if h.z > z_crit]
    if not good:
        return None
    best = min(good, key=lambda h: (key(h), -h.z))
    same = [h for h in good if h.coding == best.coding and h.params["stream"] == best.params["stream"]
            and h.params.get("segment") == best.params.get("segment") # (a slip between segments moves the offset)
            and h.params["offset"] != best.params["offset"] and fits(best, h)]
    return None if same else best
