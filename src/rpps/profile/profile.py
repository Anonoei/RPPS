"""Profiles: one config file per signal, both directions (ARCHITECTURE.md, Profiles)

    {"base": "profile", "comment": "...", "ref": "...",
     "params": {"sps": {"default": 4}},
     "defaults": {"timing": {"method": "gardner"}},       optional: how the receiver estimates what sync steps name
     "steps": [ {"<category>": "<kind>", "id": ..., ...}, ... ]}     in receive order (see rpps.profile.steps)
     "controls": {"<header step id>": {...}}              optional: header-driven control (see rpps.profile.control)
     "parallel": "<step id>"                              optional: the first step that runs frame by frame (see rpps.profile.parallel)
"""
from .. import _config
from ..base.types import Rep
from ..errors import ConfigError
from . import steps as _steps
from . import control as _control
from . import parallel as _parallel
from .runner import Runner, Slot

class Profile:
    """A signal: its steps in receive order, built from a definition. `rx()` and `tx()` give runners"""
    def __init__(self, name, obj, sync=None, params=None):
        self.name, self.obj = name, obj
        self.comment, self.ref = obj.get("comment", ""), obj.get("ref", "")
        self.params = dict(params or {})
        self._sync = {k: dict(v) for k, v in (sync or {}).items()}
        if "steps" not in obj or not obj["steps"]:
            raise ConfigError(f"profile {name}: no steps")
        self.steps = self._build()
        self.controls = _control.parse(obj.get("controls"), self.steps, f"profile {name}")
        self.region = _parallel.plan(obj.get("parallel"), self.steps, self.controls, f"profile {name}")

    def _build(self):
        """The steps, built (each runner builds its own: header-driven control changes their operators)"""
        ctx = {"params": self.params, "defaults": self.obj.get("defaults", {}), "overrides": self._sync, "objects": {}}
        specs = list(self.obj["steps"])
        # A step can refer to one after it (a pilot equalizer to its carriers): build those first
        early = {i: _steps.build(s, None, ctx) for i, s in enumerate(specs) if s.get("symbol") == "carriers"}
        built, rep = [], Rep.SAMPLES
        for i, spec in enumerate(specs):
            st = early.get(i) or _steps.build(spec, rep, ctx)
            st.takes = rep
            rep = st.gives
            built.append(st)
        if self._sync.get("timing", {}).get("fuse", True):
            self._fuse(built)
        return built

    @staticmethod
    def _fuse(steps):
        """A matched filter right before a timing that is known (not estimated) is evaluated by it, at the instants it samples:
        the step of the filter keeps no operator (Oerder & Meyr needs the filter's output at every sample, so nothing fuses)"""
        from .steps import _KnownTiming
        for a, b in zip(steps, steps[1:]):
            if (a.category == "wave" and a.kind == "shape" and len(a.rx) == 1 and getattr(a.rx[0][1], "matched", False)
                    and b.category == "sync" and b.kind == "timing" and len(b.rx) == 1 and isinstance(b.rx[0][1], _KnownTiming)):
                b.rx[0][1].fuse(a.rx[0][1])
                a.rx = []

    def rx(self, start=Rep.SAMPLES, applied=(), parallel=None):
        """Receive runner. start: the representation the source gives (a demodulator's timed symbols are SYMBOLS): the steps
        before it are skipped. applied: sync kinds the source has done already (`{"timing"}`): skipped too.
        parallel: an `rp.parallel.Parallel` (where and how the profile's parallel region runs, frame by frame); the profile
        must have one (`"parallel"` in its definition)"""
        slots = []
        for st in self._build():
            if st.takes.value < start.value or (st.category == "sync" and st.kind in applied):
                continue
            slots += [Slot(how, op, st, len(slots) + i) for i, (how, op) in enumerate(st.rx)]
        if parallel is None:
            return Runner(slots, f"{self.name} rx", self.controls, "rx")
        if self.region is None:
            raise ConfigError(f"profile {self.name}: has no parallel region (no \"parallel\" in its definition)")
        from ..parallel.runner import ParallelRunner
        first = self.region.steps[0].id
        from ..parallel.worker import RegionSteps
        make = RegionSteps(self.name, self.obj, self._sync, self.params, self.region.first)
        return ParallelRunner(slots, f"{self.name} rx (parallel)", self.controls, "rx", make, first, parallel)

    def tx(self):
        """Transmit runner: the transform steps inverted, in reverse order (sync and signal steps have no transmit side).
        Call it as `runner(block, fields=...)` to set the fields of the frames that start in the block (see Runner)"""
        slots = []
        for st in reversed(self._build()):
            slots += [Slot(how, op, st, len(slots) + i) for i, (how, op) in enumerate(st.tx)]
        return Runner(slots, f"{self.name} tx", self.controls, "tx")

    def __repr__(self):
        return f"<Profile {self.name}: " + ", ".join(f"{s.category}:{s.kind}" for s in self.steps) + ">"

    @staticmethod
    def load(name, obj):
        """Loader of the profile domain (`_config.load_obj`): the parameters are already resolved into the steps"""
        return Profile(name, obj, params=_config.declared(obj))

def load(name, sync=None, **params):
    """Load a profile: `rp.profile.load("ieee802154/2450", sps=8)`. params: values for its "params".
    sync: per sync kind, the method and its settings, over the profile's defaults: `sync={"timing": {"method": "known"}}`"""
    file = _config.find("profile", name)
    if file is None:
        raise FileNotFoundError(f"profile/{name}: not found, have {_config.ls('profile')}")
    obj = _config.read(file)
    values = {**_config.declared(obj), **params}
    resolved = _config.resolve(obj, params, f"profile/{name}")
    return Profile(name, resolved, sync=sync, params=values)

def ls():
    """The profiles there are: {family: [modes]}"""
    return _config.ls("profile")

DOMAIN = _config.register_domain("profile", {"profile": Profile.load})
