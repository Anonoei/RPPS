"""Profiles: one config file per signal, both directions (ARCHITECTURE.md, Profiles)

    {"base": "profile", "comment": "...", "ref": "...",
     "params": {"sps": {"default": 4}},
     "defaults": {"timing": {"method": "gardner"}},       optional: how the receiver estimates what sync steps name
     "steps": [ {"<category>": "<kind>", "id": ..., ...}, ... ]}     in receive order (see rpps.profile.steps)
     "controls": {"<header step id>": {...}}              optional: header-driven control (see rpps.profile.control)
"""
from .. import _config
from ..base.types import Rep
from ..errors import ConfigError
from . import steps as _steps
from . import control as _control
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
        return built

    def rx(self, start=Rep.SAMPLES, applied=()):
        """Receive runner. start: the representation the source gives (a demodulator's timed symbols are SYMBOLS): the steps
        before it are skipped. applied: sync kinds the source has done already (`{"timing"}`): skipped too"""
        slots = []
        for st in self._build():
            if st.takes.value < start.value or (st.category == "sync" and st.kind in applied):
                continue
            slots += [Slot(how, op, st, len(slots) + i) for i, (how, op) in enumerate(st.rx)]
        return Runner(slots, f"{self.name} rx", self.controls, "rx")

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
