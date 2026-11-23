"""What a profile needs and gives, before it runs: for a program building a menu (rpps.profile.Decoder uses it too)

Each profile's "requires" (config/profile, rpps.profile) says, as expressions over its params and `baud`:

    baud          its symbol rate (chips, subcarrier spacing), or null when the standard leaves it open (DVB-S, CCSDS):
                  given to the Decoder, or measured
    Fs            the sample rate it receives at
    bw            the bandwidth of the channel it takes
    min_samples   enough samples for one message
    input         "baseband" (complex samples of the channel, default) or "audio" (an FM channel's audio: AX.25)
    from_cf       params that follow from the signal's centre frequency `cf` (BLE's advertising channel, NR's fc)

    rp.profile.info("dvbs2/normal", baud=27.5e6).requires["Fs"]      # 110e6 at 4 samples a symbol
    [i.name for i in rp.profile.catalogue() if i.requires["Fs"] and i.requires["Fs"] <= 20e6]
"""
import math

from .. import _config
from ..errors import ConfigError
from .profile import ls

KEYS = ("baud", "Fs", "bw", "min_samples") # Every profile's requires has these, in this order (each may use those before)

class ProfileInfo:
    """A profile, described: name, comment, ref, tags, params ({name: {"default", "comment", and "choices" when it has them}}), requires (evaluated:
    baud, Fs, bw, min_samples, None where they wait for a baud not given; input; from_cf as written; comment), kinds (the
    message kinds it gives, sorted: what `Profile.rx(kinds=...)` and the Decoder take; the profile is built, with the params
    given, when first asked)"""
    __slots__ = ("name", "comment", "ref", "tags", "params", "requires", "_given", "_kinds")
    FIELDS = ("name", "comment", "ref", "tags", "params", "requires", "kinds")

    def __init__(self, name, comment, ref, tags, params, requires, given=None):
        self.name, self.comment, self.ref, self.tags, self.params, self.requires = name, comment, ref, tags, params, requires
        self._given, self._kinds = dict(given or {}), None

    @property
    def kinds(self):
        if self._kinds is None:
            from .profile import load
            self._kinds = sorted(load(self.name, **self._given).kinds())
        return self._kinds

    def to_dict(self):
        return {k: getattr(self, k) for k in self.FIELDS}

    def __repr__(self):
        r = self.requires
        rate = "any rate" if r["baud"] is None else f"{r['Fs']/1e6:g} MS/s"
        return f"<ProfileInfo {self.name}: {rate}, {r['input']}>"

def requirements(req, values, baud=None, name="profile"):
    """A profile's "requires", evaluated over its param values (and `baud`, for a profile whose rate is open). A profile of
    your own without one says nothing: every value None"""
    out = {"input": req.get("input", "baseband"), "from_cf": dict(req.get("from_cf") or {}), "comment": req.get("comment")}
    v = dict(values)
    for k in KEYS:
        e = req.get(k)
        if k == "baud" and baud is not None:
            if e is not None and not math.isclose(float(_config.evaluate(e, v, f"{name} requires")), float(baud), rel_tol=1e-3):
                raise ValueError(f"{name}: its rate is set ({e}), not {baud:g}")
            val = float(baud)
        elif e is None:
            val = None
        else:
            try:
                val = _config.evaluate(e, v, f"{name} requires") if isinstance(e, str) else e
            except TypeError: # Waits for a baud not given
                val = None
        if k == "min_samples" and val is not None:
            val = int(math.ceil(val))
        elif val is not None:
            val = float(val)
        out[k] = val
        v[k] = val
    return out

def info(name, baud=None, **params):
    """A profile described (ProfileInfo), its requires evaluated with these params (the defaults otherwise) and `baud`"""
    file = _config.find("profile", name)
    if file is None:
        raise FileNotFoundError(f"profile/{name}: not found, have {_config.ls('profile')}")
    obj = _config.read(file)
    decl = obj.get("params") or {}
    unknown = set(params) - set(decl)
    if unknown:
        raise ConfigError(f"profile/{name}: no params {sorted(unknown)} (has {sorted(decl)})")
    values = _config.chosen(obj, {**_config.declared(obj), **params}, f"profile/{name}")
    ps = {k: {"default": v.get("default") if isinstance(v, dict) else v,
              "comment": v.get("comment") if isinstance(v, dict) else None,
              **({"choices": list(v["choices"])} if isinstance(v, dict) and "choices" in v else {})} for k, v in decl.items()}
    return ProfileInfo(name, obj.get("comment", ""), obj.get("ref"), list(obj.get("tags") or []), ps,
                       requirements(obj.get("requires") or {}, values, baud, f"profile/{name}"), params)

def variants(name, limit=64):
    """The param values a blind search tries for a profile: every combination of the choices of its params (but those
    marked "blind": false), the defaults first, at most `limit` of them; [{}] for a profile without choices. A variant
    is `load(name, **v)`"""
    import itertools
    file = _config.find("profile", name)
    if file is None:
        raise FileNotFoundError(f"profile/{name}: not found, have {_config.ls('profile')}")
    decl = _config.read(file).get("params") or {}
    axes = []
    for k, spec in decl.items():
        if isinstance(spec, dict) and spec.get("choices") and spec.get("blind", True):
            keys = list(spec["choices"])
            default = spec.get("default")
            first = next((c for c in keys if c == default or (isinstance(c, str) and c == str(default))), None)
            vals = [default] if first is not None else []
            vals += [c for c in keys if c is not first]
            axes.append((k, vals))
    if not axes:
        return [{}]
    return [dict(zip([k for k, _ in axes], combo)) for combo in itertools.islice(itertools.product(*[v for _, v in axes]), limit)]

def catalogue():
    """Every profile there is, described (with its defaults), by name"""
    return [info(f"{family}/{mode}") for family, modes in sorted(ls().items()) for mode in modes]
