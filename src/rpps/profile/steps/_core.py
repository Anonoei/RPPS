"""The registry of step kinds and sync methods, and building one step (rpps.profile.steps)"""
from dataclasses import dataclass, field

from ... import _config
from ...base.types import Rep
from ...errors import ConfigError

STAGE, DECODE, ENCODE = "+", "/", "*" # How an operator is applied: data + op, data / op, data * op

@dataclass
class Step:
    """One built step. rx / tx: [(how, operator)] in the order they run; takes / gives: receive-side representations

    id: what controls call it. controls: {parameter: "live" | "restart"} it can take from a header. rebuild(params): the
    step built again with those parameters (how a restart is done)"""
    category: str
    kind: str
    rx: list
    tx: list
    takes: Rep
    gives: Rep
    spec: dict = field(default_factory=dict)
    obj: object = None # The main operator, for other steps to refer to (the carriers a pilot equalizer follows)
    id: str = ""
    controls: dict = field(default_factory=dict)
    rebuild: object = None

    def __repr__(self):
        return f"<Step {self.id}={self.category}:{self.kind} {self.takes}->{self.gives}>"

def _need(spec, key, where):
    if key not in spec:
        raise ConfigError(f"{where}: needs '{key}'")
    return spec[key]

# ---- the registry: what each category's kinds build, and the methods of the sync kinds that have them

REGISTRY = {c: {} for c in ("signal", "sync", "grid", "wave", "symbol", "code", "link")}
METHODS = {} # sync kind -> {method: builder(m, rep, ctx) -> operator}

def register_step(category, kind):
    """Register a builder for `{"<category>": "<kind>", ...}`: builder(kind, spec, rep, ctx) -> (rx, tx, gives), where rx and
    tx are [(how, operator)] (how: STAGE, DECODE or ENCODE) and gives the representation it gives on receive

        @rp.profile.steps.register_step("signal", "gain")
        def gain(kind, spec, rep, ctx):
            return [(STAGE, MyGain(spec["k"]))], [], rep
    """
    if category not in REGISTRY:
        raise ValueError(f"no step category {category!r} (have {sorted(REGISTRY)})")
    def deco(builder):
        if kind in REGISTRY[category]:
            raise ValueError(f"{category} step {kind!r} is registered already")
        REGISTRY[category][kind] = builder
        return builder
    return deco

def register_sync_method(kind, method):
    """Register a method of a sync kind ("timing", "carrier", "equalize"): builder(m, rep, ctx) -> the stage, where m is the
    step merged with the profile's defaults and the load-time overrides (rpps.sync)"""
    def deco(builder):
        table = METHODS.setdefault(kind, {})
        if method in table:
            raise ValueError(f"{kind} method {method!r} is registered already")
        table[method] = builder
        return builder
    return deco

def kinds():
    """{category: [kinds]} that can be built"""
    return {c: sorted(k) for c, k in REGISTRY.items()}

def methods(kind):
    """The registered methods of a sync kind"""
    return sorted(METHODS.get(kind, {}))

def context(params=None, defaults=None, overrides=None):
    """What building a step reads: the profile's parameters, its sync defaults, the load-time overrides, and the objects
    steps built so far (a pilot equalizer finds its carriers there)"""
    return {"params": params or {}, "defaults": defaults or {}, "overrides": overrides or {}, "objects": {}}

def _by_method(kind):
    """A sync kind built by its method (rx only, one stage, same representation unless the method says)"""
    def builder(_, spec, rep, ctx):
        m = _sync_method(kind, spec, ctx)
        method = m.get("method")
        table = METHODS.get(kind, {})
        if method not in table:
            raise ConfigError(f"{kind}: no method {method!r} (have {', '.join(sorted(table))})")
        st = table[method](m, rep, ctx)
        if st is None: # (the method needs nothing here: "none")
            return [], [], rep
        return [(STAGE, st)], [], getattr(st, "GIVES", None) or (Rep.SYMBOLS if kind == "timing" else rep)
    return builder

def _one(make):
    """A receive-only step of one stage made by make(spec, ctx)"""
    return lambda kind, spec, rep, ctx: ([(STAGE, make(spec, ctx))], [], rep)

def _class(path, base, cat):
    """The class a "class" step names by module path ("rpps.signal.DCBlock", "mypkg.dsp.Gain"): a subclass of `base`
    whose CATEGORY, when it says one, is the step's"""
    import importlib
    from ...base.types import Category
    mod, _, name = str(path).rpartition(".")
    try:
        cls = getattr(importlib.import_module(mod), name) if mod else None
    except (ImportError, AttributeError):
        cls = None
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise ConfigError(f"{cat} class step: {path!r} is not a {base.__name__} class (a module path, e.g. "
                          f"{'rpps.signal.DCBlock' if base.__name__ == 'Stage' else 'mypkg.link.MyFrame'})")
    own = getattr(cls, "CATEGORY", None)
    if own is not None and own != Category[cat.upper()]:
        raise ConfigError(f"{cat} class step: {path} does {own}, not {cat}: name it in a {str(own).lower()} step")
    return cls

_GIVES = {"symbol": Rep.CHANNEL, "code": Rep.DATA, "link": Rep.PACKETS} # What a transform of each category gives on receive

def _class_step(kind, spec, rep, ctx):
    """Your own operator, by its class's module path, in the category it does: {"<category>": "class", "class": "mypkg.X",
    "args": {...}}. signal and sync: a Stage, receive only. wave: a Stage on receive, "tx_class" (and "tx_args") the one
    on transmit. symbol, code, link: a Pipe, decoding on receive and encoding on transmit (one of each, from "args")"""
    from ...base import Stage, Pipe
    cat = next(k for k in ("signal", "sync", "wave", "symbol", "code", "link") if spec.get(k) == "class")
    path, args = _need(spec, "class", f"{cat} class"), spec.get("args", {})
    if cat in ("signal", "sync", "wave"):
        st = _class(path, Stage, cat)(**args)
        tx = []
        if cat == "wave" and spec.get("tx_class"):
            tx = [(STAGE, _class(spec["tx_class"], Stage, cat)(**spec.get("tx_args", args)))]
        return [(STAGE, st)], tx, getattr(st, "GIVES", None) or rep
    cls = _class(path, Pipe, cat)
    if issubclass(cls, Stage): # (a Stage is a Pipe too, but processes rather than encodes and decodes)
        raise ConfigError(f"{cat} class step: {path} is a Stage, not a Pipe that encodes and decodes: name it in a signal, "
                          "sync or wave step")
    rx = cls(**args)
    return [(DECODE, rx)], [(ENCODE, cls(**args))], getattr(rx, "DECODE_GIVES", None) or _GIVES[cat]

for _cat in ("signal", "sync", "wave", "symbol", "code", "link"):
    register_step(_cat, "class")(_class_step)

def _sync_method(kind, spec, ctx):
    """The method for a sync step: the step's own, else the load-time override, else the profile's defaults, else ours.
    What its builder reads is recorded on the step's own read set and under ctx["method_reads"][kind], so settings
    nothing took (a misspelling, a key another method takes) are reported (rpps._config.Keys)"""
    read = spec.read if isinstance(spec, _config.Keys) else None
    chosen = _config.Keys(read=read)
    for src in (_SYNC_DEFAULTS.get(kind, {}), ctx["defaults"].get(kind, {}), ctx["overrides"].get(kind, {})):
        dict.update(chosen, dict(dict.items(src)))
    dict.update(chosen, {k: v for k, v in dict.items(spec) if k != "sync"})
    ctx.setdefault("method_reads", {}).setdefault(kind, []).append(chosen.read)
    return chosen

_SYNC_DEFAULTS = {
    "timing": {"method": "oerder_meyr"}, # first class: block estimation tracked across windows (rpps.sync)
    "carrier": {"method": "mpower"},     # first class likewise
    "align": {"method": "align"},
    "puncture": {"method": "reencode"},
    "pilots": {"method": "ls"},
    "agc": {"method": "agc"},
    "frame": {"method": "preamble"},
    "equalize": {"method": "ls"},        # first class: block least squares per window
    "delimit": {"method": "word"},
}

def _word(m, where):
    """(word, bits, order) a step gives: a `sequence` of type word, or `word` + `bits` (+ `order`) inline"""
    if "sequence" in m:
        from ... import sequence
        w = sequence.load(m["sequence"])
        if not hasattr(w, "bits") or not hasattr(w, "value"):
            raise ConfigError(f"{where}: sequence {m['sequence']!r} is not a word")
        return w.bits, None, "msb"
    return _need(m, "word", where), _need(m, "bits", where), m.get("order", "msb")

def _given(spec, keys, **convert):
    """The operator's arguments the step gives (the operator's defaults stand for the rest); convert: {key: function}"""
    out = {k: spec[k] for k in keys if k in spec}
    out.update({k: f(spec[k]) for k, f in convert.items() if k in spec})
    return out

def _complex(v):
    return complex(v[0], v[1]) if isinstance(v, (list, tuple)) else complex(v)

def build(spec, rep, ctx):
    """Build one step that receives `rep`. ctx: params, defaults, overrides and objects built so far (see `context`)"""
    cats = [k for k in dict.keys(spec) if k in REGISTRY] # (the keys as written: not every key read, see rpps._config.Keys)
    if len(cats) != 1:
        raise ConfigError(f"a profile step names one category of {sorted(REGISTRY)}, got {spec}")
    cat = cats[0]
    kind = spec[cat]
    if kind not in REGISTRY[cat]:
        raise ConfigError(f"{cat} step: no kind {kind!r} (have {', '.join(sorted(REGISTRY[cat]))})")
    rx, tx, gives = REGISTRY[cat][kind](kind, spec, rep, ctx)
    obj = next((op for _, op in rx), None)
    controls = {}
    for _, op in rx + tx:
        controls.update(getattr(op, "CONTROLS", None) or {})
    st = Step(cat, kind, rx, tx, rep, gives, spec, obj, str(spec.get("id", kind)), controls)
    st.rebuild = lambda params: build({**spec, **params}, rep, ctx)
    return st
