"""What each step of a profile builds (ARCHITECTURE.md, Profiles)

A step is one dict with one category key, `{"<category>": "<kind>", ...its parameters}`, in receive order. Building it
gives a Step: the operators it runs on receive and on transmit, and the representations it takes and gives on receive.

    transform steps (wave, symbol, code, link) build a transmit/receive pair: the receive side as written, the transmit
        side inverted (the profile runs transmit steps in reverse order)
    sync and signal steps run on receive only

Sync steps name what is estimated; how is a method, chosen by `defaults` in the profile or the `sync` argument of
`rp.profile.load`, else the default here.

Every step has an id ("id", else its kind), by which a profile's `controls` name it, and says which of its parameters a
header may change and how (`controls`: {parameter: "live" | "restart"}, from its operators' CONTROLS). A parameter it
does not list cannot be controlled (see rpps.profile.control).
"""
from dataclasses import dataclass, field

from .. import _config
from ..base.types import Rep
from ..errors import ConfigError

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

# ---- signal (receive only)

def _signal(kind, spec, rep, ctx):
    from .. import signal
    if kind == "filter":
        st = signal.filters.load(_need(spec, "type", "signal filter"), _need(spec, "name", "signal filter"))
        return [(STAGE, st)], [], rep
    if kind == "decimate":
        return [(STAGE, signal.sample.down.Decimate(spec["ratio"], spec.get("offset", 0)))], [], rep
    raise ConfigError(f"signal step: no kind {kind!r} (have filter, decimate)")

# ---- sync (receive only)

def _sync_method(kind, spec, ctx):
    """The method for a sync step: the step's own, else the load-time override, else the profile's defaults, else ours"""
    chosen = {**_SYNC_DEFAULTS.get(kind, {}), **ctx["defaults"].get(kind, {}), **ctx["overrides"].get(kind, {})}
    chosen.update({k: v for k, v in spec.items() if k not in ("sync",)})
    return chosen

_SYNC_DEFAULTS = {
    "timing": {"method": "oerder_meyr"}, # first class: block estimation tracked across windows (ARCHITECTURE.md, Sync methods)
    "carrier": {"method": "mpower"},     # first class likewise; the loops (mm, gardner, costas) are there by name
    "align": {"method": "align"},
    "puncture": {"method": "reencode"},
    "pilots": {"method": "ls"},
    "agc": {"method": "agc"},
    "frame": {"method": "preamble"},
    "equalize": {"method": "ls"},        # first class: block least squares per window
    "delimit": {"method": "word"},
}

def _sync(kind, spec, rep, ctx):
    from .. import sync, signal
    m = _sync_method(kind, spec, ctx)
    method = m.get("method")
    sps = ctx["params"].get("sps")
    if kind == "timing":
        if method == "oerder_meyr":
            st = sync.time.OerderMeyr(m.get("sps"), m.get("window", 1e-3), m.get("min_symbols", 2048), m.get("bw", 0.25),
                                      m.get("gate", 2.0), m.get("name", "timing"))
        elif method in ("mm", "gardner"):
            st = sync.time.MM(m.get("alpha", 0.3), ted="mm" if method == "mm" else "gardner")
        elif method == "known":
            st = _KnownTiming(m.get("sps", sps))
        elif method == "fixed":
            st = signal.sample.down.Decimate(_need(m, "sps", "fixed timing") if sps is None else m.get("sps", sps), m["offset"])
        else:
            raise ConfigError(f"timing: no method {method!r} (have oerder_meyr, mm, gardner, known, fixed)")
        return [(STAGE, st)], [], Rep.SYMBOLS
    if kind == "carrier":
        if method == "pilots": # Data-aided, per frame, from pilot blocks (DVB-S2): their layout comes from a header
            st = sync.PilotPhase(m.get("slot", 90), m.get("every", 16), m.get("length", 36), _complex(m.get("value", [0.7071067811865476, 0.7071067811865476])),
                                 m.get("slots"), m.get("pilots", True), m.get("lead"), m.get("name", "pilots"))
            return [(STAGE, st)], [], rep
        if method == "mpower":
            order = int(m.get("order", 4))
            offset = m.get("offset")
            if offset is None:
                from .. import symbol
                offset = sync.phase.offset_of(symbol.load(m["constellation"]), order) if "constellation" in m else 0.0
            st = sync.phase.MPower(order, offset, m.get("window", 1e-3), m.get("min_symbols", 1024), m.get("bw", 0.25),
                                   m.get("gate", 3.0), m.get("relock", 4), m.get("name", "carrier"))
        elif method == "costas":
            st = sync.Costas.from_bandwidth(m.get("bw", 0.01), order=m.get("order", 4))
        elif method == "ff":
            st = sync.phase.FFCarrier(m.get("window", 64))
        elif method == "none":
            return [], [], rep # The carrier is already off (an upstream source), or there is no estimator for this signal yet
        else:
            raise ConfigError(f"carrier: no method {method!r} (have mpower, costas, ff, pilots, none)")
        return [(STAGE, st)], [], rep
    if kind == "align":
        st = sync.Align(_need(m, "period", "align"), markers=m.get("markers"), pattern=m.get("pattern"),
                        unit=m.get("unit", 8), polarity=m.get("polarity", True), name=m.get("name", "align"),
                        start=m.get("start", "any"))
        return [(STAGE, st)], [], rep
    if kind == "puncture":
        from .. import code
        c = code.load(m.get("type", "conv"), _need(m, "fec", "puncture"))
        return [(STAGE, sync.Puncture(c, m.get("window", 4000)))], [], rep
    if kind == "pilots":
        carriers = ctx["objects"].get("carriers")
        if carriers is None:
            raise ConfigError("pilots: needs a symbol carriers step after it")
        return [(STAGE, sync.PilotEqualizer(carriers, m.get("taps")))], [], rep
    if kind == "agc":
        return [(STAGE, sync.AGC(m.get("target", 1.0), m.get("mu", 0.01)))], [], rep
    if kind == "equalize":
        if method == "ls":
            from .. import symbol
            st = sync.equalize.BlockLS(symbol.load(_need(m, "constellation", "equalize ls")), m.get("taps", 7), m.get("window", 1e-3),
                                       m.get("min_symbols", 1024), m.get("iters", 3), m.get("blend", 0.5), m.get("gate", 0.5),
                                       m.get("name", "equalizer"))
        elif method == "lms":
            from .. import symbol
            pts = symbol.load(m["constellation"]).points if "constellation" in m else None
            st = sync.equalize.LMS(m.get("taps", 11), m.get("mu", 0.01), points=pts)
        elif method == "cma":
            st = sync.equalize.CMA(m.get("taps", 11), m.get("mu", 0.01))
        else:
            raise ConfigError(f"equalize: no method {method!r} (have ls, lms, cma)")
        return [(STAGE, st)], [], rep
    if kind == "frame": # Frames by a known preamble among symbols: the one a physical header definition starts with
        from .. import symbol
        h = symbol.load(_need(m, "header", "sync frame"))
        st = sync.FrameSync(h.preamble, h.words() if m.get("words", True) else None, m.get("threshold"),
                            m.get("max_length"), m.get("name", "frame"))
        return [(STAGE, st)], [], rep
    if kind == "delimit": # Frames by the word before them among bits: a link header's preamble word, or one given
        if "header" in m:
            from .. import link
            pre = _config.read(_config.find("encap", m["header"], "header")).get("preamble") or {}
            h = link.load("header", m["header"])
            word, bits, order = pre.get("word"), pre.get("bits"), h.order
        else:
            word, bits, order = _need(m, "word", "sync delimit"), _need(m, "bits", "sync delimit"), m.get("order", "msb")
        if word is None:
            raise ConfigError(f"sync delimit: {m.get('header')} has no preamble word")
        st = sync.Delimiter(word, bits, order, m.get("errors", 0), m.get("name", "delimiter"))
        return [(STAGE, st)], [], rep
    raise ConfigError(f"sync step: no kind {kind!r} (have timing, carrier, equalize, align, puncture, pilots, agc, frame, delimit)")

def _complex(v):
    return complex(v[0], v[1]) if isinstance(v, (list, tuple)) else complex(v)

from ..base import Stage as _Stage

class _KnownTiming(_Stage):
    """Timing that is known, not estimated (a loopback, a source whose timing is given): one sample per symbol at the pulse
    peaks, which the group delay so far (meta.delay) places. Tracks the absolute input index of the next peak, so any block
    cut gives the same output"""
    def __init__(self, sps):
        self.sps = sps

    def __str__(self):
        return "KnownTiming"

    def new_state(self, meta):
        from types import SimpleNamespace
        return SimpleNamespace(next=None)

    def step(self, samples, st, meta):
        import numpy as np
        sps = int(self.sps or meta.isps)
        if st.next is None:
            first = int(round(meta.delay*meta.Fs)) # absolute index of the first symbol's peak
            st.next = first if first >= meta.pos else first + -(-(meta.pos - first)//sps)*sps
        idx = np.arange(st.next - meta.pos, len(samples), sps)
        st.next += len(idx)*sps
        meta.op.Fs = meta.Fs/sps
        meta.op.sps = 1
        return samples[idx]

# ---- wave (pairs)

def _pulse(name):
    """The pulse resource and whether it is a symbol pulse or a frequency pulse"""
    from .. import signal
    f = _config.find("filters", name, "shaping")
    if f is None:
        raise ConfigError(f"wave shape: no pulse {name!r} in config/filters/shaping")
    kind = _config.read(f).get("kind", "symbol")
    return (lambda: signal.filters.load("shaping", name)), kind

def _wave(kind, spec, rep, ctx):
    from .. import wave, signal
    sps = spec.get("sps", ctx["params"].get("sps"))
    if kind == "shape":
        make, pulse_kind = _pulse(_need(spec, "pulse", "wave shape"))
        tx = [(STAGE, signal.sample.up.Zeros(int(_need({"sps": sps}, "sps", "wave shape")))), (STAGE, make())]
        rx = [(STAGE, make())] if pulse_kind == "symbol" else [] # a frequency pulse has no matched filter
        return rx, tx, Rep.SAMPLES
    if kind == "stagger":
        d = spec.get("delay", 0.5)
        return [(STAGE, wave.Stagger("i", d))], [(STAGE, wave.Stagger("q", d))], Rep.SAMPLES
    if kind == "fm":
        h, step = _need(spec, "h", "wave fm"), spec.get("step", 2.0)
        rx = [(STAGE, wave.FMDemod(wave.fm_gain(h, sps, step))),
              (STAGE, signal.filters.ShapingRect(int(sps), int(sps), 1/int(sps)))]
        return rx, [(STAGE, wave.FM(h, step))], Rep.SAMPLES
    if kind == "ofdm":
        n, cp = spec.get("n_fft", 64), spec.get("n_cp", 16)
        return [(STAGE, wave.OFDMRx(n, cp))], [(STAGE, wave.OFDMTx(n, cp))], Rep.SYMBOLS
    raise ConfigError(f"wave step: no kind {kind!r} (have shape, stagger, fm, ofdm)")

# ---- symbol, code, link (operators that encode and decode)

def _pair(op_rx, op_tx, gives):
    return [(DECODE, op_rx)], [(ENCODE, op_tx)], gives

def _symbol(kind, spec, rep, ctx):
    from .. import symbol, spread
    if kind == "constellation":
        name = _need(spec, "name", "symbol constellation")
        make = lambda: symbol.load(name, spec.get("mapping"))
        return _pair(make(), make(), Rep.CHANNEL)
    if kind == "differential":
        d = symbol.Differential()
        return [(DECODE, d)], [(ENCODE, symbol.Differential())], Rep.SYMBOLS
    if kind == "spread":
        code = spread.load(_need(spec, "sequence", "symbol spread"), **spec.get("args", {}))
        make = lambda: symbol.Spread(code, spec.get("pair", False))
        return _pair(make(), make(), Rep.CHANNEL if code.count > 1 else Rep.SYMBOLS)
    if kind == "carriers":
        c = symbol.Carriers(spec.get("n_fft", 64), spec.get("used"), tuple(spec.get("pilots", (-21, -7, 7, 21))))
        ctx["objects"]["carriers"] = c
        return [(DECODE, c)], [(ENCODE, c)], Rep.SYMBOLS
    if kind == "header":
        name = _need(spec, "name", "symbol header")
        return _pair(symbol.load(name), symbol.load(name), Rep.SYMBOLS)
    if kind == "pilots":
        make = lambda: symbol.Pilots(spec.get("slot", 90), spec.get("every", 16), spec.get("length", 36),
                                     _complex(spec.get("value", [0.7071067811865476, 0.7071067811865476])), spec.get("slots"),
                                     spec.get("pilots", True), spec.get("id", "pilots"))
        return _pair(make(), make(), Rep.SYMBOLS)
    if kind == "scramble":
        from ..code import scram
        ref = _need(spec, "sequence", "symbol scramble").split("/")
        make = lambda: symbol.PhaseScramble(scram.load(*ref), spec.get("offsets", [0]), spec.get("step", 90),
                                            spec.get("length"), spec.get("id", "scramble"))
        return _pair(make(), make(), Rep.SYMBOLS)
    raise ConfigError(f"symbol step: no kind {kind!r} (have constellation, differential, spread, carriers, header, pilots, scramble)")

def _code(kind, spec, rep, ctx):
    from .. import code
    if kind == "fec":
        folder, name = _need(spec, "type", "code fec"), _need(spec, "name", "code fec")
        def make():
            if name is None: # Nothing (yet): a header names the code
                op = {"block": code.Block, "interleave": code.Interleaver, "conv": code.Convolutional}[folder](None, None)
                op.folder, op.name = folder, "none"
                return op
            return code.load(folder, name, **spec.get("args", {}))
        return _pair(make(), make(), Rep.DATA)
    if kind == "scramble":
        ref = _need(spec, "name", "code scramble").split("/")
        make = lambda: code.scram.load(*ref, **spec.get("args", {}))
        return _pair(make(), make(), Rep.DATA)
    if kind == "uncoded":
        return _pair(code.Uncoded(), code.Uncoded(), Rep.DATA)
    raise ConfigError(f"code step: no kind {kind!r} (have fec, scramble, uncoded)")

def _link(kind, spec, rep, ctx):
    from .. import link
    if kind in ("crc", "frame", "header"):
        name = _need(spec, "name", f"link {kind}")
        make = lambda: link.load(kind, name)
        return _pair(make(), make(), Rep.PACKETS)
    raise ConfigError(f"link step: no kind {kind!r} (have crc, frame, header)")

BUILDERS = {"signal": _signal, "sync": _sync, "wave": _wave, "symbol": _symbol, "code": _code, "link": _link}

def build(spec, rep, ctx):
    """Build one step that receives `rep`. ctx: params, defaults, overrides and objects built so far"""
    cats = [k for k in spec if k in BUILDERS]
    if len(cats) != 1:
        raise ConfigError(f"a profile step names one category of {sorted(BUILDERS)}, got {spec}")
    cat = cats[0]
    kind = spec[cat]
    rx, tx, gives = BUILDERS[cat](kind, spec, rep, ctx)
    obj = next((op for _, op in rx), None)
    controls = {}
    for _, op in rx + tx:
        controls.update(getattr(op, "CONTROLS", None) or {})
    st = Step(cat, kind, rx, tx, rep, gives, spec, obj, str(spec.get("id", kind)), controls)
    st.rebuild = lambda params: build({**spec, **params}, rep, ctx)
    return st
