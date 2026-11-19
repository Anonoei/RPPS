"""What each step of a profile builds (README.dev.md, Profiles)

A step is one dict with one category key, `{"<category>": "<kind>", ...its parameters}`, in receive order. Building it
gives a Step: the operators it runs on receive and on transmit, and the representations it takes and gives on receive.

    transform steps (wave, symbol, code, link) build a transmit/receive pair: the receive side as written, the transmit
        side inverted (the profile runs transmit steps in reverse order)
    sync and signal steps run on receive only

A step names the resources it uses by their path below the resource ("additive/dvb_energy", "rrc/0p35"), so the profile
says what each is.

Sync steps name what is estimated; how is a method, chosen by `defaults` in the profile or the `sync` argument of
`rp.profile.load`, else the default here.

Every step has an id ("id", else its kind), by which a profile's `controls` name it, and says which of its parameters a
header may change and how (`controls`: {parameter: "live" | "restart"}, from its operators' CONTROLS). A parameter it
does not list cannot be controlled (see rpps.profile.control).

The kinds (`python -m rpps steps` lists them with the sync methods):

    signal   filter (a definition), lowpass / highpass / bandpass / bandstop (designed in Hz: cutoff + taps, or transition +
             atten_db), decimate, resample (up/down), channelize (freq, width), shift (freq), conjugate (spectral inversion),
             dc_block, deemphasis, stage (any Stage class by module path)
    sync     timing, carrier, equalize, agc (methods), frame (a physical header's preamble), delimit (a word among bits:
             word + bits, a word sequence, or a link header's preamble; length, keep, polarity, errors), align (a repeating
             marker), puncture, pilots, squelch, ofdm
    wave     shape (a pulse definition; symbol pulses get a matched filter), stagger, fm (h + sps: FSK; deviation: analog),
             am (the envelope; alpha null keeps its mean), ssb (bw, sideband), ofdm (n_fft, n_cp)
    symbol   constellation (name, mapping), differential, spread (a chip sequence), carriers, header, pilots, scramble
    code     fec (type block / conv / interleave, name, args), scramble (name, args; length: frames of that many bits from
             the first), line (invert, nrzi, nrzm, manchester, thomas), marker (a word before each frame: length, or frames by
             control), uncoded
    link     crc, frame (HDLC), header (fields, crc, fcs, payload)
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

# ---- the registry: what each category's kinds build, and the methods of the sync kinds that have them

REGISTRY = {c: {} for c in ("signal", "sync", "wave", "symbol", "code", "link")}
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
    step merged with the profile's defaults and the load-time overrides (README.dev.md, Sync methods)"""
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

def _stage_class(path):
    """The class a "stage" step names by module path ("rpps.signal.DCBlock", "mypkg.dsp.Gain")"""
    import importlib
    from ..base import Stage
    mod, _, name = str(path).rpartition(".")
    try:
        cls = getattr(importlib.import_module(mod), name) if mod else None
    except (ImportError, AttributeError):
        cls = None
    if not (isinstance(cls, type) and issubclass(cls, Stage)):
        raise ConfigError(f"stage step: {path!r} is not a Stage class (a module path, e.g. rpps.signal.DCBlock)")
    return cls

# ---- signal (receive only)

@register_step("signal", "filter")
def _signal_filter(kind, spec, rep, ctx):
    from .. import signal
    return [(STAGE, signal.filters.load(_need(spec, "type", "signal filter"), _need(spec, "name", "signal filter")))], [], rep

@register_step("signal", "decimate")
@_one
def _decimate(spec, ctx):
    from .. import signal
    return signal.sample.down.Decimate(_need(spec, "ratio", "signal decimate"), spec.get("offset", 0))

@register_step("signal", "dc_block")
@_one
def _dc_block(spec, ctx):
    from .. import signal
    return signal.DCBlock(**_given(spec, ("r",)))

@register_step("signal", "deemphasis")
@_one
def _deemphasis(spec, ctx):
    from .. import signal
    return signal.Deemphasis(**_given(spec, ("tau",)))

@register_step("signal", "shift")
@_one
def _shift(spec, ctx):
    from .. import signal
    return signal.filters.ShiftFreq(_need(spec, "freq", "signal shift"))

@register_step("signal", "conjugate")
@_one
def _conjugate(spec, ctx):
    """Spectral inversion of the capture (I and Q swapped, a lower-sideband mixer)"""
    from .. import signal
    return signal.Conjugate()

@register_step("signal", "resample")
@_one
def _resample(spec, ctx):
    """Rational resampling by up/down (a capture at 2.5 MS/s to 4 samples a symbol at 500 kBd: up 4, down 5)"""
    from .. import signal
    return signal.sample.re.Polyphase(_need(spec, "up", "signal resample"), _need(spec, "down", "signal resample"), spec.get("taps"))

@register_step("signal", "channelize")
@_one
def _channelize(spec, ctx):
    """One channel of a wider capture to baseband at about `oversample` times its width: {"signal": "channelize", "freq": -250e3,
    "width": 200e3}. The stages are designed from the stream's sample rate when it starts"""
    from .. import signal
    return signal.Channelize(spec.get("freq", 0.0), _need(spec, "width", "signal channelize"), spec.get("oversample", 3.0))

def _designed_fir(kind, spec, rep, ctx):
    """A windowed FIR designed in Hz from the stream's sample rate: {"signal": "lowpass", "cutoff": 100e3, "taps": 101} or
    {"signal": "bandpass", "cutoff": [50e3, 150e3], "transition": 20e3, "atten_db": 60} (a Kaiser window meeting that)"""
    from ..signal.filters.serial import _windowed, Designed
    obj = {k: v for k, v in spec.items() if k != "signal"}
    obj["pass"], obj["units"] = kind, "hz"
    _need(obj, "cutoff", f"signal {kind}")
    return [(STAGE, Designed(kind, _windowed(kind, obj)))], [], rep

for _k in ("lowpass", "highpass", "bandpass", "bandstop"):
    register_step("signal", _k)(_designed_fir)

@register_step("signal", "stage")
def _signal_stage(kind, spec, rep, ctx):
    """Any Stage, by its class's module path: {"signal": "stage", "class": "mypkg.Gain", "args": {"k": 2}}"""
    return [(STAGE, _stage_class(_need(spec, "class", "stage step"))(**spec.get("args", {})))], [], rep

# ---- sync (receive only)

def _sync_method(kind, spec, ctx):
    """The method for a sync step: the step's own, else the load-time override, else the profile's defaults, else ours"""
    chosen = {**_SYNC_DEFAULTS.get(kind, {}), **ctx["defaults"].get(kind, {}), **ctx["overrides"].get(kind, {})}
    chosen.update({k: v for k, v in spec.items() if k not in ("sync",)})
    return chosen

_SYNC_DEFAULTS = {
    "timing": {"method": "oerder_meyr"}, # first class: block estimation tracked across windows (README.dev.md, Sync methods)
    "carrier": {"method": "mpower"},     # first class likewise; the loops (mm, gardner, costas) are there by name
    "align": {"method": "align"},
    "puncture": {"method": "reencode"},
    "pilots": {"method": "ls"},
    "agc": {"method": "agc"},
    "frame": {"method": "preamble"},
    "equalize": {"method": "ls"},        # first class: block least squares per window
    "delimit": {"method": "word"},
}

for _kind in ("timing", "carrier", "equalize", "agc"):
    register_step("sync", _kind)(_by_method(_kind))

def _sps(m, ctx, where):
    sps = m.get("sps", ctx["params"].get("sps"))
    return _need({"sps": sps}, "sps", where) if sps is None else sps

@register_sync_method("timing", "oerder_meyr")
def _om(m, rep, ctx):
    from .. import sync
    return sync.time.OerderMeyr(m.get("sps"), m.get("window", 1e-3), m.get("min_symbols", 2048), m.get("bw", 0.25),
                                m.get("gate", 2.0), m.get("name", "timing"))

@register_sync_method("timing", "mm")
def _mm(m, rep, ctx):
    from .. import sync
    return sync.time.MM(m.get("alpha", 0.3), ted="mm")

@register_sync_method("timing", "gardner")
def _gardner(m, rep, ctx):
    from .. import sync
    return sync.time.MM(m.get("alpha", 0.3), ted="gardner")

@register_sync_method("timing", "tll")
def _tll(m, rep, ctx):
    from .. import sync
    return sync.time.TLL(m.get("sps", ctx["params"].get("sps")), m.get("bw", 0.01), m.get("damping", 0.7071), m.get("name", "timing"))

@register_sync_method("timing", "known")
def _known(m, rep, ctx):
    return _KnownTiming(m.get("sps", ctx["params"].get("sps")))

@register_sync_method("timing", "fixed")
def _fixed(m, rep, ctx):
    from .. import signal
    return signal.sample.down.Decimate(_sps(m, ctx, "fixed timing"), m["offset"])

@register_sync_method("carrier", "pilots")
def _pilot_phase(m, rep, ctx):
    """Data-aided, per frame, from pilot blocks: their layout ("slot", "every", "length", "value"), else PilotPhase's own"""
    from .. import sync
    return sync.PilotPhase(**_given(m, ("slot", "every", "length", "slots", "pilots", "lead", "name"), value=_complex))

@register_sync_method("carrier", "mpower")
def _mpower(m, rep, ctx):
    from .. import sync
    order = int(m.get("order", 4))
    offset = m.get("offset")
    if offset is None:
        from .. import symbol
        offset = sync.phase.offset_of(symbol.load(m["constellation"]), order) if "constellation" in m else 0.0
    return sync.phase.MPower(order, offset, m.get("window", 1e-3), m.get("min_symbols", 1024), m.get("bw", 0.25),
                             m.get("gate", 3.0), m.get("relock", 4), m.get("name", "carrier"))

@register_sync_method("carrier", "costas")
def _costas(m, rep, ctx):
    from .. import sync
    return sync.Costas.from_bandwidth(m.get("bw", 0.01), order=m.get("order", 4))

@register_sync_method("carrier", "ff")
def _ff(m, rep, ctx):
    from .. import sync
    return sync.phase.FFCarrier(m.get("window", 64))

register_sync_method("carrier", "none")(lambda m, rep, ctx: None) # The carrier is already off, or no estimator fits yet

@register_sync_method("equalize", "ls")
def _ls(m, rep, ctx):
    from .. import sync, symbol
    return sync.equalize.BlockLS(symbol.load(_need(m, "constellation", "equalize ls")), m.get("taps", 7), m.get("window", 1e-3),
                                 m.get("min_symbols", 1024), m.get("iters", 3), m.get("blend", 0.5), m.get("gate", 0.5),
                                 m.get("name", "equalizer"))

@register_sync_method("equalize", "lms")
def _lms(m, rep, ctx):
    from .. import sync, symbol
    pts = symbol.load(m["constellation"]).points if "constellation" in m else None
    return sync.equalize.LMS(m.get("taps", 11), m.get("mu", 0.01), points=pts)

@register_sync_method("equalize", "cma")
def _cma(m, rep, ctx):
    from .. import sync
    return sync.equalize.CMA(m.get("taps", 11), m.get("mu", 0.01))

def _sync_one(kind):
    """A sync kind with a single way to do it: make(m, ctx) -> the stage"""
    def deco(make):
        register_step("sync", kind)(lambda _, spec, rep, ctx: ([(STAGE, make(_sync_method(kind, spec, ctx), ctx))], [], rep))
        return make
    return deco

@_sync_one("align")
def _align(m, ctx):
    from .. import sync
    return sync.Align(_need(m, "period", "align"), markers=m.get("markers"), pattern=m.get("pattern"), unit=m.get("unit", 8),
                      polarity=m.get("polarity", True), name=m.get("name", "align"), start=m.get("start", "any"))

@_sync_one("puncture")
def _puncture(m, ctx):
    from .. import sync, code
    return sync.Puncture(code.load(m.get("type", "conv"), _need(m, "fec", "puncture")), m.get("window", 4000))

@_sync_one("pilots")
def _pilot_eq(m, ctx):
    from .. import sync
    carriers = ctx["objects"].get("carriers")
    if carriers is None:
        raise ConfigError("pilots: needs a symbol carriers step after it")
    return sync.PilotEqualizer(carriers, m.get("taps"))

@register_sync_method("agc", "agc")
def _agc(m, rep, ctx):
    from .. import sync
    return sync.AGC(m.get("target", 1.0), m.get("mu", 0.01))

@register_sync_method("agc", "peak")
def _peak_agc(m, rep, ctx):
    """The gain from the peak, followed at once (bursts, on-off keying)"""
    from .. import sync
    return sync.PeakAGC(m.get("target", 1.0), m.get("decay", 0.999))

@_sync_one("squelch")
def _squelch(m, ctx):
    from .. import sync
    return sync.Squelch(**_given(m, ("threshold_db", "hysteresis_db", "alpha")))

@_sync_one("ofdm")
def _ofdm(m, ctx):
    """OFDM symbol timing and fractional frequency from the cyclic prefix (sync.CPSync): before a wave ofdm step"""
    from .. import sync
    return sync.CPSync(m.get("n_fft", 64), m.get("n_cp", 16), **_given(m, ("window", "min_symbols", "snr", "bw", "gate", "name")))

@_sync_one("frame")
def _frame(m, ctx):
    """Frames by a known preamble among symbols: the one a physical header definition starts with"""
    from .. import sync, symbol
    h = symbol.load_header(_need(m, "header", "sync frame"))
    return sync.FrameSync(h.preamble, h.words() if m.get("words", True) else None, m.get("threshold"), m.get("max_length"),
                          m.get("name", "frame"))

def _word(m, where):
    """(word, bits, order) a step gives: a `sequence` of type word, or `word` + `bits` (+ `order`) inline"""
    if "sequence" in m:
        from .. import sequence
        w = sequence.load(m["sequence"])
        if not hasattr(w, "bits") or not hasattr(w, "value"):
            raise ConfigError(f"{where}: sequence {m['sequence']!r} is not a word")
        return w.bits, None, "msb"
    return _need(m, "word", where), _need(m, "bits", where), m.get("order", "msb")

@_sync_one("delimit")
def _delimit(m, ctx):
    """Frames by the word before them among bits: a link header's preamble word, a word sequence, or one given inline;
    `length` (bits per frame after the word, else until the next word or a header's control), `keep` (give the word too,
    for a code marker step to strip), `polarity` (match the inverted word too and put the frame's sense right)"""
    from .. import sync
    if "header" in m:
        from .. import link
        pre = _config.read(_config.find("header", m["header"], "link")).get("preamble") or {}
        h = link.load("header", m["header"])
        word, bits, order = pre.get("word"), pre.get("bits"), h.order
        if word is None:
            raise ConfigError(f"sync delimit: {m.get('header')} has no preamble word")
    else:
        word, bits, order = _word(m, "sync delimit")
    return sync.Delimiter(word, bits, order, m.get("errors", 0), m.get("name", m.get("id", "delimiter")), m.get("length"),
                          m.get("keep", False), m.get("polarity", False))

def _given(spec, keys, **convert):
    """The operator's arguments the step gives (the operator's defaults stand for the rest); convert: {key: function}"""
    out = {k: spec[k] for k in keys if k in spec}
    out.update({k: f(spec[k]) for k, f in convert.items() if k in spec})
    return out

def _complex(v):
    return complex(v[0], v[1]) if isinstance(v, (list, tuple)) else complex(v)

from ..base import Stage as _Stage

class _KnownTiming(_Stage):
    """Timing that is known, not estimated (a loopback, a source whose timing is given): one sample per symbol at the pulse
    peaks, which the group delay so far (meta.delay) places. Tracks the absolute input index of the next peak, so any block
    cut gives the same output.

    With a matched filter fused in (`fuse`, which a profile does for the filter just before it) the filter is evaluated only at
    those instants, a dot product of its taps with the signal per symbol: `sps` times less work than filtering every sample and
    keeping every `sps`-th, the same symbols. The estimators that need the filter's output at every sample (Oerder & Meyr) have
    nothing to fuse"""
    def __init__(self, sps):
        self.sps = sps
        self.matched = None

    def fuse(self, matched):
        """Take over a matched filter (a FIR stage): this stage then also filters, at the instants it samples"""
        self.matched = matched

    def __str__(self):
        return "KnownTiming" if self.matched is None else "KnownTiming(matched)"

    def new_state(self, meta):
        from types import SimpleNamespace
        taps = None if self.matched is None else self.matched.new_state(meta).taps
        return SimpleNamespace(next=None, taps=taps, hist=None)

    def step(self, samples, st, meta):
        import numpy as np
        sps = int(self.sps or meta.isps)
        delay = meta.delay + (0.0 if st.taps is None else (len(st.taps) - 1)/2/meta.Fs) # (the filter's group delay)
        if st.next is None:
            first = int(round(delay*meta.Fs)) # absolute index of the first symbol's peak
            st.next = first if first >= meta.pos else first + -(-(meta.pos - first)//sps)*sps
        first_idx = st.next - meta.pos
        count = max(0, -(-(len(samples) - first_idx)//sps))
        st.next += count*sps
        meta.op.Fs = meta.Fs/sps
        meta.op.sps = 1
        if st.taps is None:
            return samples[np.arange(first_idx, len(samples), sps)]
        meta.delay = delay
        return self._matched(samples, st, first_idx, sps, count)

    @staticmethod
    def _matched(samples, st, first_idx, sps, count):
        import numpy as np
        from .. import _accel
        nh = len(st.taps)
        keep = nh - 1
        x = np.asarray(samples)
        hist = st.hist if st.hist is not None else np.zeros(keep, dtype=x.dtype)
        buf = np.concatenate((hist, x))
        st.hist = buf[len(buf) - keep:]
        if not count:
            return np.zeros(0, np.complex64)
        if _accel.native is not None:
            return _accel.native.fir_decimate(np.ascontiguousarray(buf, dtype=np.complex64), st.taps, first_idx + keep, sps, count)
        from ..signal.filters.convolve import convolve
        return convolve(buf, st.taps, "valid")[first_idx + sps*np.arange(count)].astype(np.complex64)

# ---- wave (pairs)

def _pulse(name):
    """The pulse resource and whether it is a symbol pulse or a frequency pulse"""
    from .. import signal
    f = _config.find("pulse", name)
    if f is None:
        raise ConfigError(f"wave shape: no pulse {name!r} in config/pulse")
    kind = _config.read(f).get("kind", "symbol")
    return (lambda: signal.filters.load("pulse", name)), kind

def _wave(kind, spec, rep, ctx):
    from .. import wave, signal
    sps = spec.get("sps", ctx["params"].get("sps"))
    if kind == "shape":
        make, pulse_kind = _pulse(_need(spec, "pulse", "wave shape"))
        tx = [(STAGE, signal.sample.up.Interpolate(int(_need({"sps": sps}, "sps", "wave shape")), make()))] # Zeros + pulse, polyphase
        rx = []
        if pulse_kind == "symbol": # (a frequency pulse has no matched filter)
            f = make()
            f.matched = True # unit gain at the sampling instants, so the symbols come back at the amplitude they were sent
            rx = [(STAGE, f)]
        return rx, tx, Rep.SAMPLES
    if kind == "stagger":
        d = spec.get("delay", 0.5)
        return [(STAGE, wave.Stagger("i", d))], [(STAGE, wave.Stagger("q", d))], Rep.SAMPLES
    if kind == "fm":
        if "deviation" in spec: # Analog: the levels in, the levels out (audio), nothing per symbol
            dev = spec["deviation"]
            return [(STAGE, wave.FMDemod(deviation=dev))], [(STAGE, wave.FM(deviation=dev))], Rep.SAMPLES
        h, step = _need(spec, "h", "wave fm"), spec.get("step", 2.0)
        sps = _need({"sps": sps}, "sps", "wave fm")
        rx = [(STAGE, wave.FMDemod(wave.fm_gain(h, sps, step))),
              (STAGE, signal.filters.ShapingRect(int(sps), int(sps), 1/int(sps)))]
        return rx, [(STAGE, wave.FM(h, step))], Rep.SAMPLES
    if kind == "am": # The envelope: on transmit the levels through their pulse are the envelope already
        return [(STAGE, wave.AMDemod(spec.get("alpha", 1e-3)))], [], Rep.SAMPLES
    if kind == "ssb":
        return [(STAGE, wave.SSBDemod(spec.get("bw", 2800.0), spec.get("sideband", "usb"), spec.get("taps", 201)))], [], Rep.SAMPLES
    if kind == "ofdm":
        n, cp = spec.get("n_fft", 64), spec.get("n_cp", 16)
        return [(STAGE, wave.OFDMRx(n, cp))], [(STAGE, wave.OFDMTx(n, cp))], Rep.SYMBOLS

# ---- symbol, code, link (operators that encode and decode)

def _pair(op_rx, op_tx, gives):
    return [(DECODE, op_rx)], [(ENCODE, op_tx)], gives

def _symbol(kind, spec, rep, ctx):
    from .. import symbol, sequence
    if kind == "constellation":
        name = _need(spec, "name", "symbol constellation")
        make = lambda: symbol.load(name, spec.get("mapping"))
        return _pair(make(), make(), Rep.CHANNEL)
    if kind == "differential":
        d = symbol.Differential()
        return [(DECODE, d)], [(ENCODE, symbol.Differential())], Rep.SYMBOLS
    if kind == "spread":
        code = sequence.load(_need(spec, "sequence", "symbol spread"), **spec.get("args", {}))
        make = lambda: symbol.Spread(code, spec.get("pair", False))
        return _pair(make(), make(), Rep.CHANNEL if code.count > 1 else Rep.SYMBOLS)
    if kind == "carriers":
        c = symbol.Carriers(**_given(spec, ("n_fft", "used"), pilots=tuple))
        ctx["objects"]["carriers"] = c
        return [(DECODE, c)], [(ENCODE, c)], Rep.SYMBOLS
    if kind == "header":
        name = _need(spec, "name", "symbol header")
        return _pair(symbol.load_header(name), symbol.load_header(name), Rep.SYMBOLS)
    if kind == "pilots":
        make = lambda: symbol.Pilots(**_given(spec, ("slot", "every", "length", "slots", "pilots"), value=_complex),
                                     name=spec.get("id", "pilots"))
        return _pair(make(), make(), Rep.SYMBOLS)
    if kind == "scramble":
        from .. import sequence
        ref = _need(spec, "sequence", "symbol scramble")
        make = lambda: symbol.PhaseScramble(sequence.load(ref, **spec.get("args", {})), spec.get("offsets", [0]), spec.get("step", 90),
                                            spec.get("length"), spec.get("id", "scramble"))
        return _pair(make(), make(), Rep.SYMBOLS)

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
        ref = _need(spec, "name", "code scramble")
        def make():
            op = code.scram.load(ref, **spec.get("args", {}))
            if "length" in spec: # Frames of a fixed length from the first bit: the sequence restarts at each (a CCSDS frame)
                op.control(length=spec["length"])
            return op
        return _pair(make(), make(), Rep.DATA)
    if kind == "uncoded":
        return _pair(code.Uncoded(), code.Uncoded(), Rep.DATA)
    if kind == "line":
        make = lambda: code.Line(_need(spec, "name", "code line"))
        return _pair(make(), make(), Rep.DATA)
    if kind == "marker":
        word, bits, order = _word(spec, "code marker")
        make = lambda: code.Marker(word, bits, spec.get("length"), order)
        return _pair(make(), make(), Rep.DATA)

def _link(kind, spec, rep, ctx):
    from .. import link
    if kind in ("crc", "frame", "header"):
        name = _need(spec, "name", f"link {kind}")
        make = lambda: link.load(kind, name)
        return _pair(make(), make(), Rep.PACKETS)

for _cat, _fn, _kinds in (("wave", _wave, ("shape", "stagger", "fm", "am", "ssb", "ofdm")),
                          ("symbol", _symbol, ("constellation", "differential", "spread", "carriers", "header", "pilots", "scramble")),
                          ("code", _code, ("fec", "scramble", "uncoded", "line", "marker")),
                          ("link", _link, ("crc", "frame", "header"))):
    for _k in _kinds:
        register_step(_cat, _k)(_fn)

def build(spec, rep, ctx):
    """Build one step that receives `rep`. ctx: params, defaults, overrides and objects built so far (see `context`)"""
    cats = [k for k in spec if k in REGISTRY]
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
