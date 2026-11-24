"""What each step of a profile builds (rpps.profile)

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
             dc_block, deemphasis
    every    class: your own operator by module path, in the category it does ("class", "args"; wave: "tx_class")
    sync     timing, carrier, equalize, agc (methods), frame (a physical header's preamble, a word or a complex
             sequence, or named alternatives / a Zadoff-Chu set; before, length, segments, scale, polarity, classify), delimit (a word among bits:
             word + bits, a word sequence, or a link header's preamble; length, keep, polarity, errors), align (a repeating
             marker), puncture, pilots, squelch, quality (Es/N0 and EVM to est), ofdm
    wave     shape (a pulse definition; symbol pulses get a matched filter), stagger, fm (h + sps: FSK; deviation: analog),
             am (the envelope; alpha null keeps its mean), ssb (bw, sideband), ofdm (n_fft, n_cp)
    symbol   constellation (name, mapping), differential, spread (a chip sequence), carriers, header, pilots, scramble,
             marker (known symbols before each frame: a `sequence` with symbols, or `zc` {length, root, shift}; root,
             shift, sequence controlled per frame), puncture (symbols at `at` of every `every` from the frame start,
             dropped; `value` inserted on transmit: P25's status symbols)
    code     fec (type block / conv / interleave, name, args), scramble (name, args; length: frames of that many bits from
             the first), line (invert, nrzi, nrzm, manchester, thomas), marker (a word before each frame: length, or frames by
             control), uncoded
    link     crc, frame (HDLC), header (fields, crc, fcs, payload; select: the frames with those field values)
    grid     resource grids (rpps.grid), receive only: lte_cell (offset, span, resolution, acquire, lose, gate), lte_grid
             (n_rb), and the channels on its frames, lte_pbch, lte_pcfich, lte_pdcch (rntis), lte_pdsch; nr_cell (offset,
             span, resolution, period, acquire, lose, gate, fc), nr_grid, nr_pbch, nr_pdcch (rntis), nr_pdsch
"""
from dataclasses import dataclass, field

import numpy as np

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
    from ..base.types import Category
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
    from ..base import Stage, Pipe
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

# ---- signal (receive only)

@register_step("signal", "filter")
def _signal_filter(kind, spec, rep, ctx):
    from .. import signal
    return [(STAGE, signal.filters.load(_need(spec, "name", "signal filter")))], [], rep # "fir/...", "iir/..."

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

for _cat in ("signal", "sync", "wave", "symbol", "code", "link"):
    register_step(_cat, "class")(_class_step)

# ---- sync (receive only)

def _sync_method(kind, spec, ctx):
    """The method for a sync step: the step's own, else the load-time override, else the profile's defaults, else ours"""
    chosen = {**_SYNC_DEFAULTS.get(kind, {}), **ctx["defaults"].get(kind, {}), **ctx["overrides"].get(kind, {})}
    chosen.update({k: v for k, v in spec.items() if k not in ("sync",)})
    return chosen

_SYNC_DEFAULTS = {
    "timing": {"method": "oerder_meyr"}, # first class: block estimation tracked across windows (rpps.sync)
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
    """Mueller & Muller: "alpha" (loop gain), "interp" (interpolation factor), "biphase" (Manchester symbols, timed on
    their mid-symbol transition: RDS)"""
    from .. import sync
    return sync.time.MM(m.get("alpha", 0.3), interp=m.get("interp", 32), biphase=m.get("biphase", False), ted="mm")

@register_sync_method("timing", "gardner")
def _gardner(m, rep, ctx):
    from .. import sync
    return sync.time.MM(m.get("alpha", 0.3), ted="gardner")

@register_sync_method("timing", "tll")
def _tll(m, rep, ctx):
    from .. import sync
    return sync.time.TLL(m.get("sps", ctx["params"].get("sps")), m.get("bw", 0.01), m.get("damping", 0.7071), m.get("name", "timing"),
                         m.get("ted", "zx"))

@register_sync_method("timing", "mat")
def _mat(m, rep, ctx):
    """The matched filter and timing in one (rpps.sync.time.MatTrack): the profile fuses the `wave shape` step's matched
    filter into it. "device": true evaluates it on the GPU of rpps._accel.device()"""
    from .. import sync
    return sync.time.MatTrack(m.get("sps"), m.get("window", 1e-3), m.get("min_symbols", 2048), m.get("bw", 0.25),
                              m.get("delta"), m.get("grid", 16), m.get("gate", 0.02), m.get("relock", 4), m.get("group", 16),
                              m.get("device", False), m.get("name", "timing"))

@register_sync_method("timing", "known")
def _known(m, rep, ctx):
    """Known timing; `anchor`: the name of a preamble step whose find starts the symbol grid (one grid per burst)"""
    from .. import sync
    return sync.time.Known(m.get("sps", ctx["params"].get("sps")), anchor=m.get("anchor"))

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
        c = (symbol.load(m["constellation"]) if "constellation" in m else symbol.generate(m["shape"]) if "shape" in m
             else None) # A definition, or a generated shape, as the constellation step names them
        offset = sync.phase.offset_of(c, order) if c is not None else 0.0
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

@register_step("sync", "burst")
def _burst(kind, spec, rep, ctx):
    """Per-burst demodulation (rpps.profile.burst): the stream cut into its bursts, the steps after this one built anew for
    each. {"sync": "burst", "factor": 6, "win": 8, "min_s": 60e-6, "max_s": 3e-3} (rpps.sync.Burst's settings).
    Transmitting (rpps.profile.burst.BurstTxRunner, rpps.sync.BurstTx): "gap_s" between bursts, "ramp_s" power ramps,
    "Fs" the output rate (null: the bursts' own), "hop" carrier offsets [Hz] cycled burst by burst"""
    from .. import sync
    if rep is not None and rep != Rep.SAMPLES:
        raise ConfigError(f"sync burst: cuts samples, not {rep}")
    op = sync.Burst(spec.get("factor", 6.0), spec.get("win", 100), spec.get("min_s", 0.0), spec.get("max_s"),
                    spec.get("percentile", 20), spec.get("history", 1 << 17), spec.get("settle", 512), spec.get("quiet", 3.0))
    hop = spec.get("hop")
    if hop is not None and (not isinstance(hop, list) or not hop):
        raise ConfigError("sync burst: hop is a list of carrier offsets [Hz]")
    try:
        tx = sync.BurstTx(spec.get("Fs"), spec.get("gap_s", 0.0), spec.get("ramp_s", 0.0), hop)
    except ValueError as e:
        raise ConfigError(f"sync burst: {e}") from None
    return [(STAGE, op)], [(STAGE, tx)], Rep.SAMPLES

@register_step("sync", "preamble")
def _preamble(kind, spec, rep, ctx):
    """A burst's preamble (rpps.sync.Preamble): by its repetition ({"repeat": 16, "window": 128}), or by its waveform, made
    from bits (a `sequence` word, or `word` + `bits`, after `lead` bits) through the profile's own transmit side, from the
    steps after this one down to the symbol mapping (the profile builds it: rpps.profile.Profile._templates). Timing
    {"method": "known", "anchor": <its name>} then places the symbols on what it found.
    "attach": true transmits it (Preamble.attach_tx) before every stream (burst): for a preamble no later step sends
    (a marker step sends a word of its own)"""
    from .. import sync
    m = dict(spec)
    named = "sequence" in m or "word" in m
    if not named and "repeat" not in m:
        raise ConfigError("sync preamble: needs a `sequence` (or `word` and `bits`) or a `repeat`")
    if m.get("attach") and not named:
        raise ConfigError("sync preamble: attach sends its waveform, made from a `sequence` or `word` (a repeat alone has none)")
    op = sync.Preamble(None, m.get("repeat"), m.get("window"), m.get("search"), m.get("gate", 0.5),
                       m.get("offset", 0), m.get("correct", False), m.get("name", "preamble"), pending=named)
    return [(STAGE, op)], [(STAGE, op.attach_tx())] if m.get("attach") else [], rep

@_sync_one("align")
def _align(m, ctx):
    from .. import sync
    return sync.Align(_need(m, "period", "align"), markers=m.get("markers"), pattern=m.get("pattern"), unit=m.get("unit", 8),
                      polarity=m.get("polarity", True), name=m.get("name", "align"), start=m.get("start", "any"))

@_sync_one("puncture")
def _puncture(m, ctx):
    from .. import sync, code
    return sync.Puncture(code.load(f'conv/{_need(m, "conv", "puncture")}'), m.get("window", 4000)) # {"sync": "puncture", "conv": "k7_171_133/3_4"}

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

@_sync_one("quality")
def _quality(m, ctx):
    """Es/N0, Eb/N0 and EVM of locked symbols, published to est[name] (sync.quality.Monitor); the symbols pass unchanged.
    constellation: the definition they sit on (default: the one Acquire detected)"""
    from .. import sync, symbol
    pts = m.get("constellation")
    pts = None if pts is None else symbol.load(pts).points
    return sync.quality.Monitor(pts, **_given(m, ("bits", "rate", "history", "minimum", "name")))

@_sync_one("ofdm")
def _ofdm(m, ctx):
    """OFDM symbol timing and fractional frequency from the cyclic prefix (sync.CPSync): before a wave ofdm step"""
    from .. import sync
    return sync.CPSync(m.get("n_fft", 64), m.get("n_cp", 16), **_given(m, ("window", "min_symbols", "snr", "bw", "gate", "name")))

_FRAME_KEYS = ("before", "length", "segments", "scale", "polarity", "cfo", "classify", "margin", "min_symbols")

def frame_words(m, where="sync frame"):
    """The preambles of a frame step: [(name, kind, value, fields)], kind "symbols" (a complex sequence: a Zadoff-Chu
    definition, the "zc" set) or "bits" (a word the profile maps to symbols through its transmit side). name None: the
    one preamble of `sequence` / `word`"""
    from .. import sequence
    def one(name, spec):
        if isinstance(spec, str):
            spec = {"sequence": spec}
        fields = dict(spec.get("fields", {}))
        if "sequence" in spec:
            s = sequence.load(spec["sequence"])
            if hasattr(s, "symbols"):
                return (name, "symbols", np.asarray(s.symbols), {**getattr(s, "fields", {}), **fields})
            if not hasattr(s, "bits") or not hasattr(s, "value"):
                raise ConfigError(f"{where}: sequence {spec['sequence']!r} is neither a word nor complex symbols")
            return (name, "bits", np.asarray(s.bits, bool), fields)
        from ..sequence.word import word_bits
        w, bits, order = _need(spec, "word", where), _need(spec, "bits", where), spec.get("order", "msb")
        return (name, "bits", np.asarray(word_bits(w, bits, order), bool), fields)
    out = []
    if "sequence" in m or "word" in m:
        out.append(one(None, {k: m[k] for k in ("sequence", "word", "bits", "order") if k in m}))
    for name, spec in (m.get("sequences") or {}).items():
        out.append(one(str(name), spec))
    if m.get("zc"):
        from ..sequence import zadoff_chu
        z = m["zc"]
        n = int(_need(z, "length", f"{where} zc"))
        for u in _need(z, "roots", f"{where} zc"):
            for sh in z.get("shifts", [0]):
                out.append((f"{u}_{sh}", "symbols", zadoff_chu(u, n, sh), {"root": int(u), "shift": int(sh)}))
    if len(out) > 1 and any(n is None for n, *_ in out):
        raise ConfigError(f"{where}: one preamble (`sequence` / `word`) or named alternatives (`sequences`, `zc`), not both")
    return out

@_sync_one("frame")
def _frame(m, ctx):
    """Frames by a known preamble among symbols: the one a physical header definition starts with ("header"); a
    `sequence` (a word, or complex symbols: a Zadoff-Chu definition) or a `word` given as bits; or alternatives, named
    (`sequences`: {name: sequence, or {"sequence" | "word" + "bits", "fields"}}) or a Zadoff-Chu set (`zc`: roots,
    length, shifts), the one found published as `word` with its fields. Words given as bits become symbols through the
    profile's own transmit side, from the steps after this one down to the symbol mapping
    (rpps.profile.Profile._template), so any rotation of the constellation is resolved by the word. Also `before`,
    `length`, `segments`, `scale`, `polarity`, `classify` (rpps.sync.FrameSync)"""
    from .. import sync, symbol
    words = frame_words(m)
    if bool(words) == ("header" in m):
        raise ConfigError("sync frame: a `header`, or a word / sequence (`sequence`, `word` and `bits`, `sequences`, `zc`), one of the two")
    kw = dict(threshold=m.get("threshold"), max_length=m.get("max_length"), name=m.get("name", "frame"),
              **{k: m[k] for k in _FRAME_KEYS if k in m})
    if words:
        if any(kind == "bits" for _, kind, _, _ in words):
            return sync.FrameSync(None, None, pending=True, **kw) # (the profile makes the symbols)
        if words[0][0] is None:
            return sync.FrameSync(words[0][2], **kw)
        return sync.FrameSync({n: (v, f) for n, _, v, f in words}, **kw)
    h = symbol.load_header(m["header"])
    return sync.FrameSync(h.preamble, h.words() if m.get("words", True) else None, device=m.get("device", False), **kw)

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
        h = link.load(f"header/{m['header']}")
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

# ---- grid (receive only): a resource grid's cell search, its frames, the channels read off them

@register_step("grid", "lte_cell")
def _lte_cell(kind, spec, rep, ctx):
    """LTE cell search and half-frame timing (rpps.grid.lte.LteCell): samples from the first subframe 0, in half-frames"""
    from ..grid import lte
    if rep is not None and rep != Rep.SAMPLES:
        raise ConfigError(f"grid lte_cell: takes samples, not {rep}")
    return [(STAGE, lte.LteCell(**_given(spec, ("offset", "span", "resolution", "acquire", "lose", "gate", "name"))))], [], Rep.SAMPLES

@register_step("grid", "lte_grid")
def _lte_grid(kind, spec, rep, ctx):
    """Subframes to grid frames with every port's channel (rpps.grid.lte.LteGrid)"""
    from ..grid import lte
    return [(STAGE, lte.LteGrid(**_given(spec, ("n_rb", "cell"))))], [], Rep.SYMBOLS

def _lte_channel(kind, spec, rep, ctx):
    """A channel on LTE grid frames (rpps.grid.lte.channels): gives bits, publishes what it decoded"""
    from ..grid import lte
    name = spec.get("id", kind)
    if kind == "lte_pbch":
        op = lte.PBCH(name=name)
    elif kind == "lte_pcfich":
        op = lte.PCFICH(name=name)
    elif kind == "lte_pdcch":
        op = lte.PDCCH(**_given(spec, ("rntis",)), name=name)
    else:
        op = lte.PDSCH(name=name)
    return [(STAGE, op)], [], Rep.DATA

for _k in ("lte_pbch", "lte_pcfich", "lte_pdcch", "lte_pdsch"):
    register_step("grid", _k)(_lte_channel)

@register_step("grid", "nr_cell")
def _nr_cell(kind, spec, rep, ctx):
    """NR cell search and slot timing, LEO Doppler tracked (rpps.grid.nr.NrCell): samples from a frame start, in slots"""
    from ..grid import nr
    if rep is not None and rep != Rep.SAMPLES:
        raise ConfigError(f"grid nr_cell: takes samples, not {rep}")
    return [(STAGE, nr.NrCell(**_given(spec, ("offset", "span", "resolution", "period", "acquire", "lose", "gate", "fc", "name"))))], [], Rep.SAMPLES

@register_step("grid", "nr_grid")
def _nr_grid(kind, spec, rep, ctx):
    """Slots to grid frames (rpps.grid.nr.NrGrid)"""
    from ..grid import nr
    return [(STAGE, nr.NrGrid(**_given(spec, ("cell", "period"))))], [], Rep.SYMBOLS

def _nr_channel(kind, spec, rep, ctx):
    """A channel on NR grid frames (rpps.grid.nr.channels)"""
    from ..grid.nr import channels as ch
    name = spec.get("id", kind)
    op = {"nr_pbch": lambda: ch.PBCH(name=name), "nr_pdcch": lambda: ch.PDCCH(**_given(spec, ("rntis",)), name=name),
          "nr_pdsch": lambda: ch.PDSCH(**_given(spec, ("fc",)), name=name)}[kind]()
    return [(STAGE, op)], [], Rep.DATA

for _k in ("nr_pbch", "nr_pdcch", "nr_pdsch"):
    register_step("grid", _k)(_nr_channel)

# ---- wave (pairs)

def _pulse(name):
    """The pulse resource and whether it is a symbol pulse or a frequency pulse"""
    from .. import signal
    f = _config.find("pulse", name)
    if f is None:
        raise ConfigError(f"wave shape: no pulse {name!r} in config/pulse")
    kind = _config.read(f).get("kind", "symbol")
    return (lambda: signal.filters.load(f"pulse/{name}")), kind

def _wave(kind, spec, rep, ctx):
    from .. import wave, signal
    sps = spec.get("sps", ctx["params"].get("sps"))
    if kind == "shape":
        make, pulse_kind = _pulse(_need(spec, "pulse", "wave shape"))
        tx = [(STAGE, wave.Shape(int(_need({"sps": sps}, "sps", "wave shape")), make()))]
        rx = [(STAGE, wave.Matched(make()))] if pulse_kind == "symbol" else [] # (a frequency pulse has no matched filter)
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
              (STAGE, wave.Matched(signal.filters.ShapingRect(int(sps), int(sps), 1/int(sps)), unit=False))] # (integrate over the symbol)
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

def constellation(spec):
    """The constellation of a {"symbol": "constellation"} step: "name", a definition ("psk/qpsk"), or "shape", a generated one
    ("OOK", "16QAM"); "mapping" picks a map"""
    from .. import symbol
    if "shape" in spec:
        c = symbol.generate(spec["shape"])
        return c if spec.get("mapping") is None else c.set_mapping(spec["mapping"])
    return symbol.load(_need(spec, "name", "symbol constellation"), spec.get("mapping"))

def _symbol(kind, spec, rep, ctx):
    from .. import symbol, sequence
    if kind == "constellation":
        make = lambda: constellation(spec)
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
    if kind == "puncture": # {"symbol": "puncture", "every": 36, "at": [35], "value": -1}: P25's status symbols
        make = lambda: symbol.Puncture(_need(spec, "every", "symbol puncture"), _need(spec, "at", "symbol puncture"),
                                       _complex(spec.get("value", 0)), spec.get("id", "puncture"))
        return _pair(make(), make(), Rep.SYMBOLS)
    if kind == "marker": # Known symbols before each frame: a sequence with symbols, or a Zadoff-Chu length + root (controlled)
        def make():
            syms = None
            if "sequence" in spec:
                s = sequence.load(spec["sequence"])
                if not hasattr(s, "symbols"):
                    raise ConfigError(f"symbol marker: sequence {spec['sequence']!r} has no symbols (a bit word is `code marker`)")
                syms = s.symbols
            z = spec.get("zc") or {}
            return symbol.Marker(syms, spec.get("length"), z.get("length"), z.get("root"), z.get("shift", 0),
                                 spec.get("id", "marker"))
        return _pair(make(), make(), Rep.SYMBOLS)
    if kind == "scramble":
        from .. import sequence
        ref = _need(spec, "sequence", "symbol scramble")
        make = lambda: symbol.PhaseScramble(sequence.load(ref, **spec.get("args", {})), spec.get("offsets", [0]), spec.get("step", 90),
                                            spec.get("length"), spec.get("id", "scramble"))
        return _pair(make(), make(), Rep.SYMBOLS)

def _code(kind, spec, rep, ctx):
    from .. import code
    if kind in ("block", "conv", "interleave"): # {"code": "conv", "name": "k7_171_133/3_4"}; a null name: a header names it
        folder, name = kind, _need(spec, "name", f"code {kind}")
        def make():
            if name is None: # Nothing (yet): a header names the code
                op = {"block": code.Block, "interleave": code.Interleaver, "conv": code.Convolutional}[folder](None, None)
                op.folder, op.name = folder, "none"
                return op
            return code.load(f"{folder}/{name}", **spec.get("args", {}))
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
        def make():
            op = link.load(f"{kind}/{name}")
            if kind == "header" and spec.get("select"): # {"select": {"isi": 1}}: only the frames with those field values
                op.select = dict(spec["select"])
            return op
        return _pair(make(), make(), Rep.PACKETS)

for _cat, _fn, _kinds in (("wave", _wave, ("shape", "stagger", "fm", "am", "ssb", "ofdm")),
                          ("symbol", _symbol, ("constellation", "differential", "spread", "carriers", "header", "pilots", "scramble", "marker", "puncture")),
                          ("code", _code, ("block", "conv", "interleave", "scramble", "uncoded", "line", "marker")),
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
