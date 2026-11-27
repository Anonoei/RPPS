"""sync steps (receive only) and the methods of the sync kinds that have them"""
import numpy as np

from ... import _config
from ...base.types import Rep
from ...errors import ConfigError
from ._core import STAGE, register_step, register_sync_method, _by_method, _sync_method, _need, _given, _complex, _word

for _kind in ("timing", "carrier", "equalize", "agc"):
    register_step("sync", _kind)(_by_method(_kind))

def _sps(m, ctx, where):
    sps = m.get("sps", ctx["params"].get("sps"))
    return _need({"sps": sps}, "sps", where) if sps is None else sps

@register_sync_method("timing", "oerder_meyr")
def _om(m, rep, ctx):
    """Oerder & Meyr, block estimation tracked across windows (rpps.sync.time.OerderMeyr). "device": true measures and
    interpolates on the GPU of rpps._accel.device()"""
    from ... import sync
    return sync.time.OerderMeyr(m.get("sps"), m.get("window", 1e-3), m.get("min_symbols", 2048), m.get("bw", 0.25),
                                m.get("gate", 2.0), m.get("name", "timing"), m.get("clock", True), m.get("device", False))

@register_sync_method("timing", "mm")
def _mm(m, rep, ctx):
    """Mueller & Muller, a per-symbol loop (second class, rpps.sync.loops.MM): "alpha" (loop gain), "interp"
    (interpolation factor), "biphase" (Manchester symbols, timed on their mid-symbol transition: RDS)"""
    from ... import sync
    return sync.loops.MM(m.get("alpha", 0.3), interp=m.get("interp", 32), biphase=m.get("biphase", False), ted="mm")

@register_sync_method("timing", "gardner")
def _gardner(m, rep, ctx):
    """Gardner, a per-symbol loop (second class, rpps.sync.loops.MM): "alpha" (loop gain)"""
    from ... import sync
    return sync.loops.MM(m.get("alpha", 0.3), ted="gardner")

@register_sync_method("timing", "tll")
def _tll(m, rep, ctx):
    """A timing locked loop (second class, rpps.sync.loops.TLL): "bw" (of the symbol rate), "damping", "ted" (zx, el)"""
    from ... import sync
    return sync.loops.TLL(m.get("sps", ctx["params"].get("sps")), m.get("bw", 0.01), m.get("damping", 0.7071),
                          m.get("name", "timing"), m.get("ted", "zx"))

@register_sync_method("timing", "mat")
def _mat(m, rep, ctx):
    """The matched filter and timing in one (rpps.sync.time.MatTrack): the profile fuses the `wave shape` step's matched
    filter into it. "device": true evaluates it on the GPU of rpps._accel.device()"""
    from ... import sync
    return sync.time.MatTrack(m.get("sps"), m.get("window", 1e-3), m.get("min_symbols", 2048), m.get("bw", 0.25),
                              m.get("delta"), m.get("grid", 16), m.get("gate", 0.02), m.get("relock", 4), m.get("group", 16),
                              m.get("device", False), m.get("name", "timing"))

@register_sync_method("timing", "known")
def _known(m, rep, ctx):
    """Known timing; `anchor`: the name of a preamble step whose find starts the symbol grid (one grid per burst)"""
    from ... import sync
    return sync.time.Known(m.get("sps", ctx["params"].get("sps")), anchor=m.get("anchor"))

@register_sync_method("timing", "fixed")
def _fixed(m, rep, ctx):
    from ... import signal
    return signal.sample.down.Decimate(_sps(m, ctx, "fixed timing"), m["offset"])

@register_sync_method("carrier", "pilots")
def _pilot_phase(m, rep, ctx):
    """Data-aided, per frame, from pilot blocks: their layout ("slot", "every", "length", "value"), else PilotPhase's own"""
    from ... import sync
    return sync.PilotPhase(**_given(m, ("slot", "every", "length", "slots", "pilots", "lead", "name"), value=_complex))

@register_sync_method("carrier", "mpower")
def _mpower(m, rep, ctx):
    """The M-th power, block estimation tracked across windows (rpps.sync.phase.MPower). "device": true measures and
    corrects on the GPU of rpps._accel.device()"""
    from ... import sync
    order = int(m.get("order", 4))
    offset = m.get("offset")
    if offset is None:
        from ... import symbol
        c = (symbol.load(m["constellation"]) if "constellation" in m else symbol.generate(m["shape"]) if "shape" in m
             else None) # A definition, or a generated shape, as the constellation step names them
        offset = sync.phase.offset_of(c, order) if c is not None else 0.0
    return sync.phase.MPower(order, offset, m.get("window", 1e-3), m.get("min_symbols", 1024), m.get("bw", 0.25),
                             m.get("gate", 3.0), m.get("relock", 4), m.get("name", "carrier"), m.get("device", False),
                             m.get("acquire", 16384))

@register_sync_method("carrier", "costas")
def _costas(m, rep, ctx):
    """The Costas loop, per symbol (second class, rpps.sync.loops.Costas): "bw" (of the symbol rate), "order" """
    from ... import sync
    return sync.loops.Costas.from_bandwidth(m.get("bw", 0.01), order=m.get("order", 4))

@register_sync_method("carrier", "dd")
def _dd(m, rep, ctx):
    """From decisions on a constellation, block estimation (rpps.sync.phase.DDCarrier): "constellation" (a name),
    "window" (symbols)"""
    from ... import sync, symbol
    _need(m, "constellation", "carrier dd")
    return sync.phase.DDCarrier(symbol.load(m["constellation"]).points, m.get("window", 32), m.get("bw", 0.25), m.get("name", "carrier"))

@register_sync_method("carrier", "ff")
def _ff(m, rep, ctx):
    from ... import sync
    return sync.phase.FFCarrier(m.get("window", 64))

register_sync_method("carrier", "none")(lambda m, rep, ctx: None) # The carrier is already off, or no estimator fits yet

@register_sync_method("equalize", "ls")
def _ls(m, rep, ctx):
    """Block least squares per window (rpps.sync.equalize.BlockLS): "constellation", "taps", "window", "blind" (constant
    modulus until the eye opens)"""
    from ... import sync, symbol
    return sync.equalize.BlockLS(symbol.load(_need(m, "constellation", "equalize ls")), m.get("taps", 7), m.get("window", 1e-3),
                                 m.get("min_symbols", 1024), m.get("iters", 3), m.get("blend", 0.5), m.get("gate", 0.5),
                                 m.get("name", "equalizer"), m.get("blind", False), m.get("blind_iters", 8))

@register_sync_method("equalize", "lms")
def _lms(m, rep, ctx):
    from ... import sync, symbol
    pts = symbol.load(m["constellation"]).points if "constellation" in m else None
    return sync.loops.LMS(m.get("taps", 11), m.get("mu", 0.01), points=pts)

@register_sync_method("equalize", "cma")
def _cma(m, rep, ctx):
    from ... import sync
    return sync.loops.CMA(m.get("taps", 11), m.get("mu", 0.01))

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
    from ... import sync
    if rep is not None and rep != Rep.SAMPLES:
        raise ConfigError(f"sync burst: cuts samples, not {rep}")
    if any(k in spec for k in ("carriers", "plan", "gate", "origin")): # Many carriers, a burst time plan (rpps.sync.plan)
        return _bursts(spec)
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

_BURST_KEYS = ("factor", "win", "min_s", "max_s", "percentile", "history", "settle", "quiet")

def _bursts(spec):
    """{"sync": "burst", "carriers": [{"freq", "width"}, ...] | "plan": "<plan>", "origin": ticks, "gate": "energy" |
    "plan", "tol": s, "lock": bursts, "margin": s}: bursts on many carriers, placed in a plan (rpps.sync.Bursts);
    transmitting, a plan places them (rpps.sync.PlanTx: "origin", "cf", "Fs", "ramp_s")"""
    from ... import sync, plan as _plan
    p = spec.get("plan")
    if isinstance(p, str):
        p = _plan.load(p) if p else None # ("": no plan, detection only)
    carriers = spec.get("carriers") or None # ([]: the plan's)
    try:
        op = sync.Bursts(carriers, p, spec.get("origin"), spec.get("gate", "energy"), spec.get("tol"), spec.get("lock", 8),
                         spec.get("margin"), spec.get("oversample", 3.0), sps=spec.get("sps"),
                         **{k: spec[k] for k in _BURST_KEYS if k in spec})
        if p is not None:
            tx = sync.PlanTx(p, spec.get("origin") or 0.0, spec.get("cf", 0.0), spec.get("Fs"), spec.get("ramp_s", 0.0))
        else:
            tx = sync.BurstTx(spec.get("Fs"), spec.get("gap_s", 0.0), spec.get("ramp_s", 0.0),
                              [c["freq"] for c in carriers] if carriers else None)
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
    from ... import sync
    m = spec
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
    from ... import sync
    return sync.Align(_need(m, "period", "align"), markers=m.get("markers"), pattern=m.get("pattern"), unit=m.get("unit", 8),
                      polarity=m.get("polarity", True), name=m.get("name", "align"), start=m.get("start", "any"),
                      publish=m.get("publish", False))

@_sync_one("puncture")
def _puncture(m, ctx):
    from ... import sync, code
    c = ctx["objects"].get("constellation") # The constellation step's: its turns tried with the offsets (an M-th power
    rot = None                               # carrier leaves the M-fold phase)
    if c is not None and m.get("rotations", True):
        from ...symbol.mapping import bit_rotations
        try:
            rot = bit_rotations(c.points, c.mapping.arr, c.bits_per_symbol)
        except Exception:
            rot = None
    return sync.Puncture(code.load(f'conv/{_need(m, "conv", "puncture")}'), m.get("window", 4000), rotations=rot) # {"sync": "puncture", "conv": "k7_171_133/3_4"}

@_sync_one("pilots")
def _pilot_eq(m, ctx):
    from ... import sync
    carriers = ctx["objects"].get("carriers")
    if carriers is None:
        raise ConfigError("pilots: needs a symbol carriers step after it")
    return sync.PilotEqualizer(carriers, m.get("taps"))

@register_sync_method("agc", "agc")
def _agc(m, rep, ctx):
    """The level by block estimation: "target", "window" (seconds), "log" (rpps.sync.AGC)"""
    from ... import sync
    if "mu" in m:
        raise ConfigError('agc: `mu` is the loop\'s rate; the AGC is block estimation: give `window` (seconds), or name '
                          'the loop ("method": "loop")')
    return sync.AGC(m.get("target", 1.0), **_given(m, ("window", "log", "min_gain", "max_gain")))

@register_sync_method("agc", "loop")
def _agc_loop(m, rep, ctx):
    """The gain followed sample by sample, second class (rpps.sync.loops.AGC): "target", "mu", "log" """
    from ... import sync
    return sync.loops.AGC(m.get("target", 1.0), **_given(m, ("mu", "log", "min_gain", "max_gain")))

@register_sync_method("agc", "peak")
def _peak_agc(m, rep, ctx):
    """The gain from the peak, followed at once (bursts, on-off keying)"""
    from ... import sync
    return sync.PeakAGC(m.get("target", 1.0), m.get("decay", 0.999))

@_sync_one("squelch")
def _squelch(m, ctx):
    from ... import sync
    return sync.Squelch(**_given(m, ("threshold_db", "hysteresis_db", "alpha")))

@_sync_one("quality")
def _quality(m, ctx):
    """Es/N0, Eb/N0 and EVM of locked symbols, published to est[name] (sync.quality.Monitor); the symbols pass unchanged.
    constellation: the definition they sit on (default: the one Acquire detected)"""
    from ... import sync, symbol
    pts = m.get("constellation")
    pts = None if pts is None else symbol.load(pts).points
    return sync.quality.Monitor(pts, **_given(m, ("bits", "rate", "history", "minimum", "name")))

@_sync_one("ofdm")
def _ofdm(m, ctx):
    """OFDM symbol timing and fractional frequency from the cyclic prefix (sync.CPSync): before a wave ofdm step"""
    from ... import sync
    return sync.CPSync(m.get("n_fft", 64), m.get("n_cp", 16), **_given(m, ("window", "min_symbols", "snr", "bw", "gate", "name")))

_FRAME_KEYS = ("before", "length", "segments", "scale", "polarity", "cfo", "classify", "margin", "min_symbols", "count")

def frame_words(m, where="sync frame"):
    """The preambles of a frame step: [(name, kind, value, fields)], kind "symbols" (a complex sequence: a Zadoff-Chu
    definition, the "zc" set) or "bits" (a word the profile maps to symbols through its transmit side). name None: the
    one preamble of `sequence` / `word`"""
    from ... import sequence
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
        from ...sequence.word import word_bits
        w, bits, order = _need(spec, "word", where), _need(spec, "bits", where), spec.get("order", "msb")
        return (name, "bits", np.asarray(word_bits(w, bits, order), bool), fields)
    out = []
    if "sequence" in m or "word" in m:
        out.append(one(None, {k: m[k] for k in ("sequence", "word", "bits", "order") if k in m}))
    for name, spec in (m.get("sequences") or {}).items():
        out.append(one(str(name), spec))
    if m.get("zc"):
        from ...sequence import zadoff_chu
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
    from ... import sync, symbol
    words = frame_words(m)
    if bool(words) == ("header" in m):
        raise ConfigError("sync frame: a `header`, or a word / sequence (`sequence`, `word` and `bits`, `sequences`, `zc`), one of the two")
    kw = dict(threshold=m.get("threshold"), max_length=m.get("max_length"), name=m.get("name", "frame"),
              **{k: m[k] for k in _FRAME_KEYS if k in m})
    if words and "segments" not in m: # A word's length says how many parts it is correlated in (rpps.sync.FrameSync)
        kw["segments"] = "auto"
    if words:
        if any(kind == "bits" for _, kind, _, _ in words):
            return sync.FrameSync(None, None, pending=True, **kw) # (the profile makes the symbols)
        if words[0][0] is None:
            return sync.FrameSync(words[0][2], **kw)
        return sync.FrameSync({n: (v, f) for n, _, v, f in words}, **kw)
    h = symbol.load_header(m["header"])
    return sync.FrameSync(h.preamble, h.words() if m.get("words", True) else None, device=m.get("device", False), **kw)

@_sync_one("inversion")
def _inversion(m, ctx):
    """Spectral inversion found by known symbols that turn, and the symbols conjugated back (rpps.sync.Inversion): the
    preamble of a physical header ("header"), or a complex `sequence`; `window`, `threshold`, `evidence`, `hold`"""
    from ... import sync, symbol
    if ("header" in m) == ("sequence" in m):
        raise ConfigError("sync inversion: a `header` or a complex `sequence`, one of the two")
    if "header" in m:
        known = symbol.load_header(m["header"]).preamble
    else:
        (_, kind, known, _), = frame_words({"sequence": m["sequence"]}, "sync inversion")
        if kind != "symbols":
            raise ConfigError(f"sync inversion: {m['sequence']!r} is a word of bits; it needs complex symbols")
    return sync.Inversion(known, **{k: m[k] for k in ("window", "threshold", "evidence", "hold") if k in m},
                          name=m.get("name", m.get("id", "inversion")))

@_sync_one("delimit")
def _delimit(m, ctx):
    """Frames by the word before them among bits: a link header's preamble word, a word sequence, or one given inline;
    `length` (bits per frame after the word, else until the next word or a header's control), `keep` (give the word too,
    for a code marker step to strip), `polarity` (match the inverted word too and put the frame's sense right)"""
    from ... import sync
    if "header" in m:
        from ... import link
        pre = _config.read(_config.find("header", m["header"], "link")).get("preamble") or {}
        h = link.load(f"header/{m['header']}")
        word, bits, order = pre.get("word"), pre.get("bits"), h.order
        if word is None:
            raise ConfigError(f"sync delimit: {m.get('header')} has no preamble word")
    else:
        word, bits, order = _word(m, "sync delimit")
    return sync.Delimiter(word, bits, order, m.get("errors", 0), m.get("name", m.get("id", "delimiter")), m.get("length"),
                          m.get("keep", False), m.get("polarity", False))
