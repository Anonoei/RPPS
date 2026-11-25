"""wave, symbol, code and link steps: transmit / receive pairs"""
from ... import _config
from ...base.types import Rep
from ...errors import ConfigError
from ._core import STAGE, DECODE, ENCODE, register_step, _need, _given, _complex, _word

def _pulse(name):
    """The pulse resource and whether it is a symbol pulse or a frequency pulse"""
    from ... import signal
    f = _config.find("pulse", name)
    if f is None:
        raise ConfigError(f"wave shape: no pulse {name!r} in config/pulse")
    kind = _config.read(f).get("kind", "symbol")
    return (lambda: signal.filters.load(f"pulse/{name}")), kind

def _wave(kind, spec, rep, ctx):
    from ... import wave, signal
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
    from ... import symbol
    if "shape" in spec:
        c = symbol.generate(spec["shape"])
        return c if spec.get("mapping") is None else c.set_mapping(spec["mapping"])
    return symbol.load(_need(spec, "name", "symbol constellation"), spec.get("mapping"))

def _symbol(kind, spec, rep, ctx):
    from ... import symbol, sequence
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
        from ... import sequence
        ref = _need(spec, "sequence", "symbol scramble")
        make = lambda: symbol.PhaseScramble(sequence.load(ref, **spec.get("args", {})), spec.get("offsets", [0]), spec.get("step", 90),
                                            spec.get("length"), spec.get("id", "scramble"))
        return _pair(make(), make(), Rep.SYMBOLS)

def _code(kind, spec, rep, ctx):
    from ... import code
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
    from ... import link
    if kind == "gse": # {"link": "gse", "protocols": [130]}: GSE packets of BBFRAME data fields (rpps.link.gse)
        from ...link.gse import GSE
        make = lambda: GSE(spec.get("protocols"), spec.get("id", "gse"))
        return _pair(make(), make(), Rep.PACKETS)
    if kind == "packets": # Packets that cross frames: rpps.link.packets
        from ...link.packets import Packets
        make = lambda: Packets({k: v for k, v in spec.items() if k != "link"}, spec.get("id", "packets"))
        return _pair(make(), make(), Rep.PACKETS)
    if kind in ("crc", "frame", "header"):
        name = _need(spec, "name", f"link {kind}")
        def make():
            op = link.load(f"{kind}/{name}")
            if kind == "header" and spec.get("select"): # {"select": {"isi": 1}}: only the frames with those field values
                op.select = dict(spec["select"])
            if kind == "header" and spec.get("length") is not None: # A framed header's frame length, from the profile
                if not op.framed:
                    raise ConfigError(f"link header {name}: `length` is a framed header's frame length")
                op.length = int(spec["length"])
            return op
        return _pair(make(), make(), Rep.PACKETS)

for _cat, _fn, _kinds in (("wave", _wave, ("shape", "stagger", "fm", "am", "ssb", "ofdm")),
                          ("symbol", _symbol, ("constellation", "differential", "spread", "carriers", "header", "pilots", "scramble", "marker", "puncture")),
                          ("code", _code, ("block", "conv", "interleave", "scramble", "uncoded", "line", "marker")),
                          ("link", _link, ("crc", "frame", "header", "gse", "packets"))):
    for _k in _kinds:
        register_step(_cat, _k)(_fn)
