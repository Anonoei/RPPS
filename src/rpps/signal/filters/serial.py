"""Filter definitions, config/filter/<category>/<group>/<name>.json (`rp.signal.filters.load("fir", "taps/name")`), and pulses,
config/pulse/<type>/<name>.json (`rp.signal.filters.load("pulse", "rrc/0p35")`): two resources

All frequencies are fractions of the sample rate (0.5 is Nyquist), or Hz with "units": "hz". FIR and shaping filters build their
taps when a stream starts, from its `Fs` and `sps` (`stage.taps(Fs=..., sps=...)` shows them). "gain" scales any of them;
"fs" (optional) is the rate the definition was made for, and warns if the stream differs.

    fir      "type": "taps"      "taps": [...], "taps_imag": [...] (complex), "normalize": "dc" (unity gain at DC)
             "type": "windowed"  "pass": lowpass | highpass | bandpass | bandstop, "cutoff": f or [low, high], then
                                 "taps": n, "window": name (default np.hamming, as rp.signal.filters.low_pass; "hamming" is the built-in one)
                                 or "transition": width, "atten_db": dB (Kaiser window sized to meet it)
             "type": "remez"    "taps": n, "bands": [...], "desired": [...], "weight": [...]   "response": bandpass (default) | hilbert | differentiator; no scipy
    iir      numpy only, as second-order sections. No constant group delay, so a stream's `delay` is unchanged. Unstable: rejected.
             "type": "sos"       "sos": [[b0, b1, b2, a0, a1, a2], ...]
             "type": "ba"        "b": [...], "a": [...]       scipy sign: y[n] = sum(b x) - sum(a[1:] y)
             "type": "zpk"       "z": [...], "p": [...], "k": gain       complex values as [re, im]; fewer zeros than poles = delay
             "type": "butter" | "cheby1" | "cheby2"
                                 "order", "pass": as fir, "cutoff": f or [low, high], "ripple_db" (cheby1), "atten_db" (cheby2)
             "type": "biquad"    "shape": lowpass | highpass | bandpass | notch | allpass | peaking | lowshelf | highshelf,
                                 "f0", "q", "gain_db"                                                  (RBJ audio-EQ cookbook)
             "type": "single_pole"  "alpha": y[n] = alpha x[n] + (1 - alpha) y[n-1]
    pulse    "type": "rrc" | "rc"    "beta": 0.35, "span": 25 (symbols) or "taps": n, "sps": override the stream's
             "type": "sinc" | "rect" "taps", "sps"           "type": "biphase"   Manchester matched filter
    window   "type": a built-in window name (`Window.s()`), "params": {its arguments}: kaiser/a8 is kaiser(N, a=8).
             Plain windows (hann, blackman...) are code, not files; `Window.get` and a design's "window" accept either.

Coefficients that are in a file or another tool go in through `rpps.signal.filters.convert`, never as a path in a definition.
"""
import warnings
from types import SimpleNamespace

import numpy as np

from ... import _config
from ...base import Meta
from ...errors import ConfigError
from .filter import FIR, Filter
from .filters import kaiser_params, kaiser_window
from .impl.designer import designer
from .impl.remez import remez
from .impl import window as _windows
from . import shaping as _shaping
from . import iir as _iir

FIR_TYPES = ("taps", "windowed", "remez")
SHAPING_TYPES = ("rrc", "rc", "sinc", "rect", "biphase", "halfsine", "gaussian")

class Designed(FIR):
    """A FIR defined by a config entry: the taps are made when the stream starts, from its sample rate and symbol rate"""
    def __init__(self, name, build, fs=None):
        self.name, self.build, self.fs = name, build, fs

    def __str__(self):
        return f"Filter:{self.name}"

    def make_taps(self, meta):
        if self.fs and abs(meta.Fs - self.fs) > 1e-9*self.fs:
            warnings.warn(f"{self.name} was designed for Fs={self.fs} but the stream has Fs={meta.Fs}")
        return self.build(meta)

    def taps(self, Fs=None, sps=None):
        """The coefficients for a sample rate and samples per symbol (needed by designs in Hz and by shaping pulses)"""
        return np.asarray(self.build(Meta(Fs=Fs if Fs else float("nan"), sps=sps or 0.0)))

    def run(self, samples, Fs=None, sps=None):
        """One-shot (stateless) filtering, drops the edges"""
        return np.convolve(samples, self.taps(Fs, sps), mode="valid")

# ---- Windows

def builtin(name):
    """A window function of the code, by name (see `Window.s()`)"""
    if name not in _windows.s:
        raise ConfigError(f"No window {name!r}: not a preset ({sorted(_config.ls('filters').get('window', []))}) and not built in ({_windows.s})")
    return _windows.get(name)

def _window_def(name, obj):
    fn, params = builtin(obj["type"]), dict(obj.get("params", {}))
    if not params:
        return fn
    def preset(N, *args, **kw):
        return fn(N, *args, **{**params, **kw})
    preset.__name__ = name
    return preset

def window(spec="hamming"):
    """A window function f(N): a preset ("kaiser/a8"), a built-in ("hann"), inline {"type": "kaiser", "params": {"a": 8}}, or a function"""
    if callable(spec):
        return spec
    if isinstance(spec, dict):
        return _window_def("inline", spec)
    file = _config.find("filter", spec, "window")
    if file is not None:
        return _config.load_obj("filter", spec, _config.read(file))
    return builtin(spec)

# ---- FIR

def _freq(obj, key, Fs):
    """A frequency (or list of them) of a design, as a fraction of the sample rate"""
    v = np.asarray(obj[key], dtype=float)
    if obj.get("units", "fs") == "hz":
        if not (np.isfinite(Fs) and Fs > 0):
            raise ConfigError(f"{key} is in Hz: the sample rate is needed (taps(Fs=...), or run it on a stream)")
        v = v / Fs
    return v

def _post(h, obj):
    h = np.asarray(h)
    h = h if h.dtype.kind == "c" else h.astype(float)
    if obj.get("normalize") == "dc":
        h = h / h.sum()
    return h * obj.get("gain", 1.0)

def _taps(name, obj):
    h = np.asarray(obj["taps"], dtype=float)
    if h.ndim != 1 or h.size == 0:
        raise ConfigError(f"{name}: taps must be a list of numbers")
    if "taps_imag" in obj:
        im = np.asarray(obj["taps_imag"], dtype=float)
        if im.shape != h.shape:
            raise ConfigError(f"{name}: taps_imag has {im.size} values for {h.size} taps")
        h = h + 1j*im
    h = _post(h, obj)
    return lambda meta: h

PASSES = {"lowpass": True, "highpass": False, "bandpass": False, "bandstop": True} # -> pass_zero

def _windowed(name, obj):
    kind = obj.get("pass")
    if kind not in PASSES:
        raise ConfigError(f"{name}: pass must be one of {sorted(PASSES)}, not {kind!r}")
    if "taps" not in obj and not ("transition" in obj and "atten_db" in obj):
        raise ConfigError(f"{name}: needs `taps` (with an optional window), or `transition` and `atten_db` (Kaiser window)")
    win = (window(obj["window"]) if "window" in obj else np.hamming) if "taps" in obj else None # np.hamming, as rp.signal.filters.low_pass
    def build(meta):
        cut = _freq(obj, "cutoff", meta.Fs)
        if win is not None:
            n, w = int(obj["taps"]), win
        else:
            n, beta = kaiser_params(obj["atten_db"], float(_freq(obj, "transition", meta.Fs)))
            w = kaiser_window(beta)
        if n % 2 == 0 and kind in ("highpass", "bandstop"):
            raise ConfigError(f"{name}: a {kind} needs an odd number of taps, not {n}")
        return _post(designer(n, cut, window=w, pass_zero=PASSES[kind], fs=1.0), obj)
    return build

def _remez(name, obj):
    for k in ("taps", "bands", "desired"):
        if k not in obj:
            raise ConfigError(f"{name}: a remez design needs `{k}`")
    def build(meta):
        h = remez(int(obj["taps"]), _freq(obj, "bands", meta.Fs), obj["desired"], weight=obj.get("weight"), fs=1.0, type=obj.get("response", "bandpass"))
        return _post(h, obj)
    return build

def _fir(name, obj):
    if obj.get("units", "fs") not in ("fs", "hz"):
        raise ConfigError(f"{name}: units must be fs (fractions of the sample rate) or hz")
    t = obj.get("type", "taps")
    makers = {"taps": _taps, "windowed": _windowed, "remez": _remez}
    if t not in makers:
        raise ConfigError(f"{name}: unknown fir type {t!r}, have {list(FIR_TYPES)}")
    return Designed(name, makers[t](name, obj), obj.get("fs"))

# ---- IIR

IIR_TYPES = ("sos", "ba", "zpk", "butter", "cheby1", "cheby2", "biquad", "single_pole")

def _complex(values, name, key):
    """A list of complex numbers, each a number or a [re, im] pair"""
    out = []
    for v in values:
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ConfigError(f"{name}: each {key} is a number or a [re, im] pair")
            out.append(complex(v[0], v[1]))
        else:
            out.append(complex(v))
    return np.array(out, dtype=complex)

def _iir_sections(name, obj):
    """build(meta) -> the sections of an IIR definition"""
    t = obj.get("type")
    if t not in IIR_TYPES:
        raise ConfigError(f"{name}: unknown iir type {t!r}, have {list(IIR_TYPES)}")
    def need(*keys):
        for k in keys:
            if k not in obj:
                raise ConfigError(f"{name}: a {t} definition needs `{k}`")
    if t == "sos":
        need("sos")
        return lambda meta: obj["sos"]
    if t == "ba":
        need("b", "a")
        return lambda meta: _iir.tf2sos(obj["b"], obj["a"])
    if t == "zpk":
        need("z", "p", "k")
        z, p = _complex(obj["z"], name, "z"), _complex(obj["p"], name, "p")
        return lambda meta: _iir.zpk2sos(z, p, obj["k"])
    if t == "single_pole":
        need("alpha")
        return lambda meta: _iir.single_pole(obj["alpha"])
    if t == "biquad":
        need("shape", "f0")
        return lambda meta: _iir.biquad(obj["shape"], float(_freq(obj, "f0", meta.Fs)), obj.get("q", 0.7071067811865476), obj.get("gain_db", 0.0))
    need("order", "pass", "cutoff")
    return lambda meta: _iir.design(t, obj["order"], obj["pass"], _freq(obj, "cutoff", meta.Fs), obj.get("ripple_db"), obj.get("atten_db"))

class DesignedIIR(Filter):
    """An IIR defined by a config entry: the sections are made when the stream starts (designs in Hz need its sample rate)"""
    def __init__(self, name, build, gain=1.0, fs=None):
        self.name, self.build, self.gain, self.fs = name, build, gain, fs
        self._made = (None, None)

    def __str__(self):
        return f"IIR:{self.name}"

    def _stage(self, Fs):
        """The IIR for a sample rate (the last one is kept)"""
        if self._made[0] is None or self._made[0] != Fs:
            try:
                stage = _iir.IIR(self.build(Meta(Fs=Fs)), self.gain)
            except ValueError as exc:
                raise ConfigError(f"{self.name}: {exc}") from exc
            self._made = (Fs, stage)
        return self._made[1]

    def new_state(self, meta):
        if self.fs and abs(meta.Fs - self.fs) > 1e-9*self.fs:
            warnings.warn(f"{self.name} was designed for Fs={self.fs} but the stream has Fs={meta.Fs}")
        stage = self._stage(meta.Fs)
        return SimpleNamespace(stage=stage, inner=stage.new_state(meta))

    def step(self, samples, st, meta):
        return st.stage.step(samples, st.inner, meta)

    def sos(self, Fs=None):
        """The sections for a sample rate (needed only by designs in Hz)"""
        return self._stage(Fs if Fs else float("nan")).sos.copy()

    def response(self, Fs=None, n=512):
        """(f, H) from 0 to Nyquist, f in cycles per sample"""
        return self._stage(Fs if Fs else float("nan")).response(n)

    def run(self, samples, Fs=None):
        """One-shot (stateless) filtering from zero state"""
        return self._stage(Fs if Fs else float("nan")).run(samples)

def _iir_def(name, obj):
    if obj.get("units", "fs") not in ("fs", "hz"):
        raise ConfigError(f"{name}: units must be fs (fractions of the sample rate) or hz")
    build = _iir_sections(name, obj)
    stage = DesignedIIR(name, build, obj.get("gain", 1.0), obj.get("fs"))
    if obj.get("units", "fs") == "fs":
        stage._stage(1.0) # Fractions of the sample rate need no stream: check the design now
    return stage

# ---- Shaping

def _shaping_def(name, obj):
    t = obj.get("type")
    if t not in SHAPING_TYPES:
        raise ConfigError(f"{name}: unknown shaping type {t!r}, have {list(SHAPING_TYPES)}")
    gain, beta = obj.get("gain", 1.0), obj.get("beta", 0.35)
    if t == "biphase":
        return _shaping.ShapingBiphase(gain)
    def build(meta):
        sps = obj.get("sps") or meta.sps
        if not sps:
            raise ConfigError(f"{name}: needs the samples per symbol (`sps` in the definition, or a stream that has it)")
        if t == "halfsine":
            return _shaping.ShapingHalfSine(sps, gain).pulse
        if t == "gaussian":
            return _shaping.ShapingGaussian(sps, obj.get("bt", 0.5), obj.get("span", 4), gain).pulse
        n = obj.get("taps") or (int(round(sps)) if t == "rect" else _shaping._default_taps(sps, obj.get("span", 25)))
        cls = {"rrc": lambda: _shaping.ShapingRRC(n, sps, beta, gain), "rc": lambda: _shaping.ShapingRC(n, sps, beta, gain),
               "sinc": lambda: _shaping.ShapingSinc(n, sps, gain), "rect": lambda: _shaping.ShapingRect(n, sps, gain)}[t]
        return cls().pulse
    return Designed(name, build)

DOMAIN = _config.register_domain("filter", {"fir": _fir, "iir": _iir_def, "window": _window_def})
PULSES = _config.register_domain("pulse", {"pulse": _shaping_def})

def ls(path=None):
    """List the filter and pulse definitions: {"fir/windowed": [names], "pulse/rrc": [...], "window/kaiser": [...]}"""
    return {**_config.ls("filter", path), **{f"pulse/{k}": v for k, v in _config.ls("pulse", path).items()}}

def load(kind: str, ref: str, path=None):
    """Load a filter, config/filter/<kind>/<family>/<name>.json, or a pulse, config/pulse/<family>/<name>.json:
    `load("fir", "taps/name")`, `load("pulse", "rrc/0p35")`. kind: fir | iir | window | pulse"""
    if kind == "pulse":
        return _config.load("pulse", ref, None, path)
    return _config.load("filter", ref, kind, path)
