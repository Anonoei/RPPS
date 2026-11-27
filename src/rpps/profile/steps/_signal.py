"""signal steps (receive only)"""
from ... import _config
from ._core import STAGE, register_step, _one, _need, _given


@register_step("signal", "filter")
def _signal_filter(kind, spec, rep, ctx):
    from ... import signal
    return [(STAGE, signal.filters.load(_need(spec, "name", "signal filter")))], [], rep # "fir/...", "iir/..."

@register_step("signal", "decimate")
@_one
def _decimate(spec, ctx):
    from ... import signal
    return signal.sample.down.Decimate(_need(spec, "ratio", "signal decimate"), spec.get("offset", 0))

@register_step("signal", "dc_block")
@_one
def _dc_block(spec, ctx):
    from ... import signal
    return signal.DCBlock(**_given(spec, ("r",)))

@register_step("signal", "deemphasis")
@_one
def _deemphasis(spec, ctx):
    from ... import signal
    return signal.Deemphasis(**_given(spec, ("tau",)))

@register_step("signal", "shift")
@_one
def _shift(spec, ctx):
    from ... import signal
    return signal.filters.ShiftFreq(_need(spec, "freq", "signal shift"))

@register_step("signal", "conjugate")
@_one
def _conjugate(spec, ctx):
    """Spectral inversion of the capture (I and Q swapped, a lower-sideband mixer)"""
    from ... import signal
    return signal.Conjugate()

@register_step("signal", "resample")
@_one
def _resample(spec, ctx):
    """Rational resampling by up/down (a capture at 2.5 MS/s to 4 samples a symbol at 500 kBd: up 4, down 5)"""
    from ... import signal
    return signal.sample.re.Polyphase(_need(spec, "up", "signal resample"), _need(spec, "down", "signal resample"), spec.get("taps"))

@register_step("signal", "channelize")
@_one
def _channelize(spec, ctx):
    """One channel of a wider capture to baseband at about `oversample` times its width: {"signal": "channelize", "freq": -250e3,
    "width": 200e3}. The stages are designed from the stream's sample rate when it starts"""
    from ... import signal
    return signal.Channelize(spec.get("freq", 0.0), _need(spec, "width", "signal channelize"), spec.get("oversample", 3.0))

def _designed_fir(kind, spec, rep, ctx):
    """A windowed FIR designed in Hz from the stream's sample rate: {"signal": "lowpass", "cutoff": 100e3, "taps": 101} or
    {"signal": "bandpass", "cutoff": [50e3, 150e3], "transition": 20e3, "atten_db": 60} (a Kaiser window meeting that)"""
    from ...signal.filters.serial import _windowed, Designed
    obj = _config.Keys({k: v for k, v in dict.items(spec) if k != "signal"}, read=getattr(spec, "read", None)) # (what the
    obj["pass"], obj["units"] = kind, "hz"                                                                         # design reads is the step's)
    _need(obj, "cutoff", f"signal {kind}")
    return [(STAGE, Designed(kind, _windowed(kind, obj)))], [], rep

for _k in ("lowpass", "highpass", "bandpass", "bandstop"):
    register_step("signal", _k)(_designed_fir)
