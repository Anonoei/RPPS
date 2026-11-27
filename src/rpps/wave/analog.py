"""Analog demodulation, carrier modulation to baseband: AM envelope and SSB as stream stages (FM is `FMDemod`), and `demod`,
one block at a time with its measurements (FM deviation, AM depth, SINAD)"""
from types import SimpleNamespace

import numpy as np

from ..base import Stage
from ..signal.filters.iir import RunningMean
from ..signal.filters.filter import FIR
from ..signal.filters.impl.designer import designer
from .fm import discriminate

class AMDemod(Stage):
    """Envelope detector: |x| with the carrier (DC) removed by a slow running mean (alpha: how fast the mean follows, the
    fraction of the gap closed per sample); `AMDemod(alpha=None)` keeps the envelope as it is (on-off keying, pulse position:
    the levels themselves carry the bits)"""
    __slots__ = ("alpha",)
    def __init__(self, alpha=1e-3):
        self.alpha = alpha

    def new_state(self, meta):
        return SimpleNamespace(mean=None if self.alpha is None else RunningMean(self.alpha), z=None)

    def step(self, samples, st, meta):
        env = np.abs(samples).astype(np.float32)
        if self.alpha is None:
            return env
        mean, st.z = st.mean.run(env, st.z) # (an IIR section: no loop over the samples)
        return env - mean


class SSBDemod(FIR):
    """Single sideband: keeps `bw` Hz of the upper (or lower) side of the carrier at 0 Hz and returns the audio (real)

        audio = baseband + rp.wave.SSBDemod(2800, "usb")
    """
    __slots__ = ("bw", "sideband", "n")
    def __init__(self, bw=2800.0, sideband="usb", taps=201):
        if sideband not in ("usb", "lsb"):
            raise ValueError("sideband must be usb or lsb")
        self.bw, self.sideband, self.n = bw, sideband, taps | 1

    def make_taps(self, meta):
        lp = designer(self.n, self.bw/2, fs=meta.Fs).astype(np.float64)
        k = np.arange(self.n) - (self.n - 1)/2
        sign = 1 if self.sideband == "usb" else -1
        return (lp*np.exp(2j*np.pi*sign*(self.bw/2)*k/meta.Fs)).astype(np.complex64)

    def step(self, samples, st, meta):
        return np.real(super().step(samples, st, meta)).astype(np.float32)


# ---- One block at a time: a display's demodulator

class Measurements:
    """What `demod` measured on its block

    kind: fm, pm, am, usb or lsb. power_db: power in the channel (bw) [dB]
    offset_hz: FM and PM's carrier offset from 0 Hz (FM: the mean instantaneous frequency, PM: the slope of the phase);
        None for the others
    deviation_hz / deviation_rms_hz: FM peak (99.9th percentile) and rms deviation from that mean
    phase_dev_rad / phase_dev_rms_rad: PM peak (99.9th percentile) and rms phase deviation [rad]
    depth: AM modulation depth, (max - min)/(max + min) of the envelope (0.1th and 99.9th percentiles)
    tone_hz / sinad_db: the audio's strongest tone, and the power of everything over everything but it, when one tone
        holds most of the audio (a test tone); None otherwise
    """
    __slots__ = ("kind", "power_db", "offset_hz", "deviation_hz", "deviation_rms_hz", "phase_dev_rad", "phase_dev_rms_rad",
                 "depth", "tone_hz", "sinad_db")
    def __init__(self, kind, **kw):
        self.kind = kind
        for k in self.__slots__[1:]:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        bits = [f"{k} {getattr(self, k):.4g}" for k in self.__slots__[1:] if getattr(self, k) is not None]
        return f"<Measurements {self.kind}: " + ", ".join(bits) + ">"

TONE_FRACTION = 0.5 # A tone holding this much of the audio's power is a test tone: SINAD is measured against it

def _band(x, Fs, lo, hi):
    """x with only the frequencies in [lo, hi] Hz kept (a mask on its spectrum: no delay, no state); x itself when that is
    all of them"""
    if lo <= -Fs/2 and hi >= Fs/2:
        return x
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), 1/Fs)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.ifft(X)

def _band_real(a, Fs, hi):
    """Real a with only the frequencies up to hi Hz kept (as _band, on the half spectrum); a itself when that is all"""
    if hi >= Fs/2:
        return a
    A = np.fft.rfft(a)
    A[np.fft.rfftfreq(len(a), 1/Fs) > hi] = 0
    return np.fft.irfft(A, len(a))

def _tone(audio, Fs):
    """(tone Hz, SINAD dB) when one tone holds TONE_FRACTION of the audio's power, else (None, None)"""
    a = np.asarray(audio, dtype=np.float64)
    if len(a) < 64:
        return None, None
    a = a - a.mean()
    p = np.abs(np.fft.rfft(a*np.hanning(len(a))))**2
    p[0] = 0.0
    k = int(np.argmax(p))
    tone = p[max(k - 3, 0):k + 4].sum() # The Hann window's main lobe and a little of its skirt
    total = p.sum()
    if total <= 0 or tone < TONE_FRACTION*total:
        return None, None
    return k*Fs/len(a), float(10*np.log10(total/max(total - tone, 1e-30*total)))

def demod(x, Fs, kind="fm", bw=None, audio_bw=None):
    """Demodulate one block of analog signal, centred at 0 Hz, and measure it: for a display that wants each frame on its own
    (the stream stages `FMDemod`, `AMDemod` and `SSBDemod` carry state and filter delay from block to block instead)

    kind: "fm", "pm", "am", "usb" or "lsb". bw: the channel [Hz] (default: FM 200 kHz, PM 25 kHz, AM 10 kHz, SSB 2.8 kHz;
    capped at Fs)
    audio_bw: the audio kept [Hz] (default: FM 15 kHz, PM and AM bw/2, SSB bw)
    Returns (audio, audio_Fs, Measurements). audio is float32 in the signal's own units: FM the frequency deviation [Hz],
    PM the phase deviation [rad] (unwrapped, the carrier's offset and phase taken out), AM the envelope over the carrier
    less one (1.0 is 100% modulation), SSB the real signal. audio_Fs = Fs/D, the whole
    decimation D that keeps audio_bw with some room.
    A channel as wide as Fs, and audio as wide as Fs/2, are not filtered (the whole span shown); complex64 samples are
    worked in single precision (PM's unwrapped phase in double: a long ramp would lose its resolution).
    """
    if kind not in ("fm", "pm", "am", "usb", "lsb"):
        raise ValueError(f"kind must be fm, pm, am, usb or lsb, not {kind!r}")
    x = np.asarray(x)
    if x.dtype != np.complex64:
        x = x.astype(np.complex128)
    if len(x) < 2:
        raise ValueError(f"{len(x)} samples: nothing to demodulate")
    bw = min(float(bw or {"fm": 200e3, "pm": 25e3, "am": 10e3}.get(kind, 2800.0)), Fs)
    audio_bw = min(float(audio_bw or {"fm": 15e3, "pm": bw/2, "am": bw/2}.get(kind, bw)), Fs/2)
    m = Measurements(kind)
    if kind in ("usb", "lsb"):
        y = _band(x, Fs, 0.0, bw) if kind == "usb" else _band(x, Fs, -bw, 0.0)
        m.power_db = float(10*np.log10(np.mean(np.abs(y)**2) + 1e-30))
        audio = y.real
    else:
        y = _band(x, Fs, -bw/2, bw/2)
        m.power_db = float(10*np.log10(np.mean(np.abs(y)**2) + 1e-30))
        if kind == "fm":
            f = discriminate(y, Fs)
            m.offset_hz = float(np.mean(f))
            audio = f - m.offset_hz
            m.deviation_hz = float(np.percentile(np.abs(audio), 99.9))
            m.deviation_rms_hz = float(np.sqrt(np.mean(audio**2)))
        elif kind == "pm":
            ph = np.unwrap(np.angle(y).astype(np.float64))
            n = np.arange(len(ph))
            slope, icpt = np.polyfit(n, ph, 1) # The carrier: its offset (a phase ramp) and its phase
            m.offset_hz = float(slope*Fs/(2*np.pi))
            audio = ph - (slope*n + icpt)
            m.phase_dev_rad = float(np.percentile(np.abs(audio), 99.9))
            m.phase_dev_rms_rad = float(np.sqrt(np.mean(audio**2)))
        else:
            env = np.abs(y)
            lo, hi = np.percentile(env, (0.1, 99.9))
            m.depth = float((hi - lo)/(hi + lo)) if hi + lo > 0 else 0.0
            carrier = float(np.mean(env))
            audio = env/carrier - 1 if carrier > 0 else env
        audio = _band_real(audio, Fs, audio_bw)
    D = max(1, int(Fs // (2.5*audio_bw)))
    audio = audio[::D].astype(np.float32)
    m.tone_hz, m.sinad_db = _tone(audio, Fs/D)
    return audio, Fs/D, m
