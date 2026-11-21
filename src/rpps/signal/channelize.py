from types import SimpleNamespace

import numpy as np

from ..base import Stage
from .filters.filters import low_pass
from .filters.shift import ShiftFreq

def _factors(D, largest=16):
    """Split D into stage factors of at most `largest`, largest first"""
    primes, d, p = [], D, 2
    while d > 1:
        while d % p == 0:
            primes.append(p)
            d //= p
        p += 1
    stages = []
    for q in sorted(primes, reverse=True):
        for i, s in enumerate(stages):
            if s*q <= largest:
                stages[i] = s*q
                break
        else:
            stages.append(q)
    return sorted(stages, reverse=True)

def channelize(mid, width, Fs, oversample=3.0, max_taps=401, largest=16):
    """Filter stages that bring a signal to baseband at about `oversample` times its width

    Decimates in several stages, each with an anti-alias filter just sharp enough for its own rate, so
    a 100 kHz signal in a 100 MS/s capture needs a few short filters rather than one of thousands of taps.
    Returns (pipeline, Fs_out, total decimation): `block + pipeline`. State carries across consecutive blocks of a stream.
    """
    from .sample.down import Decimate  # Late: sample imports filters
    D = max(1, int(Fs/(oversample*width)))
    while D > 1 and max(_factors(D, largest)) > largest:
        D -= 1
    stages = [ShiftFreq(-mid)]
    fs = Fs
    edge = 0.5*width*1.05 # Passband edge
    for f in _factors(D, largest):
        out = fs/f
        stop = out - 0.5*width*1.05 # What would alias onto the passband
        trans = stop - edge
        taps = int(np.ceil(3.3*fs/trans)) | 1 if trans > 0 else max_taps
        taps = int(np.clip(taps, 15, max_taps)) | 1
        stages.append(low_pass(taps, 0.5*(edge + stop), fs))
        stages.append(Decimate(f))
        fs = out
    from ..base import Pipeline
    return Pipeline(*stages), fs, D

class Channelize(Stage):
    """`channelize` as one stage: one channel of the capture to baseband, designed from the stream's sample rate when the
    first block arrives (a profile does not know the capture's rate)

        narrow = wide + rp.signal.Channelize(freq=-250e3, width=200e3)      # Fs out about 3 x 200 kHz

    freq: the channel's centre relative to the capture's [Hz]; width: its bandwidth [Hz]"""
    def __init__(self, freq, width, oversample=3.0):
        self.freq, self.width, self.oversample = float(freq), float(width), float(oversample)

    def __str__(self):
        return f"Channelize({self.freq:g}, {self.width:g})"

    def new_state(self, meta):
        from ..base.meta import Stream
        pipe, Fs_out, D = channelize(self.freq, self.width, meta.Fs, self.oversample)
        return SimpleNamespace(pipe=pipe, D=D, stream=Stream(meta.Fs))

    def step(self, samples, st, meta):
        from ..base import Data
        inner = Data(np.asarray(samples), Fs=meta.Fs, sps=meta.sps, cf=meta.cf, stream=st.stream)
        inner.flush = meta.flush
        out = inner + st.pipe
        meta.op.Fs, meta.op.sps = out.Fs, out.sps
        meta.op.cf = meta.cf + self.freq
        meta.delay += out.delay
        return np.asarray(out.data)

def baseband(x, Fs, freq, width, oversample=3.0):
    """A channel of a whole capture at baseband, in one call: (samples, Fs_out). freq: the channel's centre relative to the
    capture's centre [Hz]; width: its bandwidth [Hz]. The stages of `channelize` run once over x, flushed, and their group
    delay is taken off, so output sample k is the channel at input time k/Fs_out"""
    from ..base import Data
    pipe, Fs_out, D = channelize(freq, width, Fs, oversample)
    out = Data(np.asarray(x), Fs=Fs, flush=True) + pipe
    lag = int(round(out.delay*Fs_out))
    return np.asarray(out.data)[lag:], Fs_out
