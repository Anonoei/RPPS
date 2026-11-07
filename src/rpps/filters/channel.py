import numpy as np

from .filters import low_pass
from .shift import ShiftFreq

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

def channel(mid, width, Fs, oversample=3.0, max_taps=401, largest=16):
    """Filter stages that bring a signal to baseband at about `oversample` times its width

    Decimates in several stages, each with an anti-alias filter just sharp enough for its own rate, so
    a 100 kHz signal in a 100 MS/s capture needs a few short filters rather than one of thousands of taps.
    Returns (pipeline, Fs_out, total decimation): `block + pipeline`. State carries across consecutive blocks of a stream.
    """
    from ..sample.down import Decimate # Late: sample imports filters
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
