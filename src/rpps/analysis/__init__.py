"""Analysis: blind estimation and measurement at any representation, outside the data path

 - fec: which error-correcting code (convolutional, block/LDPC, RS, scrambler) a hard-bit stream carries
 - signal_profile: `profile(source)`, every step that processes each signal in a capture, and what it might be where unsolved
 - rs, scrambler, block, puncture: blind Reed-Solomon codes, self-synchronising scramblers, binary block codes (BCH / cyclic,
   product, LDPC: length, alignment and checks by rank) and puncture patterns, for what no definition covers
 - modulation: symbol rate, timing, modulation, frequency offset and phase from baseband samples
 - quality: Es/N0, Eb/N0 and EVM of locked symbols
 - psd, stft: spectra; rate: symbol rate; signal: finding the signals in a capture (survey)
Results feed the configured stages (`Guess.save` writes a config/fec definition, `Detection` fills meta.stream.det,
`fec.CodeDetection` (from `fec.identify`, which rp.sync.AcquireFEC runs) names the profile that decodes a stream).
"""
from . import fec
from . import rs, scrambler, block, puncture # Blind tools for what no definition covers (see fec.solve)
from . import signal_profile
from .signal_profile import profile, SignalProfile
from . import modulation
from . import quality
from .modulation import detect, Detection

from . import stft
from . import rate
from . import signal
from .rate import find_rate, find_baud
from .signal import Signal, Survey, Channel, find_signals, survey, welch, noise_floor
from .psd import psd, vbw, vbw_calc
