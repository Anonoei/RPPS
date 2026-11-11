"""Analysis: blind estimation and measurement at any representation, outside the data path

 - fec: which error-correcting code (convolutional, block/LDPC, RS, scrambler) a hard-bit stream carries
 - modulation: symbol rate, timing, modulation, frequency offset and phase from baseband samples
 - quality: Es/N0, Eb/N0 and EVM of locked symbols
 - psd, stft: spectra; rate: symbol rate; signal: finding the signals in a capture (survey)
Results feed the configured stages (`Guess.save` writes a config/fec definition, `Detection` fills meta.stream.det).
"""
from . import fec
from . import modulation
from . import quality
from .modulation import detect, Detection

from . import stft
from . import rate
from . import signal
from .rate import find_rate, find_baud
from .signal import Signal, Survey, Channel, find_signals, survey, welch, noise_floor
from .psd import psd, vbw, vbw_calc
