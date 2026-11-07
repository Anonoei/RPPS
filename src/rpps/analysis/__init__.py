"""Blind analysis: work out what a signal or bit stream is, without being told

 - fec: which error-correcting code (convolutional, block/LDPC, RS, scrambler) a hard-bit stream carries
 - modulation: symbol rate, timing, modulation, frequency offset and phase from baseband samples
Results feed the configured stages (`Guess.save` writes a config/coding definition, `Detection` fills meta.stream.det).
"""
from . import fec
from . import modulation
from .modulation import detect, Detection
