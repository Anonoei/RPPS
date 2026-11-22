"""Carrier frequency and phase: first class `MPower` (M-th power: FFT acquisition, then per-window phase, tracked);
`FFCarrier` (Viterbi & Viterbi, per frame) (rpps.sync)
"""
from .ff import FFCarrier
from .mpower import MPower, offset_of
