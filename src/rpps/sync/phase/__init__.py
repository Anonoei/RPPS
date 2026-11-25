"""Carrier frequency and phase, by block estimation (rpps.sync): `MPower` (M-th power: FFT acquisition, then per-window
phase, tracked), `DDCarrier` / `decided.follow` (from decisions, any constellation: APSK, QAM), `FFCarrier` (Viterbi &
Viterbi, per frame)
"""
from .ff import FFCarrier
from .mpower import MPower, offset_of
from . import decided
from .decided import DDCarrier
