"""Conditioning: samples to samples, without knowing what signal they carry

    filters     FIR/IIR design, windows, frequency shifts, the polyphase channelizer, pulse shaping
    sample      sample-rate changes: decimation, interpolation, rational resampling
    channelize  bring one signal of a wider capture to baseband at a few times its width
    channel     the simulated medium between transmitter and receiver: AWGN, offsets, multipath, fading
"""
from . import filters
from . import sample
from . import channel
from .channelize import channelize
from .condition import DCBlock, Deemphasis
