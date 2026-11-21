"""Visualization helpers. rpps' figures use `STYLE` (a matplotlib style, default "dark_background"; None: whatever
matplotlib is set to); importing this module no longer changes the style of anyone else's figures"""
from .base import show, figure, subplot, subplots, ion, ioff, cla, pause, styled

from .plots import eye
from .plots import freq
from .plots import spectrogram
from .plots.survey import survey
from .plots import time
from .plots.phasor import phasor

from . import save
