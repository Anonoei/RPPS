"""Visualization helpers"""
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from .base import show, figure, subplot, subplots, ion, ioff, cla, pause

from .plots import eye
from .plots import freq
from .plots import spectrogram
from .plots.survey import survey
from .plots import time
from .plots.phasor import phasor

from . import save
