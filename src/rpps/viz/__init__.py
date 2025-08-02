"""Visualization helpers"""
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from .dialog import get_file
from .base import show, figure, subplot, subplots, ion, ioff, cla, pause

from .plots.phasor import phasor
from .plots.eye import eye
from .plots.time import time, timeI, timeQ, timeIQ, ot_complex
# from .plots.freq import psd, phase, magnitude, mag_phase

from .plots import freq
from . import rt

from . import save
