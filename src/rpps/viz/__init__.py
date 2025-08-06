"""Visualization helpers"""
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from .dialog import get_file
from .base import show, figure, subplot, subplots, ion, ioff, cla, pause
from . import blit

from .plots import eye
from .plots import freq
from .plots import spectrogram
from .plots import time
from .plots.phasor import phasor

from . import ot
from . import rt

from . import save
