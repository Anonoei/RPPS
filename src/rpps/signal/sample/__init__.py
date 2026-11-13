"""Sample-rate changes as stages: `down` (decimation), `up` (interpolation) and `re` (rational resampling), `fractional` (any ratio, with a shift fused in); `resample_poly` and `fractional_resample` resample an array in one call
"""
from ... import _config

from . import down
from . import up
from . import re
from .resample import resample_poly
from .fractional import FractionalPolyphase, fractional_resample
