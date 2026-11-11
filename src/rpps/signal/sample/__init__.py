"""Sample-rate changes as stages: `down` (decimation), `up` (interpolation) and `re` (rational resampling); `resample_poly` resamples an array in one call
"""
from ... import _config

from . import down
from . import up
from . import re
from .resample import resample_poly
