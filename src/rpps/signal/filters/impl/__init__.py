"""Building blocks behind rpps.signal.filters: `designer` (simple lowpass/highpass prototypes by name: sinc, delta,
exponential...), `gaussian` (the Gaussian and its FWHM), `window` (window functions and their properties)
"""
from . import designer
from . import gaussian
from . import window
