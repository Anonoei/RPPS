"""Shared maths: finite fields, GF(2) linear algebra, cyclic-code decoding, probability, interpolation, bit tricks

Numpy only, no knowledge of the rest of rpps: anything two modules both need and that isn't about signals goes here.
"""
from . import bits
from . import gf
from . import gf2
from . import cyclic
from . import interp
from . import prob
from .gf import GF
