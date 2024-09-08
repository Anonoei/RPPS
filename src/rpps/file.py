"""Wrapper for processing files"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from . import Meta

from .helpers import Formats, Format


class file:
    fmt = Format()

    @staticmethod
    def get_format():
        return file.fmt

    @staticmethod
    def read(meta: Meta, path: str, bins: int, offset: int = 0, skip=1, max_ittr=-1):
        file.fmt = Formats[meta.fmt]()
        for sym in file.fmt.read(meta, path, bins, offset, skip, max_ittr):
            yield sym
