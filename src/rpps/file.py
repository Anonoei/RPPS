"""Wrapper for processing files"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from .helpers import Formats, Format


class file:
    fmt = Format()

    @staticmethod
    def get_format():
        return file.fmt

    @staticmethod
    def read(fmt, path: str, bins: int, offset: int = 0, skip=1, max_ittr=-1):
        file.fmt = Formats[fmt]()
        for sym in file.fmt.read(fmt, path, bins, offset, skip, max_ittr):
            yield sym
