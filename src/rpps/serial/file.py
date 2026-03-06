"""Wrapper for processing files"""

import os
import numpy as np

from .formats import Formats
from . import err

class File:
    def __init__(self, fmt: str, path):
        self.fmt: Formats = Formats[fmt]
        self.path = path
        self.cur_samp = 0
        self.max_samp = os.path.getsize(path) // self.fmt.bytes
        self.f = None

    def __str__(self):
        return f"File ({self.cur_samp}/{self.max_samp},{self.fmt.name}): {self.path}"

    def open(self):
        if self.f is None:
            self.f = open(self.path, "rb")
            self.f.seek(self.cur_samp // self.fmt.bytes)

    def close(self):
        if self.f is not None:
            self.f.close()

    def read(self, count):
        if self.cur_samp + count > self.max_samp:
            raise err.Overflow(f"{self.cur_samp}+{count} > {self.max_samp}")
        samps = self.fmt.read(self.f, count)
        self.cur_samp += count
        return samps

    def __call__(self, count):
        return self.iter(count)

    def iter(self, count):
        self.open()
        while self.cur_samp + count <= self.max_samp:
            yield self.read(count)
        self.close()

    @property
    def percent(self):
        return self.cur_samp/self.max_samp
