"""Wrapper for processing files"""

from . import formats

import os
from enum import Enum
import numpy as np

from . import err

def read_i8(path, count, offset):
    samps = np.fromfile(path, offset=offset, count=count, dtype=np.int8)
    return formats.from_i8(samps)
def read_i16(path, count, offset):
    samps = np.fromfile(path, offset=offset, count=count, dtype=np.int16)
    return formats.from_i16(samps)
def read_f16(path, count, offset):
    samps = np.fromfile(path, offset=offset, count=count, dtype=np.float16)
    return formats.from_f16(samps)
def read_f32(path, count, offset):
    samps = np.fromfile(path, offset=offset, count=count, dtype=np.float32)
    return formats.from_f32(samps)

class Formats(Enum):
    i8  = (2, read_i8)
    i16 = (4, read_i16)
    f16 = (4, read_f16)
    f32 = (8, read_f32)

    @property
    def size(self):
        return self.value[0]

    def read(self, path, count, offset):
        return self.value[1](path, count, offset*self.size)

class File:
    def __init__(self, fmt, path):
        self.fmt = Formats[fmt]
        self.path = path
        self.cur_samp = 0
        self.max_samp = os.path.getsize(path) // self.fmt.size

    def __str__(self):
        return f"File ({self.cur_samp}/{self.max_samp},{self.fmt.name}): {self.path}"

    def read(self, count):
        count *= 2
        if self.cur_samp + count > self.max_samp:
            raise err.Overflow(f"{self.cur_samp}+{count} > {self.max_samp}")
        samps = self.fmt.read(self.path, count, self.cur_samp)
        self.cur_samp += count
        return samps

    def __call__(self, count):
        return self.iter(count)

    def iter(self, count):
        while self.cur_samp + count <= self.max_samp:
            yield self.read(count)

    @property
    def percent(self):
        return self.cur_samp/self.max_samp
