"""File format helpers"""
import os
import numpy as np
from abc import ABC, abstractmethod

from enum import Enum

def from_i8(samples):
    samps = np.array(samples, dtype=np.int8)
    return samps.astype(np.float32).view(dtype=np.complex64)

def from_i16(samples):
    samps = np.array(samples, dtype=np.int16)
    return samps.astype(np.float32).view(dtype=np.complex64)

def from_f16(samples):
    samps = np.array(samples, dtype=np.float16)
    return samps.astype(np.float32).view(dtype=np.complex64)

def from_f32(samples):
    samps = np.array(samples, dtype=np.float32)
    return samps.view(dtype=np.complex64)

class Formats:
    i8  = (2, from_i8)
    i16 = (4, from_i16)
    f16 = (4, from_f16)
    f32 = (8, from_f32)

    def bytes(self):
        return self.value[0]

    def read(self, path, count, offset):
        return self.value[1](path, count, offset)

class Format:
    """File Format parent class"""
    byte_count = 0

    def __init__(self):
        self.cur_samp = 0
        self.max_samp = -1
        self._last_path = None

    def __str__(self):
        return str(type(self).__name__)

    def init(self, path, max_samp=-1):
        """Initialize the format"""
        print(f"Initializing with path: {path}")
        if not self._last_path == path:
            self._last_path = path
        if max_samp == -1:
            self.max_samp = os.path.getsize(path) // self.byte_count
        self.cur_samp = 0

    def read(self, count: int, path: str=None, offset: int=-1, skip=1):
        """Read next block from file"""
        if path is None and self._last_path is None:
            raise Exception("No path provided!")

        if path is not None:
            if self._last_path is None or not self._last_path == path:
                self.init(path)

        if offset == -1:
            offset = self.cur_samp

        # samps = self._read(path=path, count=count, offset=offset)

        for cur_ittr in range(0, self.max_samp):
            print(f"Reading {path} using count {count}, offset {offset}")
            yield self._read(path=path, count=count, offset=offset)
            self.cur_samp += count * skip
            offset = offset + (count * skip)

    @staticmethod
    @abstractmethod
    def _read(path: str, count: int, offset: int):
        pass
