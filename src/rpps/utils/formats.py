"""File format helpers"""
import os
import numpy as np
from abc import ABC, abstractmethod

class Format(ABC):
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
        if not self._last_path == path:
            self._last_path = path
        if max_samp == -1:
            self.max_samp = os.path.getsize(path) // self.byte_count
        self.cur_samp = 0

    def read(self, count: int, path: str=None, offset: int=0, skip=1):
        """Read next block from file"""
        if path is None and self._last_path is None:
            raise Exception("No path provided!")

        if path is not None:
            if self._last_path is None or not self._last_path == path:
                self.init(path)
                if not offset == 0:
                    offset = 0

        samps = self._read(path=path, count=count, offset=offset)

        for cur_ittr in range(0, self.max_samp):
            print(f"Reading {path} using count {count}, offset {offset}")
            yield self._read(path=path, count=count, offset=offset)
            self.cur_samp += count * skip
            offset = offset + (count * skip)

    @staticmethod
    @abstractmethod
    def _read(path: str, count: int, offset: int):
        pass

class ci8(Format):
    """Complex Int8"""
    byte_count = 1

    @staticmethod
    def _read(path, count, offset):
        count = count*2
        samps = np.fromfile(path, offset=offset, count=count, dtype=np.int8)
        samps = samps.astype(np.float32)
        samps = samps.view(dtype=np.complex64) # View the array of float32s as complex64 (real, imag, real, imag)
        return samps

class ci16(Format):
    """Complex Int16"""
    byte_count = 2

    @staticmethod
    def _read(path, count, offset):
        count = count*2
        samps = np.fromfile(path, offset=offset, count=count, dtype=np.int16)
        samps = samps.astype(np.float32)
        samps = samps.view(dtype=np.complex64) # View the array of float32s as complex64 (real, imag, real, imag)
        return samps

class cf32(Format):
    """Complex Float32"""
    byte_count = 32

    @staticmethod
    def _read(path, count, offset):
        count = count*2
        samps = np.fromfile(path, offset=offset, count=count, dtype=np.float16)
        samps = samps.astype(np.float32)  # Expand float16s to float32s
        samps = samps.view(dtype=np.complex64) # View the array of float32s as complex64 (real, imag, real, imag)
        return samps

class cf64(Format):
    """Complex Float64"""
    byte_count = 64

    @staticmethod
    def _read(path, count, offset):
        return np.fromfile(path, offset=offset, count=count, dtype=np.float32).view(np.complex64)

Formats = {
    "ci8": ci8,
    "ci16": ci16,
    "cf32": cf32,
    "cf64": cf64,
}
