"""File format helpers"""
import os
import numpy as np

from ..process import find_rate

class Format:
    """File Format parent class"""
    byte_count = 0
    _cache = {
        "read_time": 0.0,
        "block_time": 0.0,
        "sample_time": 0.0,
        "max_ittr": -1,
        "cur_ittr": 0
    }
    _last_path = None

    def __str__(self):
        return str(type(self).__name__)

    def init(self, path, meta, count, max_ittr):
        """Initialize the format"""
        if not self._last_path == path:
            self._last_path = path
            if max_ittr == -1:
                max_ittr = os.path.getsize(path) // count
            self._cache["sample_time"] = 1 / meta.freq.fields["SampleRate"]
            self._cache["block_time"] = count * self._cache["sample_time"]
            self._cache["read_time"] = self._cache["block_time"] * max_ittr
            self._cache["max_ittr"] = max_ittr
            self._cache["cur_ittr"] = 0
        return self._cache

    @property
    def cache(self):
        """Get the formats cache"""
        return self._cache

    @property
    def cur_time(self):
        """Get relative time for current iteration"""
        return self._cache["block_time"] * self._cache["cur_ittr"]

    @property
    def block_time(self):
        """Get file time per block"""
        return self._cache["block_time"]

    @property
    def read_time(self):
        """Get total file time"""
        return self._cache["read_time"]

    @property
    def block(self):
        """Get current block"""
        return self._cache["cur_ittr"]

    @property
    def blocks(self):
        """Get max block"""
        return self._cache["max_ittr"]

    def read(self, meta, path: str, count: int, offset: int = 0, skip=1, max_ittr=-1):
        """Read next block from file"""
        if not self._last_path == path:
            if max_ittr == -1:
                max_ittr = os.path.getsize(path) // count
            samps = self._read(path=path, count=count, offset=offset)
            if meta.freq.fields.get("SampleRate", None) is None:
                meta.freq["SampleRate"] = find_rate(samps)[0]
            self._cache["sample_time"] = 1 / meta.freq["SampleRate"]
            self._cache["block_time"] = count * self._cache["sample_time"]
            self._cache["read_time"] = self._cache["block_time"] * max_ittr
            self._cache["max_ittr"] = max_ittr
            self._cache["cur_ittr"] = 0
        for cur_ittr in range(0, max_ittr):
            # print(f"Reading {path} using count {count}, offset {offset}")
            yield self._read(path=path, count=count, offset=offset)
            self._cache["cur_ittr"] = cur_ittr
            offset = (offset + type(self).byte_count) * skip

    @staticmethod
    def _read(path: str, count: int, offset: int):
        ...

class cf32(Format):
    """Complex Float32"""
    byte_count = 32

    @staticmethod
    def _read(path: str, count, offset):
        count = count*2
        syms = np.fromfile(path, offset=offset, count=count, dtype=np.float16)
        syms = syms.astype(np.float32)  # Expand float16s to float32s
        syms = syms.view(dtype=np.complex64)  # View the array of float32s as complex64 (real, imag, real, imag)
        return syms

class cf64(Format):
    """Complex Float64"""
    byte_count = 64

    @staticmethod
    def _read(path, count, offset):
        return np.fromfile(path, offset=offset, count=count, dtype=np.complex64)

Formats = {
    "cf32": cf32,
    "cf64": cf64,
}
