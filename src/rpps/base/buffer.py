"""Buffered data types"""
from typing import Iterable

import numpy as np

class Buffer:
    """Generic buffer"""
    def __init__(self, buffer_size, data=None, dtype=np.float64):
        self._pnt = 0
        self._size = buffer_size
        if not data is None:
            self._stor = np.empty(self._size, dtype=data.dtype)
            self.set(data)
        else:
            self._stor = np.empty(self._size, dtype=dtype)

    def __str__(self):
        return f"{type(self).__name__} {self._pnt}/{self._size}"

    def __len__(self):
        return self._pnt

    @property
    def full(self):
        if self._pnt == self._size:
            return True

    @property
    def dtype(self):
        return self._stor.dtype

    @dtype.setter
    def dtype(self, dtype):
        self._stor = self._stor.astype(dtype)

    def get(self, size):
        while self._pnt > 0:
            return self._readN(size)

    def set(self, data):
        clip = None
        while self._pnt < self._size:
            clip = self._insertN(data)
            if clip is None:
                break
            data = data[clip:]
        return clip

    def reset(self):
        self._pnt = 0

    def resize(self, new_size):
        int_buf = np.empty(new_size, dtype=self._stor.dtype)
        self._pnt = self._pnt if self._pnt < new_size else new_size
        int_buf[:self._pnt] = self._stor[:self._pnt]
        self._stor = int_buf
        self._size = new_size

    def _insertN(self, data):
        if isinstance(data, Iterable):
            clip = np.min([len(data), self._size-self._pnt]) # type: ignore
            self._stor[self._pnt:self._pnt+clip] = data[:clip] # type: ignore
            self._pnt += clip
            if not clip == len(data): # type: ignore
                return clip
        else:
            self._stor[self._pnt] = data
            self._pnt += 1

    def _readN(self, size):
        if self._pnt > size:
            data = self._stor[:size+1]
            self._stor = np.roll(self._stor, -size)
            self._pnt -= size
            return data

    def __enter__(self, *args, **kwargs):
        return self._stor.__enter__(*args, **kwargs) # type: ignore

    def __exit__(self, *args, **kwargs):
        return self._stor.__exit__(*args, **kwargs) # type: ignore

    def __iter__(self, *args, **kwargs):
        return self._stor.__iter__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self._stor.__getitem__(*args, **kwargs)

class Vector(Buffer):
    def __init__(self, data=None, dtype=np.float64):
        super().__init__(2, data, dtype)
    def set(self, data):
        self._insertN(data)
    def _insertN(self, data):
        if isinstance(data, Iterable):
            while self._pnt + len(data) > self._size: # type: ignore
                self.resize(self._size*2)
            self._stor[self._pnt:self._pnt+len(data)] = data[:] # type: ignore
            self._pnt += len(data) # type: ignore
        else:
            if self._pnt + 1 > self._size:
                self.resize(self._size*2)
            self._stor[self._pnt] = data
            self._pnt += 1
