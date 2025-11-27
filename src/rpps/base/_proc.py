from typing import Iterable

import numpy as np

class Processor:
    """Fill to buffer_size, then process"""
    def __init__(self, buffer_size):
        self.done = False
        self._pnt = 0
        self._size = buffer_size
        self._stor = np.empty(buffer_size)

    def __str__(self):
        return f"Processor ({self.done}) {self._pnt}/{self._size}"

    def encode(self, data):
        for _ in self.iter(data):
            if self.done:
                print("encode")
                self.reset()
            else:
                break

    def decode(self, data):
        for _ in self.iter(data):
            if self.done:
                print("decode")
                self.reset()
            else:
                break

    def iter(self, data):
        while True:
            clip = self._insertN(data)
            yield clip
            if type(clip) == bool:
                break
            data = data[clip:]
    def reset(self, new_size=None):
        self._pnt = 0
        self.done = False
        if not new_size is None:
            self._size = new_size
            self._stor = np.empty(new_size)

    def __call__(self, data):
        return self.iter(data)

    def _insertN(self, data):
        if not self.done:
            if isinstance(data, Iterable):
                clip = np.min([len(data), self._size-self._pnt]) # type: ignore
                self._stor[self._pnt:self._pnt+clip] = data[:clip] # type: ignore
                self._pnt += clip
                if self._pnt == self._size:
                    self.done = True
                if not clip == len(data): # type: ignore
                    return clip
            else:
                self._stor[self._pnt] = data
                self._pnt += 1
                if self._pnt == self._size:
                    self.done = True
            return True
        return False

    def __mul__(self, other):
        other.encode(self)
    def __truediv__(self, other):
        other.encode(self)
    def __rmul__(self, other):
        self.encode(other)
    def __rtruediv__(self, other):
        self.decode(other)

class Process:
    def encode(self):
        pass
    def decode(self):
        pass

class Pipeline:
    def __init__(self):
        self.e_stack = []
        self.d_stack = []

    def add_encode(self, process: Process):
        self.e_stack.append(process)

    def add_decode(self, process: Process):
        self.d_stack.append(process)

    def encode(self, data):
        if not self.e_stack:
            return data
        e_data = data
        for proc in self.e_stack:
            e_data = proc.encode(e_data)
        return e_data

    def decode(self, data):
        if not self.d_stack:
            return data
        d_data = data
        for proc in self.d_stack:
            d_data = proc.decode(d_data)
        return d_data
