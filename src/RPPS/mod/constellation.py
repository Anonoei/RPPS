import numpy as np

import helpers.binary as binh
from helpers.encoding import Encoding
from meta import Meta

from pyboiler.logger import Logger, Level

class Constellation:
    __slots__ = ("log", "_points", "_mapping", "_bps")

    def __init__(self, points, map = None, log=Logger().Child("Modulation")):
        self.log = log.Child("Constellation")
        self._points = np.array(points)
        if map is None:
            self._mapping = np.array(list(range(0, len(points), 1)))
        else:
            self._mapping = np.array(map)
        self._bps = len(self.points) // 2 # Bits per symbol

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = np.array(points)
        self._bps = len(self.points) // 2

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, map):
        self._mapping = np.array(map)

    @property
    def bits_per_symbol(self):
        return self._bps


    def encode(self, data, meta: Meta, noise: bool = True):
        indexes = self.index(data, meta)
        points = self.map(indexes, meta)
        symbols = self.to_symbols(points, meta, noise=noise)
        return symbols

    def decode(self, symbols, meta: Meta):
        points = self.from_symbols(symbols, meta)
        indexes = self.unmap(points, meta)
        data = self.unindex(indexes, meta)
        return data


    ##############################
    #  Encoding
    ##############################

    def index(self, data, meta):
        data = Encoding.from_bytes(data)
        self.log.trace(f"Data is {data}")
        self.log.trace(f"Data hex is {data.hex}")
        self.log.trace(f"Data bits are {data.bin}")

        padding = len(data.bin) % self._bps
        if not padding == 0:
            self.log.trace(f"Padding by {padding}")
            data.bin = data.bin + ("0" * (self._bps - padding))
        num_symbols = len(data.bin) // self._bps

        self.log.trace(f"Data requires {num_symbols} indexes to encode")

        indexes = []
        for i in range(0, len(data.bin), self._bps):
            bit_int = int(data.bin[i:i+self._bps], 2)
            indexes.append(int(bit_int))

        self.log.trace(f"Indexes are {indexes}")
        return indexes

    def map(self, indexes, meta):
        points = []
        for idx in indexes:
            points.append(int(np.where(self.mapping == idx)[0][0]))
        self.log.trace(f"Points are {points}")
        return points

    def to_symbols(self, points, meta, noise: bool = True):
        points = np.array(points)
        symbols = self.points[points]
        self.log.trace(f"Symbols are:\n{symbols}")
        # Add noise
        if noise:
            n = (np.random.randn(len(symbols)) + 1j*np.random.randn(len(symbols)))/np.sqrt(2) # AWGN with unity power
            symbols = symbols + n * np.sqrt(0.01) # noise power of 0.01

        symbols = symbols.astype(np.complex64)
        meta.fmt = type(symbols[0]).__name__
        self.log.trace(f"Symbols are: {symbols}")
        return symbols

    ##############################
    #  Decoding
    ##############################

    def from_symbols(self, symbols, meta):
        #self.log.trace(f"Symbols are:\n{symbols}")
        syms = []
        for symbol in symbols: # Clean up the symbols
            diff = np.abs(self.points - symbol)
            index = np.argmin(diff)
            closest = self.points[index]
            syms.append(closest)
        syms = np.array(syms)
        self.log.trace(f"Symbols are:\n{syms}")
        points = []
        for symbol in syms:
            points.append(int(np.where(self.points == symbol)[0][0]))
        self.log.trace(f"Points are {points}")
        return points

    def unmap(self, points, meta):
        indexes = []
        for pnt in points:
            indexes.append(int(self.mapping[pnt]))
        self.log.trace(f"Indexes are {indexes}")
        return indexes

    def unindex(self, indexes, meta):

        bits = ""
        for i in range(0, len(indexes), 1):
            cur_bits = bin(indexes[i])[2:]
            if len(cur_bits) == 1:
                cur_bits = "0" + cur_bits
            bits += cur_bits

        data = Encoding.from_bin(bits)

        self.log.trace(f"Data bits are {data.bin}")

        self.log.trace(f"Data hex is {data.hex}")
        self.log.trace(f"Data is {data.bytes}")

        return data
