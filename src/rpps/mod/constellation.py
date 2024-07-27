import math
import numpy as np

from pyboiler.logger import Logger, Level

from . import Stream
from . import bitarray
from . import Meta


class Mapping:
    __slots__ = ("arr")

    def __init__(self, map=None):
        if isinstance(map, int):
            map = np.array([0] * map)
        elif not isinstance(map, np.ndarray):
            map = np.array(map)
        elif map is None:
            map = np.array([])
        self.arr = map

    @staticmethod
    def new(map):
        return Mapping(map)

    @staticmethod
    def empty(length: int):
        return Mapping(length)

    def str(self):
        return "-".join(self.arr.astype(str))

    def __len__(self):
        return len(self.arr)

    def __str__(self):
        return str(self.arr)

    def __getitem__(self, item):
        return self.arr[item]

    def __setitem__(self, item, val):
        self.arr[item] = val


class Maps:
    __slots__ = ("maps")

    def __init__(self, maps):
        self.maps = maps

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, item):
        return self.maps[item]

    def __setitem__(self, item, val):
        self.maps[item] = val

class Points:
    __slots__ = ("arr")

    def __init__(self, points):
        self.arr = np.array(points)

    def __len__(self):
        return len(self.arr)

    def __str__(self):
        return str(self.arr)

    def __getitem__(self, item):
        return self.arr[item]

    def __setitem__(self, item, val):
        self.arr[item] = val

    def real(self):
        return np.real(self.arr)

    def imag(self):
        return np.imag(self.arr)


class Constellation:
    __slots__ = ("log", "_points", "_mapping", "_bps")

    def __init__(self, points: Points, mapping = None, log=Logger().Child("Modulation")):
        self.log = log.Child("Constellation", Level.TRACE)
        if not isinstance(points, Points):
            points = Points(points)
        self._points = points

        self._mapping = mapping

        self._bps = int(math.log2(len(self.points))) # Bits per symbol

    def __str__(self) -> str:
        return f"Points: {self._points}, Map: {self._mapping}"

    def __repr__(self) -> str:
        return f"<Constellation: {self._bps}>"

    def __len__(self) -> int:
        return len(self._points)

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
    def mapping(self, map: Mapping):
        self._mapping = map

    @property
    def bits_per_symbol(self):
        return self._bps

    def modulate(self, data: Stream, meta: Meta, noise: bool = True):
        indexes = self.index(data, meta)
        points = self.map(indexes, meta)
        symbols = self.to_symbols(points, meta, noise=noise)
        return symbols

    def demodulate(self, symbols, meta: Meta):
        points = self.from_symbols(symbols, meta)
        indexes = self.unmap(points, meta)
        data = self.unindex(indexes, meta)
        return data

    ##############################
    #  Modulate
    ##############################

    def index(self, data: Stream, meta):
        self.log.trace(f"Data bitarray is {data.bitarray}")
        self.log.trace(f"Bits per symbol: {self._bps} / {len(data.bitarray)}")

        padding = len(data.bitarray) % self._bps
        if not padding == 0:

            for _ in range(0, (self._bps - padding)):
                data.bitarray.append(0)
            self.log.trace(f"Padded by {self._bps - padding}: {len(data.bitarray)}")
        num_symbols = len(data.bitarray) // self._bps

        self.log.trace(f"Data requires {num_symbols} indexes to encode")

        indexes = np.split(data.bitarray.arr, num_symbols)
        indexes = [int("".join(data.astype(str)), 2) for data in indexes]
        # indexes = []
        # for i in range(0, len(data.bin), self._bps):
        #    bit_int = int(data.bin[i:i+self._bps], 2)
        #    indexes.append(int(bit_int))

        self.log.trace(f"Indexes are {indexes}")
        return indexes

    def map(self, indexes, meta):
        self.log.trace(f"Using mapping: {self.mapping}")
        points = []
        for idx in indexes:
            points.append(
                int(
                    np.where(self.mapping.arr == idx)[0][0]
                )
            )
        self.log.trace(f"Points are {points}")
        return points

    def to_symbols(self, points, meta, noise: bool = False):
        points = np.array(points)
        symbols = self.points[points]
        self.log.trace(f"Symbols are:\n{symbols} / {len(symbols)}")
        # Add noise
        if noise:
            n = (np.random.randn(len(symbols)) + 1j*np.random.randn(len(symbols)))/np.sqrt(2) # AWGN with unity power
            symbols = symbols + n * np.sqrt(0.01) # noise power of 0.01

        symbols = symbols.astype(np.complex64)
        meta.fmt = type(symbols[0]).__name__
        # self.log.trace(f"Symbols are: {symbols}")
        return symbols

    ##############################
    #  Demodulate
    ##############################

    def from_symbols(self, symbols, meta, clean: bool=True):
        # self.log.trace(f"Symbols are:\n{symbols}")
        points = []
        if clean:
            syms = []
            for symbol in symbols: # Clean up the symbols
                diff = np.abs(self.points - symbol)
                index = np.argmin(diff)
                closest = self.points[index]
                syms.append(closest)
            syms = np.array(syms)
            self.log.trace(f"Symbols are:\n{syms}  / {len(symbols)}")

            for symbol in syms:
                points.append(int(np.where(self.points == symbol)[0][0]))
        else:
            self.log.trace(f"Symbols are:\n{symbols}  / {len(symbols)}")
            for symbol in symbols: # Clean up the symbols
                diff = np.abs(self.points - symbol)
                index = np.argmin(diff)
                points.append(int(index))
        self.log.trace(f"Points are {points} / Demodulated {len(points)} symbols")
        return points

    def unmap(self, points, meta):
        self.log.trace(f"Using mapping: {self.mapping}")
        indexes = []
        for pnt in points:
            indexes.append(int(self.mapping[pnt]))
        self.log.trace(f"Indexes are {indexes}")
        return indexes

    def unindex(self, indexes, meta):
        self.log.trace(f"Bits per symbol: {self._bps}")
        bits = ""
        for ind in indexes:
            bits += bin(ind)[2:].zfill(self._bps)

        padding = len(bits) % 8
        if not padding == 0:
            self.log.trace(f"Unpadding by {padding}")
            bits = bits[:-padding]

        data = bitarray([int(bit) for bit in bits])

        #data = Stream.from_bin(bits, len(bits))

        self.log.trace(f"Data bits are {data} / {len(data)}")

        return data
