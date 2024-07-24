import math
import numpy as np

from helpers.stream import Stream
from meta import Meta

from pyboiler.logger import Logger, Level


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


class Constellation:
    __slots__ = ("log", "_points", "_mapping", "_bps")

    def __init__(self, points: Points, mapping = None, log=Logger().Child("Modulation")):
        self.log = log.Child("Constellation", Level.WARN)
        if not isinstance(points, Points):
            points = Points(points)
        self._points = points

        if mapping is None:
            mapping = Mapping()
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
    def mapping(self, map):
        self._mapping = np.array(map)

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
        self.log.trace(f"Data is {data}")
        self.log.trace(f"Data bitarray is {data.bitarray}")

        padding = len(data.bin) % self._bps
        if not padding == 0:
            self.log.trace(f"Padding by {padding}")
            for _ in range(0, (self._bps - padding)):
                data.bitarray.append(0)
        num_symbols = len(data.bitarray) // self._bps

        self.log.trace(f"Data requires {num_symbols} indexes to encode")

        indexes = np.split(data.bitarray.arr, num_symbols)
        indexes = [int(f"{data[0]}{data[1]}", 2) for data in indexes]
        #for i in range(0, len(data.bin), self._bps):
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
    #  Demodulate
    ##############################

    def from_symbols(self, symbols, meta, clean: bool=True):
        #self.log.trace(f"Symbols are:\n{symbols}")
        points = []
        if clean:
            syms = []
            for symbol in symbols: # Clean up the symbols
                diff = np.abs(self.points - symbol)
                index = np.argmin(diff)
                closest = self.points[index]
                syms.append(closest)
            syms = np.array(syms)
            self.log.trace(f"Symbols are:\n{syms}")

            for symbol in syms:
                points.append(int(np.where(self.points == symbol)[0][0]))
        else:
            self.log.trace(f"Symbols are:\n{symbols}")
            for symbol in symbols: # Clean up the symbols
                diff = np.abs(self.points - symbol)
                index = np.argmin(diff)
                points.append(int(index))
        self.log.trace(f"Points are {points}")
        return points

    def unmap(self, points, meta):
        self.log.trace(f"Using mapping: {self.mapping}")
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

        data = Stream.from_bin(bits)

        self.log.trace(f"Data bits are {data.bin}")

        self.log.trace(f"Data hex is {data.hex}")
        self.log.trace(f"Data is {data.bytes}")

        return data
