"""IQ/Constellation implementation"""
import math
import numpy as np

from pyboiler.logger import Logger, Level

from . import base
from . import dobject
from . import Meta


class Mapping:
    """Constellation map"""
    __slots__ = ("arr", "comment")

    def __init__(self, map=None, comment=""):
        self.comment = comment
        if isinstance(map, int):
            map = np.array([0] * map)
        elif not isinstance(map, np.ndarray):
            map = np.array(map)
        elif map is None:
            map = np.array([])
        self.arr = map

    @staticmethod
    def new(map):
        """Create a new mapping"""
        return Mapping(map)

    @staticmethod
    def empty(length: int):
        """Create a new empty mapping"""
        return Mapping(length)

    def str(self):
        """Return map values as str"""
        return "-".join(self.arr.astype(str))

    def __len__(self):
        return len(self.arr)

    def __str__(self):
        return str(self.arr)

    def __repr__(self):
        return f"({str(self)}, {self.comment})"

    def __getitem__(self, item):
        return self.arr[item]

    def __setitem__(self, item, val):
        self.arr[item] = val


class Maps:
    """Collection of constellation mappings"""
    __slots__ = ("maps")

    def __init__(self, maps):
        self.maps = maps

    def __str__(self):
        return str(self.maps)

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, item):
        return self.maps[item]

    def __setitem__(self, item, val):
        self.maps[item] = val

class Points:
    """Complex points"""
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
        """Get real from points"""
        return np.real(self.arr)

    def imag(self):
        """Get imag from points"""
        return np.imag(self.arr)

    def degrees(self):
        """Get point degrees"""
        return np.angle(self.arr, deg=True)


class Constellation:
    """Constellation implementation"""
    __slots__ = ("log", "_points", "_mapping", "_bps")

    def __init__(self, points: Points, mapping = None, log=Logger().Child("Modulation")):
        self.log = log.Child("Constellation", Level.WARN)
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
        """Get constellation points"""
        return self._points

    @points.setter
    def points(self, points):
        self._points = np.array(points)
        self._bps = len(self.points) // 2

    @property
    def mapping(self):
        """Get constellation mapping"""
        return self._mapping

    @mapping.setter
    def mapping(self, map: Mapping):
        self._mapping = map

    @property
    def bits_per_symbol(self):
        """Get bits per symbol"""
        return self._bps

    def modulate(self, dobj: dobject.BitObject, noise: bool = True):
        """Modulate BitObject to IQ symbols"""
        indexes = self.index(dobj)
        points = self.map(indexes, dobj.meta)
        symbols = self.to_symbols(points, dobj.meta, noise=noise)
        return symbols

    def demodulate(self, syms: dobject.SymData):
        """Demodulate IQ symbols to ModData"""
        # Distances[i] are values 0-1 of how far away sym[i] was from each constellation point
        distances = np.zeros((len(syms.data), len(self.points)), dtype=np.float16)

        distances[:] = np.abs(self.points.arr - syms.data.reshape(-1, 1))
        distances[:] = 1 - np.round(distances / distances.max(axis=0), decimals=2)

        bits = np.array([bin(n)[2:].zfill(self._bps) for n in self.mapping.arr])
        codewords = np.zeros((len(self.points), self._bps), dtype=int)

        for i, b in enumerate(bits):
            for j, c in enumerate(b):
                codewords[i, j] = True if c == '1' else False

        mod = dobject.ModData()
        mod.soft = base.SoftDecision(codewords, distances)

        return mod

    ##############################
    #  Modulate
    ##############################

    def index(self, dobj: dobject.BitObject):
        """Convert bits to indexes"""
        self.log.trace(f"Data is {dobj.data}")
        self.log.trace(f"Bits per symbol: {self._bps} / {len(dobj)}")

        padding = len(dobj) % self._bps
        if not padding == 0:

            for _ in range(0, (self._bps - padding)):
                dobj.append(0)
            self.log.trace(f"Padded by {self._bps - padding}: {len(dobj)}")
        num_symbols = len(dobj) // self._bps

        self.log.trace(f"Data requires {num_symbols} indexes to encode")
        self.log.trace(f"Data is: {dobj}")

        indexes = np.split(dobj.data, num_symbols)
        indexes = [int("".join(data.astype(int).astype(str)), 2) for data in indexes]
        # indexes = []
        # for i in range(0, len(data.bin), self._bps):
        #    bit_int = int(data.bin[i:i+self._bps], 2)
        #    indexes.append(int(bit_int))

        self.log.trace(f"Indexes are {indexes}")
        return indexes

    def map(self, indexes, meta):
        """Convert indexes to self.mapping values"""
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
        """Convert mapping values to symbols"""
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
        return dobject.SymData(symbols)

    ##############################
    #  Demodulate
    ##############################

    def from_symbols(self, symbols: dobject.SymObject):
        """Convert symbols to soft decisions"""
        # self.log.trace(f"Symbols are:\n{symbols}")
        # codewords = np.zeros((len(self.points), self._bps), dtype=bool)
        distances = np.zeros((len(symbols), len(self.points)), dtype=np.float16)

        distances[:] = np.abs(self.points.arr - symbols.data.reshape(-1, 1))
        distances[:] = 1 - np.round(distances / distances.max(axis=0), decimals=2)
        self.log.trace(f"Points are {distances} / Demodulated {len(distances)} symbols")
        return distances

    def unmap(self, points, meta):
        """Convert points to map indexes"""
        self.log.trace(f"Using mapping: {self.mapping}")
        indexes = []
        for pnt in points:
            indexes.append(int(self.mapping[pnt]))
        self.log.trace(f"Indexes are {indexes}")
        return indexes

    def unindex(self, indexes, meta):
        """Convert indexes to bits"""
        self.log.trace(f"Bits per symbol: {self._bps}")
        bits = ""
        for ind in indexes:
            bits += bin(ind)[2:].zfill(self._bps)

        padding = len(bits) % 8
        if not padding == 0:
            self.log.trace(f"Unpadding by {padding}")
            bits = bits[:-padding]

        data = dobject.ModData(np.array([int(bit) for bit in bits], dtype=bool))

        # data = Stream.from_bin(bits, len(bits))

        self.log.trace(f"Data bits are {data} / {len(data)}")

        return data
