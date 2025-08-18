"""IQ/Constellation implementation"""
import math
import numpy as np

from pyboiler.logger import Logger, Level

from . import _config

class Mapping:
    """Constellation map"""
    __slots__ = ("arr", "_comment", "_inv")

    def __init__(self, map=None, comment=""):
        self._comment = comment
        if isinstance(map, int):
            map = np.array([0] * map)
        elif not isinstance(map, np.ndarray):
            map = np.array(map)
        elif map is None:
            map = np.array([])
        self.arr = map
        self._inv = False

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

    @property
    def inverted(self):
        return self._inv

    @inverted.setter
    def inverted(self, val: bool):
        self._inv = val

    @property
    def comment(self):
        if self._inv:
            return f"{self._comment} Inverted"
        return f"{self._comment} Normal"


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
        return f"{self.arr}"

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
    __slots__ = ("log", "_points", "_mapping", "_bps", "_bit_mul")

    def __init__(self, points: Points, mapping: Mapping = Mapping(), log=Logger().Child("Modulation", _config.LOG_MOD)):
        self.log = log.Child("Constellation", _config.LOG_MOD_CONST)
        if not isinstance(points, Points):
            points = Points(points)
        self._points = points

        self._mapping = mapping

        self._bps = int(math.log2(len(self.points))) # Bits per symbol
        self._bit_mul = np.zeros(self._bps, dtype=np.int8)

        for i in range(0, self._bps):
            if _config.BYTE_ORDER == "<":
                self._bit_mul[self._bps-(i+1)] = 2**i
            else:
                self._bit_mul[i] = 2**i

    def __str__(self) -> str:
        return f"Points: {self._points}, Map: {self._mapping}, {self._mapping.comment}"

    def __repr__(self) -> str:
        return f"<Constellation: {self._bps}>"

    def __len__(self) -> int:
        return len(self._points)

    @property
    def points(self):
        """Get constellation points"""
        return self._points

    @property
    def inverted(self):
        """ Returns if constellation is spectral inverted"""
        return self._mapping.inverted
    # { "real": 0.7, "imag": -0.7 },
    # { "real": -0.7, "imag": -0.7 },
    # { "real": 0.7, "imag": 0.7 },
    # { "real": -0.7, "imag": 0.7 }
    def invert(self):
        """Spectral invert the constellation"""
        self._mapping.inverted = not self._mapping.inverted
        rotpoints = self._points.imag() + self._points.real() * 1j
        swaps = np.where(self.points.arr == rotpoints)[0]
        map1 = swaps[0:len(swaps)//2]
        map2 = swaps[len(swaps)//2:]
        for m1, m2 in zip(map1, map2):
            self._mapping.arr[[m1,m2]] = self._mapping.arr[[m2,m1]]

    @points.setter
    def points(self, points):
        self._points = Points(np.array(points))
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

    def modulate(self, data):
        """Modulate BitObject to IQ symbols"""
        self.log.debug(f"Modulating {len(data)}, {data.dtype}")
        indexes = self.index(data)
        points = self.map(indexes)
        symbols = self.to_symbols(points)
        return symbols

    def demodulate(self, syms):
        """Demodulate IQ symbols to ModData"""
        self.log.debug(f"Demodulating {len(syms)}, {syms.dtype}")
        distances = self.distances(syms)
        codewords = self.codewords()
        return codewords, distances

    ##############################
    #  Modulate
    ##############################
    def index(self, data):
        """Convert bits to base10 representation"""
        self.log.trace(f"index: Bits per symbol: {self._bps} / {len(data)}")

        padded = 0
        while not len(data) % self._bps == 0:
            data = np.append(data, 0)
            padded += 1
        if not padded == 0:
            self.log.debug(f"Padded by {padded}: {len(data)}")
        # num_symbols = len(data) // self._bps

        indexes = data.reshape((-1, self._bps))
        indexes = np.multiply(indexes, self._bit_mul)
        indexes = np.sum(indexes, axis=1)
        return indexes

    def map(self, indexes):
        """Convert indexes to self.mapping values"""
        self.log.trace(f"map: using map {self.mapping}")
        points = np.empty_like(indexes)
        for i in range(len(self.mapping)):
            points[indexes==i] = np.where(self.mapping.arr==i)[0][0]
        self.log.trace(f"map: mapped {len(points)} points")
        return points

    def to_symbols(self, points):
        """Convert mapping values to symbols"""
        self.log.trace(f"to_sym: using {len(points)} mapped points")
        symbols = self._points.arr[points.astype(int)].astype(np.complex64)
        self.log.trace(f"to_sym: modulated {len(symbols)} symbols")
        return symbols

    ##############################
    #  Demodulate
    ##############################
    def from_symbols(self, symbols):
        """Convert symbols to soft decisions"""
        self.log.trace(f"from_sym: using {len(symbols)} symbols")
        # self.log.trace(f"Symbols are:\n{symbols}")
        # codewords = np.zeros((len(self.points), self._bps), dtype=bool)
        # Distances[i] are values 0-1 of how far away sym[i] was from each constellation point
        distances = np.zeros((len(symbols), len(self.points)), dtype=np.float16)

        distances[:] = np.abs(self.points.arr - symbols.reshape(-1, 1))
        distances[:] = 1 - np.round(distances / distances.max(axis=0), decimals=2)
        self.log.trace(f"from_sym: demodulated {len(distances)} symbols")
        return distances

    def unmap(self, points):
        """Convert points to map indexes"""
        self.log.trace(f"Using mapping: {self.mapping}")
        indexes = []
        for pnt in points:
            indexes.append(int(self.mapping[pnt]))
        self.log.trace(f"Indexes are {indexes}")
        return indexes

    def unindex(self, indexes):
        """Convert indexes to bits"""
        self.log.trace(f"Bits per symbol: {self._bps}")
        bits = ""
        for ind in indexes:
            bits += bin(ind)[2:].zfill(self._bps)

        padding = len(bits) % 8
        if not padding == 0:
            self.log.trace(f"Unpadding by {padding}")
            bits = bits[:-padding]

        data = np.array([int(bit) for bit in bits], dtype=bool)

        self.log.trace(f"Data bits are {data} / {len(data)}")

        return data

    def distances(self, symbols):
        distances = np.zeros((len(symbols), len(self.points)), dtype=np.float16)

        distances[:] = np.abs(self.points.arr - symbols.reshape(-1, 1))
        distances[:] = 1 - np.round(distances / distances.max(axis=0), decimals=2)
        return distances

    def codewords(self):
        # bits = np.array([bin(n)[2:].zfill(self._bps) for n in self.mapping.arr])
        # codewords = np.zeros((len(self.points), self._bps), dtype=int)

        # for i, b in enumerate(bits):
        #     for j, c in enumerate(b):
        #         codewords[i, j] = True if c == '1' else False
        codewords = np.zeros((len(self.mapping.arr), self._bps), dtype=np.uint8)
        for i, m in enumerate(self.mapping.arr):
            if m == 0:
                continue
            residue = 0
            for j in range(self._bps):
                div = np.divide(m-residue, self._bit_mul).astype(int)
                idx = (div==1)
                if idx.any() == True:
                    residue += self._bit_mul[idx]
                    codewords[i][idx] = 1


        # codewords = np.binary_repr(self.mapping.arr)
        return codewords
