"""IQ/Constellation implementation"""
import math
import numpy as np

from pyboiler.logger import Logger

from ... import _config
from ...math.prob import logsumexp, mixture_variance

_MIN_VAR = 1e-6 # Floor for noise variances: a noiseless input still gives finite LLRs

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

    def demodulate(self, syms, maxlog=False):
        """Demodulate IQ symbols to one LLR per bit, log(P(0)/P(1)): positive means 0. The noise level is estimated"""
        self.log.debug(f"Demodulating {len(syms)}, {syms.dtype}")
        return self.llr(syms, maxlog=maxlog)

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

    def labels(self):
        """Bits of every constellation point under the current mapping: (points, bits per symbol), first bit first"""
        arr = np.asarray(self.mapping.arr).astype(np.int64)
        return ((arr[:, None] // self._bit_mul.astype(np.int64)[None, :]) & 1).astype(np.uint8)

    def estimate_noise(self, syms, iters=8):
        """Noise variance E|n|^2 of received symbols (assumes the constellation has unit mean power and the gain is right)

        Expectation-maximisation over the Gaussian mixture of the constellation points
        """
        y = np.asarray(syms, dtype=np.complex128).reshape(-1, 1)
        d2 = np.abs(y - self.points.arr[None, :])**2
        return 2*mixture_variance(d2, iters, _MIN_VAR) # E|n|^2 is twice the variance per real dimension

    def llr(self, syms, noise_var=None, maxlog=False):
        """Per-bit log-likelihood ratios of received symbols, log(P(bit=0)/P(bit=1)), positive means 0

        noise_var: complex noise variance E|n|^2 of the symbols, estimated from them when None (the demodulator
            always does; passing it is for tests). LLRs are only calibrated when this is right: decoders that are scale-invariant (Viterbi, min-sum LDPC) don't care,
            log-MAP decoders do
        maxlog: use the nearest point of each bit value rather than summing over all of them

        Returns a flat array, symbol after symbol, bits in the order `index` packs them
        """
        syms = np.asarray(syms, dtype=np.complex128).reshape(-1)
        if noise_var is None:
            noise_var = self.estimate_noise(syms)
        noise_var = max(float(noise_var), 2*_MIN_VAR)
        logp = -np.abs(syms[:, None] - self.points.arr[None, :])**2/noise_var # (N, points)
        lab = self.labels().astype(bool)
        out = np.empty((len(syms), self._bps), dtype=np.float32)
        for j in range(self._bps):
            one = lab[:, j]
            if maxlog:
                out[:, j] = logp[:, ~one].max(axis=1) - logp[:, one].max(axis=1)
            else:
                out[:, j] = logsumexp(logp[:, ~one]) - logsumexp(logp[:, one])
        return out.reshape(-1)
