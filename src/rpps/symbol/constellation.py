"""Constellations: a set of points and the bit maps that say which bits each point carries

A constellation is data, not code: any point set with any labelling is one, from config/mod/constellation (`rp.symbol.load("QPSK")`) or
built from points (`rp.symbol.Constellation(points, maps)`). PSK, QAM, APSK and ASK are generators of such point sets.
"""
import math

import numpy as np

from ..base import Mod, dtype
from ..base.types import Category, Rep
from ..base.rpps import align
from ..errors import ConfigError
from ..math.prob import logsumexp, mixture_variance

from .modulation import Modulation
from .mapping import Mapping, expand_maps
from . import shapes

_MIN_VAR = 1e-6 # Floor for noise variances: a noiseless input still gives finite LLRs

class Constellation(Modulation):
    """Symbols from bits and back: `bits * c` maps bits to points, `symbols / c` gives one LLR per bit

        c = rp.symbol.load("QPSK")                       # a definition, or a generated one ("16QAM")
        syms = bits * c
        llr = syms / c                                # log(P(0)/P(1)), positive means 0

    points: complex points. They are scaled to unit mean power (what the noise estimate and the LLR scale assume) unless
        normalize is False. The number must be a power of two; a point carries log2(number) bits.
    maps: the ways of labelling the points: Mapping objects, or (labels, comment), or just labels. labels[point] is the value
        (bits, first bit most significant) the point carries. The first map is the one used unless `mapping` picks another.
        The first map is also turned by every rotation the points allow, with and without spectral inversion (see `.maps`), the
        maps a receiver that does not know the carrier phase or the IQ sense has to try. No maps: the points in order, 0, 1, 2...
    mapping: which map to use: an index into `maps`, words of its comment ("R090", "R090 inverted"), a Mapping, or labels
    """
    name = "Constellation"
    maxlog = False # Max-log LLRs (nearest point) instead of the exact sum. The noise level is always estimated from the symbols

    def __init__(self, points, maps=None, mapping=None, name=None, normalize=True):
        super().__init__()
        pts = np.asarray(points, dtype=np.complex128).reshape(-1)
        bps = int(round(math.log2(len(pts)))) if len(pts) else 0
        if len(pts) < 2 or 2**bps != len(pts):
            raise ConfigError(f"A constellation needs a power of two points (2, 4, 8...), got {len(pts)}")
        if normalize:
            pts = pts/np.sqrt(np.mean(np.abs(pts)**2))
        self.points = pts
        self.bits_per_symbol = bps
        self._weights = 1 << np.arange(bps - 1, -1, -1) # Bits are most significant first, on every host
        if name is not None:
            self.name = name # The instance's name; `type(self).name` stays the class's (the logger uses it)
        given = [self._as_mapping(m) for m in (maps if maps else [Mapping(np.arange(len(pts)), "natural")])]
        self.maps = expand_maps(pts, given)
        self.mapping = self.maps[0] if mapping is None else self._pick_map(mapping)

    def takes(self, encode):
        """Encodes channel bits (data bits too: a link with no channel code), decodes symbols"""
        return {Rep.CHANNEL, Rep.DATA} if encode else {Rep.SYMBOLS}

    # ---- header-driven control (rpps.profile.control)

    CONTROLS = {"name": "live", "mapping": "live"}

    def control(self, name=None, mapping=None):
        """Become another constellation (`name`, as `rp.symbol.load` takes it) and/or use another of its maps, from here on:
        what a header that changes the modulation does. Nothing else carries over, so it applies between symbols"""
        if name is not None:
            from .serial import load
            new = load(name)
            self.points, self.bits_per_symbol, self._weights, self.maps = new.points, new.bits_per_symbol, new._weights, new.maps
            self.name = new.name
            if mapping is None:
                self.mapping = new.mapping
        if mapping is not None:
            self.mapping = self._pick_map(mapping)

    def pos_rate(self, decode):
        """Bits per symbol decoding, symbols per bit encoding (rpps.profile positions)"""
        return self.bits_per_symbol if decode else 1/self.bits_per_symbol

    # ---- the map in use

    @property
    def order(self):
        """Number of points"""
        return len(self.points)

    @property
    def min_distance(self):
        """Distance between the two closest points"""
        d = np.abs(self.points[:, None] - self.points[None, :])
        return float(d[~np.eye(len(self.points), dtype=bool)].min())

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, mapping):
        m = self._as_mapping(mapping)
        if sorted(np.asarray(m.arr).tolist()) != list(range(self.order)):
            raise ValueError(f"A map of {self.order} points holds each of 0..{self.order - 1} once, got {np.asarray(m.arr).tolist()}")
        self._mapping = m
        self._point_of = np.empty(self.order, dtype=np.int64) # Value -> the point that carries it
        self._point_of[np.asarray(m.arr, dtype=np.int64)] = np.arange(self.order)

    def set_mapping(self, mapping):
        """Choose the bit mapping: an index into `maps`, words of its comment ("R090", "R090 inverted"), a `Mapping`, or a list of labels.
        A constellation starts on its first map, so this is only needed for another one"""
        self.mapping = self._pick_map(mapping)
        return self

    @staticmethod
    def _as_mapping(m):
        if isinstance(m, Mapping):
            return m
        if isinstance(m, dict):
            return Mapping(m["map"], m.get("comment", ""))
        if isinstance(m, tuple) and len(m) == 2 and isinstance(m[1], str):
            return Mapping(m[0], m[1])
        return Mapping(m)

    def _pick_map(self, mapping):
        if isinstance(mapping, Mapping):
            return mapping
        if isinstance(mapping, (int, np.integer)):
            return self.maps[int(mapping)]
        if isinstance(mapping, str):
            words = mapping.lower().split() # Every word has to be in the name: "R090 Inverted"
            hits = [m for m in self.maps if all(w in m.comment.lower() for w in words)]
            if len(hits) > 1 and not {"normal", "inverted"} & set(words):
                hits = [m for m in hits if not m.inverted] # Unless it says inverted, the normal sense is meant
            if len(hits) != 1:
                raise ValueError(f"{len(hits)} maps match {mapping!r}, have {[m.comment for m in self.maps]}")
            return hits[0]
        return Mapping(mapping)

    def __str__(self):
        return f"{self.name}:{self.mapping.str()}"

    # ---- symbols from bits and back

    def labels(self):
        """Bits of every point under the current map: (points, bits per symbol), first bit first"""
        arr = np.asarray(self.mapping.arr).astype(np.int64)
        return ((arr[:, None] & self._weights[None, :]) != 0).astype(np.uint8)

    def modulate(self, bits):
        """Bits to symbols (complex64). A partial last symbol is padded with zeros"""
        b = np.asarray(bits).astype(np.int64).reshape(-1)
        if len(b) % self.bits_per_symbol:
            b = np.concatenate((b, np.zeros(-len(b) % self.bits_per_symbol, dtype=np.int64)))
        value = b.reshape(-1, self.bits_per_symbol) @ self._weights
        return self.points[self._point_of[value]].astype(np.complex64)

    def encode(self, data):
        ctx = dict(data.ctx)
        bits = align(ctx, (self, True), data.bits(), self.bits_per_symbol, pad=True)
        sym = self.modulate(bits) if len(bits) else np.zeros(0, dtype=dtype.SYMBOLS.dtype)
        return data.derive(sym, Category.SYMBOL, dtype.SYMBOLS, Mod, ctx)

    def decode(self, data):
        syms = data.data
        llr = self.llr(syms, maxlog=self.maxlog) if len(syms) else np.zeros(0, dtype=dtype.LLR.dtype)
        return data.derive(llr, Category.SYMBOL, dtype.LLR, Mod, rep=Rep.CHANNEL)

    def estimate_noise(self, syms, iters=8):
        """Noise variance E|n|^2 of received symbols (assumes the constellation has unit mean power and the gain is right)

        Expectation-maximisation over the Gaussian mixture of the constellation points
        """
        y = np.asarray(syms, dtype=np.complex128).reshape(-1, 1)
        d2 = np.abs(y - self.points[None, :])**2
        return 2*mixture_variance(d2, iters, _MIN_VAR) # E|n|^2 is twice the variance per real dimension

    def llr(self, syms, noise_var=None, maxlog=False):
        """Per-bit log-likelihood ratios of received symbols, log(P(bit=0)/P(bit=1)), positive means 0

        noise_var: complex noise variance E|n|^2 of the symbols, estimated from them when None (the demodulator
            always does; passing it is for tests). LLRs are only calibrated when this is right: decoders that are scale-invariant (Viterbi, min-sum LDPC) don't care,
            log-MAP decoders do
        maxlog: use the nearest point of each bit value rather than summing over all of them

        Returns a flat array, symbol after symbol, bits in the order `modulate` takes them
        """
        syms = np.asarray(syms, dtype=np.complex128).reshape(-1)
        if noise_var is None:
            noise_var = self.estimate_noise(syms)
        noise_var = max(float(noise_var), 2*_MIN_VAR)
        logp = -np.abs(syms[:, None] - self.points[None, :])**2/noise_var # (N, points)
        lab = self.labels().astype(bool)
        out = np.empty((len(syms), self.bits_per_symbol), dtype=np.float32)
        for j in range(self.bits_per_symbol):
            one = lab[:, j]
            if maxlog:
                out[:, j] = logp[:, ~one].max(axis=1) - logp[:, one].max(axis=1)
            else:
                out[:, j] = logsumexp(logp[:, ~one]) - logsumexp(logp[:, one])
        return out.reshape(-1)

    def draw_refs(self, points: bool = True, ref: bool = True, ax=None):
        """Draw the points with their labels (and the unit circle)"""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot()
        if points:
            ax.scatter(x=self.points.real, y=self.points.imag, s=200, c="r")
            for p, label in zip(self.points, self.mapping.arr):
                ax.annotate(bin(int(label))[2:].zfill(self.bits_per_symbol), (p.real, p.imag), fontsize=20)
        if ref:
            angle = np.linspace(0, 2*np.pi, 150)
            ax.plot(np.cos(angle), np.sin(angle), "g")

    # ---- definitions

    @staticmethod
    def load(name: str, obj: dict):
        """{"base": "constellation", "points": [...], "maps": [{"map": [...], "comment": "..."}], "normalize": true}

        A point is {"mag": 1, "pha": 45} (phase in degrees), {"re": 0.7, "im": -0.7} or [re, im]
        """
        def point(p):
            if isinstance(p, dict):
                if "mag" in p:
                    return p["mag"]*np.exp(1j*np.deg2rad(p.get("pha", 0.0)))
                return complex(p["re"], p.get("im", 0.0))
            if isinstance(p, (list, tuple)) and len(p) == 2:
                return complex(p[0], p[1])
            return complex(p)
        if "points" not in obj:
            raise ConfigError(f"{name}: a constellation needs 'points'")
        return Constellation([point(p) for p in obj["points"]], obj.get("maps"), name=name, normalize=obj.get("normalize", True))

# ---- Generators: the usual point sets

def _gen(M, shape, labels, comment, name, mapping, normalize=True):
    return Constellation(shape, [Mapping(np.asarray(labels), comment)], mapping, name=name, normalize=normalize)

def PSK(M=4, mapping=None, offset=None):
    """M points on the unit circle with a Gray map; offset: the first point's phase [rad], default pi/M (QPSK at 45 degrees)"""
    pts, lab = shapes.psk(M, offset)
    return _gen(M, pts, lab, "Gray", f"{M}PSK", mapping)

def QAM(M=16, mapping=None):
    """Square (4, 16, 64, 256) and cross (32, 128, 512) QAM; square sizes are Gray, cross sizes as close as a cross allows"""
    pts, lab = shapes.qam(M)
    return _gen(M, pts, lab, "Gray", f"{M}QAM", mapping)

_APSK_RINGS = {16: (4, 12), 32: (4, 12, 16), 8: (1, 7), 64: (4, 12, 20, 28)}

def APSK(M=16, mapping=None, rings=None, ratios=None):
    """Amplitude-phase keying on rings: `APSK(16)` is 4+12, `APSK(32)` 4+12+16; or give the rings, `APSK(rings=(8, 8))`
    ratios: radius of the outer rings over the innermost"""
    rings = _APSK_RINGS[M] if rings is None else tuple(rings)
    pts, lab = shapes.apsk(rings, ratios)
    return _gen(sum(rings), pts, lab, "ring + Gray", f"{sum(rings)}APSK", mapping)

def ASK(M=2, mapping=None, bipolar=None):
    """Amplitude levels on the real axis: `ASK(2)` is on-off keying, `ASK(4)` has four Gray levels; bipolar: symmetric about zero
    (PAM, the default for M > 2) rather than 0..1"""
    bipolar = M > 2 if bipolar is None else bipolar
    pts, lab = shapes.ask(M, bipolar)
    return _gen(M, pts, lab, "Gray", f"{M}ASK" if M > 2 else "OOK", mapping)
