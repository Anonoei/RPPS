"""Constellations: a set of points and the bit maps that say which bits each point carries

A constellation is data, not code: any point set with any labelling is one, from config/constellation (`rp.symbol.load("QPSK")`) or
built from points (`rp.symbol.Constellation(points, maps)`). PSK, QAM, APSK and ASK are generators of such point sets.
"""
import math

import numpy as np

from ..base import Mod, dtype
from ..base.types import Category, Rep
from ..base.rpps import align, FRAME
from ..errors import ConfigError
from ..math import demap
from .. import _effort

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
        A map with a `turn` (radians; "rotate" in degrees in a definition) uses the points turned by it: a standard that
        labels the same set turned (DVB-S2 8PSK at k*45 degrees on 8PSK at 22.5 + k*45) is a map, not another constellation.
        Such maps are in `turned`, picked by name; `maps` holds the ones on the points as given. `blind_maps`: `maps`, then
        each turned map with every turn and inversion of it the points allow, what a blind search tries.
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
        self._base = pts # The points as given; a map with a turn uses them turned (`points`)
        self.points = pts
        self.bits_per_symbol = bps
        self._weights = 1 << np.arange(bps - 1, -1, -1) # Bits are most significant first, on every host
        if name is not None:
            self.name = name # The instance's name; `type(self).name` stays the class's (the logger uses it)
        given = [self._as_mapping(m) for m in (maps if maps else [Mapping(np.arange(len(pts)), "natural")])]
        every = expand_maps(pts, given)
        self.maps = [m for m in every if not m.turn] # The maps on these points: what a receiver that does not know the phase tries
        self.turned = [m for m in every if m.turn] # Maps that turn the points (a standard's labelling of the set turned): by name only
        self.blind_maps, self._family = list(self.maps), {} # (_family: id of a turn of a turned map -> that map's name)
        for t in self.turned:
            for m in self._turns_of(pts, t):
                self.blind_maps.append(m)
                self._family[id(m)] = t._comment
        self.mapping = self.maps[0] if mapping is None else self._pick_map(mapping)

    def turned_name(self, m):
        """The name of the turned map `m` (one of `blind_maps`) is a turn of ("DVB-S2"), as `load(name, mapping)` takes it;
        None for a map on the points as given"""
        return self._family.get(id(m)) if m.turn else None

    @staticmethod
    def _turns_of(pts, t):
        """A turned map and each turn and inversion of it the points allow (named R<degrees> <its name>): what a receiver
        whose carrier phase is known only to the points' symmetry tries of it"""
        return [t] + expand_maps(pts, [t])[1:]

    def takes(self, encode):
        """Encodes channel bits (data bits too: a link with no channel code), decodes symbols"""
        return {Rep.CHANNEL, Rep.DATA} if encode else {Rep.SYMBOLS}

    # ---- header-driven control (rpps.profile.control)

    CONTROLS = {"name": "live", "mapping": "live", "params": "live"}
    SCOPE = FRAME # Whole symbols of bits; nothing is held across frames

    def control(self, name=None, mapping=None, params=None):
        """Become another constellation (`name`, as `rp.symbol.load` takes it, with the definition's `params`, a dict) and/or
        use another of its maps, from here on: what a header that changes the modulation does. Nothing else carries over,
        so it applies between symbols"""
        if params is not None and name is None:
            raise ValueError("control: params are the definition's, so they come with a name")
        if name is not None:
            from .serial import load
            new = load(name, **(params or {}))
            self._base, self.points, self.bits_per_symbol, self._weights = new._base, new.points, new.bits_per_symbol, new._weights
            self.maps, self.turned, self.blind_maps, self._family = new.maps, new.turned, new.blind_maps, new._family
            self.name = new.name
            if mapping is None:
                self.mapping = new.mapping
        if mapping is not None:
            self.mapping = self._pick_map(mapping)

    def kernel(self, params, how):
        """The batch demapper, for the constellation as `params` (live parameters) make it; encoding has none yet"""
        if how != "/":
            return None
        from .kernels import Demap
        if params:
            self.control(**params)
        return Demap(self.points.copy(), self.labels().astype(bool), _effort.choice("maxlog", self.maxlog), self.name, floor=_MIN_VAR) # (the level when the region is built)

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
        self.points = self._base*np.exp(1j*m.turn) if m.turn else self._base
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
            return Mapping(m["map"], m.get("comment", ""), np.deg2rad(m.get("rotate", 0.0)), m.get("tags", ()))
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
            hits = [m for m in self.maps + self.turned if all(w in m.comment.lower() for w in words)]
            if not hits: # A turn of a turned map (the blind tools name them: "R045 DVB-S2")
                hits = [m for m in self.blind_maps[len(self.maps):] if all(w in m.comment.lower() for w in words)]
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
        llr = self.llr(syms, maxlog=_effort.choice("maxlog", self.maxlog)) if len(syms) else np.zeros(0, dtype=dtype.LLR.dtype)
        return data.derive(llr, Category.SYMBOL, dtype.LLR, Mod, rep=Rep.CHANNEL)

    def estimate_noise(self, syms, iters=8):
        """Noise variance E|n|^2 of received symbols (assumes the constellation has unit mean power and the gain is right)

        Expectation-maximisation over the Gaussian mixture of the constellation points
        """
        y = np.asarray(syms, dtype=np.complex128).reshape(1, -1)
        d2, dmin = demap.distances(np, y, self.points)
        return float(2*demap.noise_variance(np, d2, dmin, iters, _MIN_VAR)[0]) # E|n|^2 is twice the variance per real dimension

    def llr(self, syms, noise_var=None, maxlog=False):
        """Per-bit log-likelihood ratios of received symbols, log(P(bit=0)/P(bit=1)), positive means 0

        noise_var: complex noise variance E|n|^2 of the symbols, estimated from them when None (the demodulator
            always does; passing it is for tests). LLRs are only calibrated when this is right: decoders that are scale-invariant (Viterbi, min-sum LDPC) don't care,
            log-MAP decoders do
        maxlog: use the nearest point of each bit value rather than summing over all of them

        Returns a flat array, symbol after symbol, bits in the order `modulate` takes them. One implementation with the
        batch demapper (rpps.math.demap): a frame in a batch comes out as it does alone
        """
        y = np.asarray(syms, dtype=np.complex128).reshape(1, -1)
        lab = self.labels().astype(bool)
        return demap.llr_rows(np, y, self.points, lab, demap.mask_matrix(lab), noise_var, maxlog, floor=_MIN_VAR)[0]

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
        """{"base": "constellation", "points": [...] or "rings": [...], "rotate": 0, "maps": [{"map": [...], "comment": "..."}], "normalize": true}

        A point is {"mag": 1, "pha": 45} (phase in degrees), {"re": 0.7, "im": -0.7} or [re, im].
        A ring is {"n": 12, "mag": 2.85, "pha": 15}: n points of magnitude mag (default 1), the first at pha degrees (default
        180/n, so a 4-point ring is QPSK at 45 degrees) and the rest evenly round, counter-clockwise; rings are listed innermost
        first and the points come out ring by ring, so a map indexes them in that order (PSK is one ring, APSK several).
        rotate: degrees added to every point's phase (QPSK on the axes is QPSK rotated by -45)
        A map may say "rotate" too: degrees the points are turned by while that map is used (DVB-S2 8PSK on 8PSK)
        """
        def point(p):
            if isinstance(p, dict):
                if "mag" in p:
                    return p["mag"]*np.exp(1j*np.deg2rad(p.get("pha", 0.0)))
                return complex(p["re"], p.get("im", 0.0))
            if isinstance(p, (list, tuple)) and len(p) == 2:
                return complex(p[0], p[1])
            return complex(p)
        if ("points" in obj) == ("rings" in obj):
            raise ConfigError(f"{name}: a constellation has 'points' or 'rings' (one of them)")
        if "rings" in obj:
            pts = []
            for r in obj["rings"]:
                n = int(r["n"])
                pha = r.get("pha", 180.0/n)
                pts += [r.get("mag", 1.0)*np.exp(1j*np.deg2rad(pha + 360.0*k/n)) for k in range(n)]
        else:
            pts = [point(p) for p in obj["points"]]
        pts = np.asarray(pts, dtype=np.complex128)*np.exp(1j*np.deg2rad(obj.get("rotate", 0.0)))
        return Constellation(list(pts), obj.get("maps"), name=name, normalize=obj.get("normalize", True))

# ---- Generators: the usual point sets

def _gen(M, shape, labels, comment, name, mapping, normalize=True):
    return Constellation(shape, [Mapping(np.asarray(labels), comment)], mapping, name=name, normalize=normalize)

def PSK(M=4, mapping=None, offset=None):
    """M points on the unit circle with a Gray map; offset: the first point's phase [rad], default pi/M (QPSK at 45 degrees),
    0 for BPSK (+-1, as `load("psk/bpsk")`)"""
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
