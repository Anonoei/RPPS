"""Modulation parent classes"""

import numpy as np
import matplotlib.pyplot as plt

from pyboiler.logger import Logger, Level

from . import Meta
from . import base
from . import dobject


class Scram(base.rpps.Pipe):
    """Scram Pipe"""

    def __init__(self):
        self.log = (
            Logger().Child("Coding", Level.WARN).Child(type(self).__name__, Level.WARN)
        )

    def __str__(self):
        return f"{type(self).__name__}"

    def init_meta(self, meta: Meta):
        """Initialize scram metadata"""
        meta.coding.fields["Name"] = type(self).__name__
        meta.coding.fields["RateNum"] = None
        meta.coding.fields["RateDen"] = None

    def scram(self, dobj: dobject.BitObject) -> dobject.ScramData:
        """Encode dobject using specified scram"""
        ...

    def descram(self, dobj: dobject.BitObject) -> dobject.BitObject:
        """Decode dobject using specified scram"""
        ...

    def __rmul__(self, other):
        return self.scram(dobject.ensure_bit(other))

    def __rtruediv__(self, other):
        return self.descram(dobject.ensure_bit(other))


class Feedthrough(Scram):
    seed = np.ndarray((16,), dtype=bool)
    taps = np.ndarray((4,), dtype=int)

    def __str__(self):
        return f"{type(self).__name__}:{"-".join([str(i) for i in self.taps])}"

    def scram(self, dobj: dobject.BitObject, _cache={}):
        if _cache.get("lfsr", None) is None:
            _cache["lfsr"] = np.copy(self.seed)  # Initialize the scrambler state

        scrambled_data = np.empty_like(dobj.data, dtype=bool)

        for i, bit in enumerate(dobj.data):
            # Xor taps sequentially
            new_bit = np.bitwise_xor.reduce(_cache["lfsr"][self.taps - 1]) & 1
            # shift one place to the right, [1,0,1,0,1,#] >> 1 = [#,1,0,1,0,1]
            _cache["lfsr"][1:] = _cache["lfsr"][:-1]
            _cache["lfsr"][0] = new_bit  # set left-most bit
            # output right-most bit
            scrambled_data[i] = _cache["lfsr"][-1] ^ bit

        return dobject.ScramData(scrambled_data)

    def descram(self, dobj: dobject.BitObject, _cache={}):
        if _cache.get("lfsr", None) is None:
            _cache["lfsr"] = np.copy(self.seed)  # Initialize the scrambler state

        descrambled_data = np.empty_like(dobj.data, dtype=bool)

        for i, bit in enumerate(dobj.data):
            # Xor taps sequentially
            new_bit = np.bitwise_xor.reduce(_cache["lfsr"][self.taps - 1]) & 1
            # shift one place to the right, [1,0,1,0,1,#] >> 1 = [#,1,0,1,0,1]
            _cache["lfsr"][1:] = _cache["lfsr"][:-1]
            _cache["lfsr"][0] = new_bit  # set left-most bit
            # output right-most bit
            descrambled_data[i] = _cache["lfsr"][-1] ^ bit

        return dobject.BitObject(descrambled_data)

    @staticmethod
    def load(name, obj):
        i_s = np.array(obj["seed"], dtype=bool)
        i_t = np.array(obj["taps"], dtype=int)
        impl = type(name, (Feedthrough,), dict(name=name, seed=i_s, taps=i_t))()
        return impl


class Additive(Scram):

    @staticmethod
    def load(name, obj):
        impl = type(name, (Additive,), dict(name=name, poly=obj["poly"]))()
        return impl
