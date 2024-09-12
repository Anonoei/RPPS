"""Modulation parent classes"""

import numpy as np
import matplotlib.pyplot as plt

from pyboiler.logger import Logger, Level

from . import Meta
from . import base
from . import dobject
from . import lfsr


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
    def __init__(self, scram_lfsr: lfsr.LFSR, descram_lfsr: lfsr.LFSR):
        super().__init__()
        self.s_lfsr = scram_lfsr
        self.d_lfsr = descram_lfsr
    def __str__(self):
        return f"{type(self).__name__}:{self.s_lfsr}"

    def reset(self):
        self.s_lfsr.reset()
        self.d_lfsr.reset()

    def scram(self, dobj: dobject.BitObject):
        scrambled_data = np.empty_like(dobj.data, dtype=bool)

        for i, bit in enumerate(dobj.data):
            scrambled_data[i] = self.s_lfsr.get_bit() ^ bit

        return dobject.ScramData(scrambled_data)

    def descram(self, dobj: dobject.BitObject):
        descrambled_data = np.empty_like(dobj.data, dtype=bool)

        for i, bit in enumerate(dobj.data):
            descrambled_data[i] = self.d_lfsr.get_bit() ^ bit

        return dobject.BitObject(descrambled_data)

    @staticmethod
    def load(name, obj):
        i_lfsr = getattr(lfsr, obj["type"])
        i_seed = np.array(obj["seed"], dtype=bool)
        i_poly = np.array(obj["poly"], dtype=int)
        i_s_lfsr = i_lfsr(np.copy(i_seed), np.copy(i_poly))
        i_d_lfsr = i_lfsr(np.copy(i_seed), np.copy(i_poly))
        impl = type(name, (Feedthrough,), dict())(i_s_lfsr, i_d_lfsr)
        return impl


class Additive(Scram):

    @staticmethod
    def load(name, obj):
        impl = type(name, (Additive,), dict(name=name, poly=obj["poly"]))()
        return impl
