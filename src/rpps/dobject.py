"""Containers for data to ensure we can track where it came from

StreamData:
 StreamData @ Scram  -> ScrambledData
 StreamData @ Coding -> CodedData
 StreamData @ Mod    -> SymData

 ScrambledData @ Scram -> ScramData|StreamData
 CodedData @ Coding -> CodingData|ScramData|StreamData
 SymData @ Mod -> ModData
 ModData @ Coding -> CodingData
 ModData @ Scram -> ScramData

"""

from .base.stream import Stream
from .base.bitarray import bitarray
from .meta import Meta

class DataObject:
    __slots__ = ("stream", "meta")

    def __init__(self, stream=None, meta=None):
        if stream is None:
            stream = Stream()
        if meta is None:
            meta = Meta()
        self.stream = stream
        self.meta = meta

    def __len__(self):
        return len(self.stream)

    def name(self):
        return type(self).__name__

    def from_bitarray(self, ba: bitarray):
        self.stream = Stream.from_bitarray(ba)

class StreamData(DataObject):
    """Raw input/output data"""
    def __str__(self):
        return f"StreamData: {str(self.stream)}"


# --- Scram
class ScramData(DataObject):
    pass

# --- Coding
class CodingData(DataObject):
    """Data post-coding"""
    pass

# --- Mod
class ModData(DataObject):
    """"""
    pass
class SymData(DataObject):
    def __init__(self, data):
        super().__init__()
        self.stream = data
