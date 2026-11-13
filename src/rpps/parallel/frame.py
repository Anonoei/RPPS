"""The unit of parallel work: one frame in, one result out (README.dev.md, Parallelism)

A `Frame` holds everything needed to process it, and nothing shared: no Stream, no Meta, no ctx. It can be pickled, put in
shared memory or sent to a device. `params` are the controlled parameters the serial front already resolved for it
(`{step id: {parameter: value}}`), so a worker never reads a header or a table.
"""
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np

# What a block of data is: the class and tags of rpps.Data (CAT, DT, REP) and the rates and delay it carries. A tuple, so it
# is picklable and hashable
Form = namedtuple("Form", "cls cat dt rep Fs sps cf delay")

OK, DROPPED = "ok", "dropped"

@dataclass
class Frame:
    """One frame of a parallel region's input

    seq: its order (0, 1, 2 ... per stream); results come out in this order.
    pos: absolute position of its first item in the region's input; n_out: items it will give, and pos_out where, both
        worked out by the front from the step maps, so output positions do not depend on completion order.
    tags: what the data is (CAT, DT, REP of rpps.Data), to put it back together.
    """
    seq: int
    data: np.ndarray
    params: dict = field(default_factory=dict)
    pos: int = 0
    pos_out: int = 0
    n_out: int = 0
    tags: tuple = ()

    @property
    def key(self):
        """What frames must share to run as one batch: their parameters (a MODCOD, a pilot flag) and shape"""
        return (self.data.shape, self.data.dtype.str, _freeze(self.params))

    def __len__(self):
        return len(self.data)

@dataclass
class Result:
    """What a frame gave. status DROPPED: the frame could not be processed (a gap of `n_out` items at `pos_out`), why in
    `reports`, which go to the stream's control reports in order"""
    seq: int
    data: np.ndarray
    pos_out: int = 0
    n_out: int = 0
    status: str = OK
    reports: list = field(default_factory=list)
    tags: tuple = ()

    @staticmethod
    def dropped(frame, why):
        return Result(frame.seq, frame.data[:0], frame.pos_out, frame.n_out, DROPPED,
                      [{"what": "dropped", "frame": frame.seq, "at": frame.pos, "note": str(why)}], frame.tags)

def _freeze(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, np.ndarray):
        return (v.shape, v.dtype.str, v.tobytes())
    return v
