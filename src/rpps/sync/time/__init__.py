"""Symbol timing: where to sample. First class `OerderMeyr` (block estimation, tracked across windows) and `MatTrack`
(the matched filter fused in and evaluated only at the instants, timing from its energy maximum, tracked across windows;
on a GPU too); `Mat` (matrix matched filter, per block); second class the loops `MM` (Mueller & Muller, Gardner) and `TLL` around a detector from
`ted`; `Known` when the timing is given (rpps.sync)
"""
from . import ted
from .mm import MM
from .mat import Mat
from .om import OerderMeyr
from .matrack import MatTrack
from .tll import TLL
from .known import Known
