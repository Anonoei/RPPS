"""Symbol timing: where to sample, by block estimation (rpps.sync). `OerderMeyr` (the square-law line, tracked across
windows), `MatTrack` (the matched filter fused in and evaluated only at the instants, timing from its energy maximum,
tracked across windows; on a GPU too), `Mat` (matrix matched filter, per block), `Known` when the timing is given
"""
from .mat import Mat
from .om import OerderMeyr
from .matrack import MatTrack
from .known import Known
