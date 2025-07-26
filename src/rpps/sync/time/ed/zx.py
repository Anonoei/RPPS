"""Zero-crossing TED
- This requires SPS>=2
- Carrier independent
"""

import numpy as np

def zx_nd(early, mid, late):
    """Gardner TED
    early, (mTm)        : near-peak sample (current symbol)
    mid,   (mTm + Tm/2) : near-zero sample
    late,  ((m+1)Tm)    : near-peak sample (next symbol)
    """
    return mid*(np.conj(early) - np.conj(late))
