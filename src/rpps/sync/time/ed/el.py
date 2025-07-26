"""Early-late TED
- This requires SPS>=2
- Carrier independent
"""

import numpy as np

def el_nd(early, mid, late):
    """
    early, (mTm)       : near-peak sample (symbol)
    mid,  (mTm - Tm/2) : near-zero sample
    late, (mTm + Tm/2) : near-zero sample
    """
    return mid*(np.conj(late) - np.conj(early))
