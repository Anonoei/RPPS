"""Early-late TED
- This requires SPS>=2
- Carrier independent
"""

from ._ed import TED_ND
import numpy as np

def el_nd(early, mid, late):
    """
        early, (mTm)       : near-peak sample (symbol)
        mid,  (mTm - Tm/2) : near-zero sample
        late, (mTm + Tm/2) : near-zero sample
    """
    return np.real(mid*(np.conj(late) - np.conj(early)))

class EL_ND(TED_ND):
    iSPS = 2
    Inputs = 3
    Lookahead = True
    Derivative = False

    def run(self):
        """
        samples[0] = late, (mTm + Tm/2) : near-zero sample
        samples[1] = mid,  (mTm - Tm/2) : near-zero sample
        samples[2] = early, (mTm)       : near-peak sample (symbol)
        """
        r_term = (self.samples[0].real - self.samples[2].real) * self.samples[1].real
        i_term = (self.samples[0].real - self.samples[2].real) * self.samples[1].iamg
        return r_term + i_term
