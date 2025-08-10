"""Zero-crossing TED
- This requires SPS>=2
- Carrier independent
"""

from ._ed import TED_ND
import numpy as np

def zx_nd(early, mid, late):
    """Gardner TED
    early, (mTm)        : near-peak sample (current symbol)
    mid,   (mTm + Tm/2) : near-zero sample
    late,  ((m+1)Tm)    : near-peak sample (next symbol)
    """
    return mid*(np.conj(early) - np.conj(late))

class ZX_ND(TED_ND):
    iSPS = 2
    Inputs = 3
    Lookahead = False
    Derivative = False

    def run(self):
        """
        samples[0] = late,  ((m+1)Tm)    : near-peak sample (next symbol)
        samples[1] = mid,   (mTm + Tm/2) : near-zero sample
        samples[2] = early, (mTm)        : near-peak sample (current symbol)
        """
        r_term = (self.samples[2].real - self.samples[0].real) * self.samples[1].real
        i_term = (self.samples[2].imag - self.samples[0].imag) * self.samples[1].imag
        return r_term + i_term
