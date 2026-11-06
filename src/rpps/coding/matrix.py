"""Linear block code matrices. The maths lives in rpps.math.gf2; these are the names rp.coding.matrix always had"""
from ..math.gf2 import parity_to_gh as linear, gh_to_parity

def linear_p(n, k, G=None, H=None):
    """Parity block from G (shape (k, n-k)) or from H (shape (n-k, k): H's left block as it is, so transpose it for `linear`)"""
    if G is not None:
        return gh_to_parity(n, k, G=G)
    return gh_to_parity(n, k, H=H).T
