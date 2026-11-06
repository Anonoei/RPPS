import numpy as np

def entropy(bits):
    """
    H(X) = -p*log2(p-(1-p))*log2(1-p)
    """
    p0 = np.mean(bits == 0)
    p1 = np.mean(bits == 1)
    return -p0 * np.log2(p0) - p1 * np.log2(p1)
