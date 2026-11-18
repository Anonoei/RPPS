import numpy as np

def entropy(bits):
    """Entropy of a bit stream [bits per bit]: H = -p0 log2 p0 - p1 log2 p1 (0 log 0 = 0)"""
    bits = np.asarray(bits)
    p = np.array([np.mean(bits == 0), np.mean(bits == 1)])
    p = p[p > 0]
    return float(-np.sum(p*np.log2(p)))
