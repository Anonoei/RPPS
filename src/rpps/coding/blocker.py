import numpy as np

def block(bits: np.ndarray, block_size: int):
    return bits.reshape(-1, block_size)

def unblock(bits: np.ndarray):
    return bits.reshape(-1)

def hamming_dist(ib: np.ndarray, ob: np.ndarray):
    return np.bitwise_xor(ib, ob).sum()
