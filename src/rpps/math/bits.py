"""Bit manipulation"""

def reflect(v, width):
    """Reverse the order of the low `width` bits of an integer"""
    return int(format(v, f"0{width}b")[::-1], 2)

def to_base2(order):
    """Every `order`-bit pattern in counting order: (2^order, order) array of 0/1, most significant bit first"""
    import numpy as np
    return ((np.arange(2**order)[:, None] >> np.arange(order - 1, -1, -1)) & 1)

def to_base10(arr):
    """Rows of bits (most significant first) -> integers"""
    import numpy as np
    arr = np.asarray(arr)
    return np.dot(arr, 2**np.arange(arr.shape[1] - 1, -1, -1))
