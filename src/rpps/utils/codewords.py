import numpy as np

def to_base2(order):
    # Generate all binary numbers from 0 to n (inclusive)
    n = 2**order
    binary_nums = []
    for i in range(n):
        binary = format(i, 'b').zfill(order)  # Convert to binary and pad with zeros
        binary_nums.append([int(x) for x in binary])
    return np.array(binary_nums)

def to_base10(arr):
    return np.dot(arr, 2**np.arange(arr.shape[1]-1, -1, -1))
