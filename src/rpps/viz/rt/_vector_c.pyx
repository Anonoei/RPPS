# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport fabs

def dot_vector(np.ndarray[np.int8_t, ndim=2] np_matrix):
    cdef int height = np_matrix.shape[0]
    cdef int width = np_matrix.shape[1]

    cdef np.int8_t[:, :] matrix = np_matrix
    cdef int x, y, i, j, y_start, y_end
    cdef int nn, nn_off
    cdef double min_dist, dist

    # Fixed-size lists for nonzero indices (worst-case: all rows nonzero)
    cdef int[:] f1 = np.empty(height, dtype=np.int32)
    cdef int[:] fx1 = np.empty(height, dtype=np.int32)
    cdef int f1_size, fx1_size

    for x in range(1, width):
        f1_size = 0
        fx1_size = 0

        # Manually find nonzero indices in matrix[:, x]
        for y in range(height):
            if matrix[y, x] == 1:
                f1[f1_size] = y
                f1_size += 1
            if matrix[y, x - 1] == 1:
                fx1[fx1_size] = y
                fx1_size += 1

        if f1_size == 0 or fx1_size == 0:
            continue

        if f1_size > 1:
            for i in range(f1_size):
                y = f1[i]
                min_dist = 1e9
                nn = -1
                for j in range(fx1_size):
                    dist = fabs(fx1[j] - y)
                    if dist < min_dist:
                        min_dist = dist
                        nn = j

                nn_off = fx1[nn] - y
                if nn_off < 0:
                    y_start = y + nn_off + 1
                    nn_off = -nn_off
                else:
                    y_start = y

                y_end = y_start + nn_off
                if 0 <= y_start < height and 0 <= y_end <= height:
                    for j in range(y_start, y_end):
                        matrix[j, x] = 1
        else:
            y = f1[0]
            min_dist = 1e9
            nn = -1
            for j in range(fx1_size):
                dist = fabs(fx1[j] - y)
                if dist < min_dist:
                    min_dist = dist
                    nn = j

            nn_off = fx1[nn] - y
            if nn_off == 0:
                continue

            if nn_off > 0:
                y_start = y
                y_end = y + nn_off
            else:
                y_start = y + nn_off + 1
                y_end = y + 1

            if 0 <= y_start < height and 0 <= y_end <= height:
                for j in range(y_start, y_end):
                    matrix[j, x] = 1

    return np_matrix
