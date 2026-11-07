import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

hot = colors.LinearSegmentedColormap.from_list("RT-hot", (
    (0.00, (0.0, 0.0, 0.0)),
    (0.05, (0.0, 0.0, 0.1)),
    (0.10, (0.0, 0.0, 0.4)),
    (0.50, (0.0, 0.8, 0.0)),
    (1.00, (1.0, 0.0, 0.0)),
))

fig, ax = plt.subplots(3)

# --- small test case first ---
# arr = np.zeros((6, 3), dtype=int)

matrix = np.array([
    [0,0,0,0,2,0,0,0],
    [0,0,0,2,0,2,0,0],
    [0,0,0,0,0,0,0,0],
    [0,1,1,2,0,2,1,1],
    [0,0,0,1,0,1,0,0],
    [1,1,0,0,0,0,0,1],
], dtype=np.int8)



print(matrix.shape)
print(matrix)
ax[0].imshow(matrix / np.max(matrix), cmap=hot, vmin=0, vmax=1)

rows, cols = matrix.shape
mask = matrix != 0
s_idx = np.argmax(mask, axis=0)
e_idx = rows - 1 - np.argmax(mask[::-1], axis=0)

empty = mask.sum(axis=0) == 0
s_idx[empty] = 0
e_idx[empty] = 0

diff = np.zeros_like(matrix, dtype=np.int8)
col_idx = np.arange(cols)

s_rows = s_idx + 1
s_valid = s_rows < e_idx
diff[s_rows[s_valid], col_idx[s_valid]] += 1

e_rows = e_idx
e_valid = e_idx > s_idx
diff[e_rows[e_valid], col_idx[e_valid]] -= 1
ax[1].imshow(diff/ np.max(diff), cmap=hot, vmin=0, vmax=1)
inc = np.cumsum(diff, axis=0)
matrix += inc

print(matrix)
matrix = matrix / np.max(matrix)
ax[2].imshow(matrix, cmap=hot, vmin=0, vmax=1)
plt.show()
