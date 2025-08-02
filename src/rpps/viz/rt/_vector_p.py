import numpy as np

def dot_vector(matrix):
    for x in range(1, matrix.shape[1]):
        f1 = np.where(matrix[:,x]==1)[0]
        if f1.size > 0:
            fx1 = np.where(matrix[:,x-1]==1)[0]
            if f1.size > 1:
                dist = fx1[:,None]-f1
                nn = np.argmin(np.abs(dist), axis=0)
                nn_off = dist[nn, np.arange(f1.size)]

                ys = f1
                mask = nn_off < 0
                ys[mask] += nn_off[mask]+1
                nn_off[mask] = np.abs(nn_off[mask])
                for i in range(ys.size):
                    matrix[ys[i]:ys[i]+nn_off[i],x] = 1
            else:
                for y in f1:
                    dist = fx1-y
                    nn = np.argmin(np.abs(dist))
                    nn_off = dist[nn]
                    if nn_off == 0:
                        continue

                    ys, nn_off = (y, nn_off) if nn_off >= 0 else (y+nn_off+1, np.abs(nn_off))
                    matrix[ys:ys+nn_off,x] = 1
    return matrix
