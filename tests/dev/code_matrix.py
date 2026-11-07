import rpps as rp

import numpy as np

def hamming_3_1():
    print("Hamming [3,1]")
    n, k, d = 3, 1, 2
    H = np.array([
        [1,1,0],
        [1,0,1],
    ])
    p = rp.coding.matrix.linear_p(n,k,H=H)
    print("P:")
    p = p.T # Transpose because it was generated from H
    print(p.astype(int))
    G, H = rp.coding.matrix.linear(n,k,p)
    print("G:")
    print(G.astype(int))
    print("H:")
    print(H.astype(int))

def hamming_7_4():
    print("Hamming [7,4]")
    n, k, d = 7, 4, 3
    p = np.array([
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [1,1,1],
    ])
    G, H = rp.coding.matrix.linear(n,k,p)
    print("G:")
    print(G.astype(int))
    print("H:")
    print(H.astype(int))

def hamming_15_11():
    print("Hamming [15,11]")
    n, k, d = 15, 11, 4
    H = np.array([
        [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
        [0,1,1,1,0,0,0,1,1,1,1,0,1,0,0],
        [1,0,1,1,0,1,1,0,0,1,1,0,0,1,0],
        [1,1,0,1,1,0,1,0,1,0,1,0,0,0,1]
    ])
    p = rp.coding.matrix.linear_p(n,k,H=H)
    p = p.T # came from H, need to transpose
    print("P:")
    print(p.astype(int))
    G, H = rp.coding.matrix.linear(n,k,p)
    print("G:")
    print(G.astype(int))
    print("H:")
    print(H.astype(int))


if __name__ == "__main__":
    hamming_7_4()
