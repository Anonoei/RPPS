import numpy as np

mm = 4 # RS code over GF(2**4)
nn = 15 # nn=2**mm-1, length of codeword
tt = 3 # number of correctable errors
kk = 9 # kk = nn-3*tt

pp = np.array([1,1,0,0,1], dtype=int)
alpha_to = np.zeros(nn+1, dtype=int)
index_of = np.zeros(nn+1, dtype=int)
gg = np.zeros(nn-kk+1, dtype=int)

recd = np.zeros(nn, dtype=int)
data = np.zeros(kk, dtype=int)
bb = np.zeros(nn-kk, dtype=int)

def gen_gf():
    mask = 1
    alpha_to[mm] = 0
    for i in range(mm):
        alpha_to[i] = mask
        index_of[alpha_to[i]] = i
        if not pp[i] == 0:
            alpha_to[mm] ^= mask
        mask <<= 1
    index_of[alpha_to[mm]] = mm
    mask >>= 1
    for i in range(mm+1, nn):
        if alpha_to[i-1] >= mask:
            alpha_to[i] = alpha_to[mm] ^ ((alpha_to[i-1]^mask)<<1)
        else:
            alpha_to[i] = alpha_to[i-1]<<1
        index_of[alpha_to[i]] = i
    index_of[0] = -1

def gen_poly():
    gg[0] = 2 # primitive element alpha=2 for GF(2**mm)
    gg[1] = 1 # g(x) = (X+alpha) initially
    for i in range(2, nn-kk+1):
        gg[i] = 1
        for j in range(i-1, 0, -1):
            if not gg[j] == 0:
                gg[j] = gg[j-1] ^ alpha_to[(index_of[gg[j]]+i) % nn]
            else:
                gg[j] = gg[j-1]
        gg[0] = alpha_to[(index_of[gg[0]]+i) % nn] # gg[0] can never be 0
    for i in range(nn-kk+1):
        gg[i] = index_of[gg[i]]

def encode_rs():
    for i in range(nn-kk):
        bb[i] = 0
    for i in range(kk-1, -1, -1):
        feedback = index_of[data[i] ^ bb[nn-kk-1]]
        if not feedback == -1:
            for j in range(nn-kk-1, 0, -1):
                if not gg[j] == -1:
                    bb[j] = bb[j-1]^alpha_to[(gg[j]+feedback) % nn]
                else:
                    bb[j] = bb[j-1]
                bb[0] = alpha_to[(gg[0]+feedback) % nn]
        else:
            for j in range(nn-kk-1, 0, -1):
                bb[j] = bb[j-1]
            bb[0] = 0

def main():
    gen_gf()
    print(f"LUT tables for GF(2**{mm})")
    print("  i   alpha_to[i]  index_of[i]")
    for i in range(nn+1):
        print(f"{i:03}      {alpha_to[i]:03}          {index_of[i]:03}")
    print("\n")

    gen_poly()

    for i in range(kk):
        data[i] = 0
    data[0] = 0
    data[1] = 6
    data[2] = 8
    data[3] = 1
    data[4] = 2
    data[5] = 4
    data[6] = 15
    data[7] = 9
    data[8] = 9

    encode_rs()

    for i in range(nn-kk):
        recd[i] = bb[i]
    for i in range(kk):
        recd[i+nn-kk] = data[i]

    print(recd)

if __name__ == "__main__":
    main()
