import numpy as np

from typing import overload

def GF2(m, p=np.array([1,1,0,0,1], dtype=int)):
    """GF(2**m, poly)

    Description:
        lookup tables:
            index->polynomial form  alpha_to[] contains `j=alpha**i`
            polynomial form->index  index_of[`j=alpha**i`] = i

        alpha=2 is the primitive element of GF2(m)

        See: https://www.eccpage.com/rs.c

    Args:
        m (int): 2**m
        p (np.ndarray): irreducible polynomial coefficients
    Returns:
        alpha_to: array of j=alpha**i
        index_of: [j=alpha**i] = i
    """
    size = 2**m-1

    alpha_to = np.zeros(size+1, dtype=int)
    index_of = np.zeros(size+1, dtype=int)
    mask = 1
    alpha_to[m] = 0
    for i in range(m):
        alpha_to[i] = mask
        index_of[alpha_to[i]] = i
        if not p[i] == 0:
            alpha_to[m] ^= mask
        mask <<= 1
    index_of[alpha_to[m]] = m
    mask >>= 1
    for i in range(m+1, size):
        if alpha_to[i-1] >= mask:
            alpha_to[i] = alpha_to[m] ^ ( (alpha_to[i-1] ^ mask)<<1 )
        else:
            alpha_to[i] = alpha_to[i-1]<<1
        index_of[alpha_to[i]] = i
    # index_of[0] = -1
    ## Clean up
    alpha_to = alpha_to[1:]
    index_of = index_of[1:]
    return alpha_to, index_of

def main():
    alp, idx = GF2(4)
    print( "alp   idx")
    for a, i in zip(alp, idx):
        print(f"{a:03}   {i:03}")

if __name__ == "__main__":
    main()
