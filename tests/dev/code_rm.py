import numpy as np

def rm_2_4():
    r = 2
    m = 4
    k = 11 # 4/2 + 4/1 + 4/0 = 6 + 4 + 1 = 11
    n = 2**m # 16

    # C(x) = (px(Z) % 2)ZE{0,1}**m
    msg = np.array([1, 1,0,1,0,0,1,0,1,0,1], dtype=int)
    msg_pad = np.zeros(n)
    msg_pad[:len(msg)] = msg
    # px = 1 + Z1 + Z3 + Z1*Z3 + Z3*Z4

    def p(z):
        c = (1+z[0]+z[2]+z[0]*z[2]+z[1]*z[2]+z[2]*z[3]) % 2
        return c

    p_mat = np.array([
        [0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1],
        [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1],
        [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1],
        [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1],
    ])
    enc = np.zeros(n)
    for i, m in enumerate(p_mat):
        enc[i] = p(m)
    print(enc)

def rm_1_3():
    m = 3
    n = 2**m
    X = np.array([
        [0,0,0], [0,0,1], [0,1,0], [0,1,1],
        [1,0,0], [1,0,1], [1,1,0], [1,1,1]
    ])
    v = np.array([
        [1,1,1,1,1,1,1,1],
        [1,0,1,0,1,0,1,0],
        [1,1,0,0,1,1,0,0],
        [1,1,1,1,0,0,0,0],
    ])
    G = v

def rm_2_3():
    m = 3
    n = 2**m
    v = np.array([[ # {v0,v1,v2,v3,v1 ^ v2,v1 ^ v3,v2 ^ v3}
        [1,1,1,1,1,1,1,1],
        [1,0,1,0,1,0,1,0],
        [1,1,0,0,1,1,0,0],
        [1,1,1,1,0,0,0,0],
        [1,0,0,0,1,0,0,0],
        [1,0,1,0,0,0,0,0],
        [1,1,0,0,0,0,0,0]
    ]])
    G = v

def generate_RM_codes(m, order):
    """
    Generate RM codes of order 'order' and length 2^m
    """
    # Initialize the codewords list
    codewords = []

    # Generate all possible binary vectors of length 2^m
    for i in range(2**m):
        # Convert integer to binary vector
        binary = [int(x) for x in format(i, f'0{m}b')]
        # Pad with zeros if necessary
        if len(binary) < m:
            binary = binary + [0]*(m - len(binary))
        # Create the polynomial evaluation
        poly_eval = []
        for j in range(m):
            # Compute the polynomial evaluation
            term = 1
            for k in range(j+1):
                term = term * binary[k]
            poly_eval.append(term)
        # Add to codewords list
        codewords.append(poly_eval)

    return codewords

def rm_32_6():
    pass


if __name__ == "__main__":
    # rm_2_4()
    m=3
    order = 1
    G = generate_RM_codes(m, order)
    print(G)
