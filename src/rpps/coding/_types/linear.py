import numpy as np

class linear:
    def __init__(self, generator, check):
        input(np.matmul(generator, check))
        self.generator = generator
        self.check = np.transpose(check)

        self.num = generator.shape[0]
        self.den = generator.shape[1]


    def encode(self, bits: np.ndarray):
        encoded = np.matmul(bits.astype(int), self.generator)
        return encoded % 2

    def decode(self, bits: np.ndarray):
        decoded = np.matmul(bits.astype(int), self.check)
        parity_bits = decoded % 2
        if sum(parity_bits) == 0:
            return bits[0:self.num]
        print(f"{parity_bits}")
        raise NotImplementedError("Bit error!")

def generate(length: int, data_bits: int, p: np.ndarray):
    # length: n
    # data_bits: k
    def make_id(size: int):
        idm = np.zeros((size, size))
        idm[np.arange(size),np.arange(size)] = 1
        return idm

    redundant = length - data_bits

    g_id = make_id(data_bits)
    c_id = make_id(redundant)

    print(f"n: {length}")
    print(f"k: {data_bits}")
    print(f"r: {redundant}")
    print()
    print(f"generator id:\n{g_id.astype(int)}")
    print(f"parity check id:\n{c_id.astype(int)}")

    if p.shape[0] == data_bits and p.shape[1] == redundant:
        pass
    elif p.shape[0] == redundant and p.shape[0] == data_bits:
        p = np.transpose(p)
    else:
        print(f"p must be shape ({data_bits},{redundant}) or ({redundant},{data_bits})")
        exit()

    g_matrix = np.zeros((data_bits,length))
    for i in range(data_bits):
        g_matrix[i,0:data_bits] = g_id[i]
        g_matrix[i,data_bits:] = p[i]
    print(f"parity matrix:\n{p.astype(int)}")
    print(f"generator:\n{g_matrix.astype(int)}")

    p = np.transpose(p)
    c_matrix = np.zeros((redundant,length))
    for i in range(redundant):
        c_matrix[i,0:data_bits] = p[i]
        c_matrix[i,data_bits:] = c_id[i]
    print(f"parity matrix:\n{p.astype(int)}")
    print(f"parity check:\n{c_matrix.astype(int)}")
