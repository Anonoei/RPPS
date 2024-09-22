import numpy as np

from ._code import _code

from ..blocker import block, unblock

class linear(_code):
    def __init__(self, generator, check):
        super().__init__(generator.shape[0], generator.shape[1])
        self.generator = generator
        self.check = np.transpose(check)
    def encode(self, bits: np.ndarray):
        blocks = block(bits, self.num)

        encoded = np.empty((len(blocks), self.den), dtype=int)
        for i, blk in enumerate(blocks):
            encoded[i] = np.matmul(blk.astype(int), self.generator)
        encoded = unblock(encoded) % 2
        return encoded.astype(bool)

    def decode(self, bits: np.ndarray):
        blocks = block(bits, self.den)

        decoded = np.empty((len(blocks), self.den - self.num), dtype=int)

        for i, blk in enumerate(blocks):
            decoded[i] = np.matmul(blk.astype(int), self.check)

        decoded = unblock(decoded)
        parity_bits = decoded % 2
        if sum(parity_bits) == 0:
            return unblock(blocks[:,0:self.num])
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
