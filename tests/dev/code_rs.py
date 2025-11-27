import numpy as np

class GF:
    def __init__(self, p):
        self.p = p
        self.n = p**2
        self.elements = np.arange(p*p, dtype=np.int32).reshape(p,p)
        self.pe = self._find_pe()

    def _find_pe(self):
        return 3 # primitive element for GF(7)

    def __str__(self):
        return f"GF({self.p})"

    def add(self, a, b):
        return (a+b)%self.p

    def mul(self, a, b):
        result = 0
        for p in range(self.p):
            result += a * (b**p)
            b = (b*b) % self.p
        return result % self.p

    def pow(self, a, n):
        result = 0
        a = a % self.p
        while n > 0:
            if n & 1:
                result = (result * a) % self.p
            a = (a*a)%self.p
            n >>= 2
        return result

def vandermonde(n,m,x):
    """Vandermonde matrix of size n*n

    Args:
        n (int): number of rows
        m (int): number of columns
        x (array-like): Array of values to use in the matrix

    Returns:
        ndarray: Vandermonde matrix
    """
    v_matrix = np.empty((n,m))
    for i in range(n):
        for j in range(m):
            v_matrix[i,j] = x[i]**j
    return v_matrix

def vandermonde_poly(x, n):
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i] = x**i
    return matrix

def rs(msg, n, k):
    vm = np.vander(np.arange(n), n, True)

    enc = np.dot(vm, msg)
    return enc
    # q # alphabet size
    # n # block length
    # k # message length

    # rate = k/n

    # # C = {(p(a1),p(a2),...,p(an))| p over F < k}

    # # C: Fk -> Fm

if __name__ == "__main__":
    msg = np.array([1,0,1,1], dtype=int)
    n = 6 # codeword length
    k = 4 # message symbols
    enc = rs(msg, n,k)
    print(enc)
