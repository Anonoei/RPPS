import numpy as np

# class GF:
#     def __init__(self, p):
#         self.p = p
#         self.n = p**2
#         self.elements = np.arange(p*p, dtype=np.int32).reshape(p,p)
#         self.pe = self._find_pe()

#     def _find_pe(self):
#         return 3 # primitive element for GF(7)

#     def __str__(self):
#         return f"GF({self.p})"

#     def add(self, a, b):
#         return (a+b)%self.p

#     def mul(self, a, b):
#         result = 0
#         for p in range(self.p):
#             result += a * (b**p)
#             b = (b*b) % self.p
#         return result % self.p

#     def pow(self, a, n):
#         result = 0
#         a = a % self.p
#         while n > 0:
#             if n & 1:
#                 result = (result * a) % self.p
#             a = (a*a)%self.p
#             n >>= 2
#         return result

def bch_7_4():
    n,k,m = 7,4,3

    G = np.array([1,1,1,0,1,1,0], dtype=int)

    data = np.array([1,0,1,0])
    for i in range(m):
        data = np.convolve(data, G, mode="full")
        data = data[n-k+i:]

    print(data)

def get_prim_poly(m):
    """ Generate primitive polynomials of degree m
    Args
        m (int): polynomial degree
    """
    n = 2**m
    polys = []
    for i in range(n):
        poly = np.poly1d([1 if j == i else 0 for j in range(m+1)])
        if np.allclose(np.roots(poly), 0):
            polys.append(poly)
    return polys

def gen_bch(m):
    p = np.poly1d([1] + [0] * m + [1])
    gen = np.poly1d([1])
    for i in range(m):
        gen = np.polymul(p, gen)
        # gen = gen[:-i-1]
    return gen

def bch_t(n,m):
    return np.floor((n-1)/m)

def bch_encode(m, p):
    """
    n: total bits
    k: message length
    t: error correction capability

    m=3, t=1 g(x) = x^2 + x + 1
    m=4, t=1 g(x) = x^4 + x + 1
    """
    # generator (g(x)) of degree n-k

def main():
    m = 3
    n = 2**m
    t = bch_t(n,m)
    p = gen_bch(m)
    print(f"t: {t}")
    print("poly")
    print(f"order: {p.order}, coef: {p.c}")
    print(p)

if __name__ == "__main__":
    main()
