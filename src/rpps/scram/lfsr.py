import numpy as np

class LFSR:
    __slots__ = ("_seed", "lfsr")
    def __init__(self, seed):
        self._seed = seed
        self.lfsr = seed

    def __str__(self):
        return f"{type(self).__name__}"

    def reset(self):
        self.lfsr = self._seed

    def get_bit(self):
        ...

class fibonacci(LFSR):
    __slots__ = ("taps")
    name = "Fibonacci"

    def __init__(self, seed, taps):
        super().__init__(seed)
        self.taps = taps

    def __str__(self):
        return f"{super().__str__()}:{self.taps}"

    def get_bit(self):
        # Xor taps sequentially
        new_bit = np.bitwise_xor.reduce(self.lfsr[self.taps - 1]) & 1
        # shift one place to the right, [1,0,1,0,1,#] >> 1 = [#,1,0,1,0,1]
        self.lfsr[1:] = self.lfsr[:-1]  # clock LFSR
        self.lfsr[0] = new_bit  # set left-most bit (input bit)
        # return right-most bit (output bit)
        return self.lfsr[-1]

class Galois(LFSR):
    __slots__ = ("_poly", "toggle")
    name = "Galois"

    def __init__(self, seed, toggle):
        super().__init__(seed)
        self._poly = toggle
        self.toggle = np.zeros((len(seed),), bool)
        self.toggle[len(seed) - toggle] = True

    def __str__(self):
        return f"{super().__str__()}:{self._poly.astype(int)}"
# 1110001001110000

class galois_left(Galois):
    name = "Galois Left"
    def get_bit(self):
        lsb = self.lfsr[0]
        self.lfsr[1:] = self.lfsr[:-1]  # clock LFSR
        self.lfsr[0] = 0

        if lsb:
            self.lfsr = np.bitwise_xor(self.lfsr, self.toggle)
        return lsb


class galois_right(Galois):
    name = "Galois Right"
    def get_bit(self):
        msb = self.lfsr[-1]
        self.lfsr[:-1] = self.lfsr[1:]  # clock LFSR
        self.lfsr[-1] = 0

        if msb:
            self.lfsr = np.bitwise_xor(self.lfsr, self.toggle)
        return msb
