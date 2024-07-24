
from .coding import Coding

class Block(Coding):
    def __init__(self, num, den, length):
        super().__init__(num, den)
        self.length = length
