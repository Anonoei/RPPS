"""RPPS parent classes"""

class Pipe:
    """Parent class for RPPS implementations"""
    def __matmul__(self, other):
        raise NotImplementedError()

    def __rmatmul__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    def __div__(self, other):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()

    def __sub__(self, other):
        raise NotImplementedError()
