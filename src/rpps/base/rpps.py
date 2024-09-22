"""RPPS parent classes"""

class Pipe:
    """Parent class for RPPS implementations"""

    # --- Arithmetic --- #
    def __matmul__(self, other):  # self @ other
        raise NotImplementedError()

    def __rmatmul__(self, other):  # other @ self
        raise NotImplementedError()

    def __mul__(self, other):  # self * other
        raise NotImplementedError()

    def __div__(self, other):  # self // other
        raise NotImplementedError()

    def __truediv__(self, other):  # self / other
        raise NotImplementedError()

    def __add__(self, other):  # self + other
        raise NotImplementedError()

    def __sub__(self, other):  # self - other
        raise NotImplementedError()

    def __mod__(self, other):  # self % other
        raise NotImplementedError()

    def __pow__(self, other):  # self ** other
        raise NotImplementedError()

    # --- Comparison --- #

    def __eq__(self, other):  # self == other
        raise NotImplementedError()

    def __ne__(self, other):  # self != other
        raise NotImplementedError()

    def __lt__(self, other):  # self < other
        raise NotImplementedError()

    def __le__(self, other):  # self <= other
        raise NotImplementedError()

    def __gt__(self, other):  # self > other
        raise NotImplementedError()

    def __ge__(self, other):  # self >= other
        raise NotImplementedError()
