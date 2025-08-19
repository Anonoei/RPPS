"""RPPS parent classes"""
import numpy as np


class Pipe:
    """Parent class for RPPS implementations"""
    def __str__(self) -> str:
        return "Pipe"
    def encode(self, data):
        """Encode data"""
        return NotImplemented
    def decode(self, data):
        """Decode data"""
        return NotImplemented
    def __rmul__(self, data):
        """Encode data"""
        return NotImplemented
    def __rtruediv__(self, data):
        """Decode data"""
        return NotImplemented
    def __radd__(self, other) -> list:
        """Get encoding minimum size"""
        if isinstance(other, list):
            return other
        data = []
        data = other.__radd__(data)
        return data
    def __rsub__(self, other) -> list:
        """Get decoding minimum size"""
        if isinstance(other, list):
            return other
        data = []
        data = other.__rsub__(data)
        return data
