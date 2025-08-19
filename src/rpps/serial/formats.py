"""File format helpers"""
from enum import Enum

import numpy as np

def from_i8(samples):
    """Returns samples as int8"""
    samps = np.array(samples, dtype=np.int8)
    return samps.astype(np.float32).view(dtype=np.complex64)

def from_i16(samples):
    """Returns sampkes as int16"""
    samps = np.array(samples, dtype=np.int16)
    return samps.astype(np.float32).view(dtype=np.complex64)

def from_f16(samples):
    """Returns samples as float16"""
    samps = np.array(samples, dtype=np.float16)
    return samps.astype(np.float32).view(dtype=np.complex64)

def from_f32(samples):
    """Returns samples as float32"""
    samps = np.array(samples, dtype=np.float32)
    return samps.view(dtype=np.complex64)

def from_f64(samples):
    """Returns samples as float32"""
    samps = np.array(samples, dtype=np.float64)
    return samps.view(dtype=np.complex128)

class Formats(Enum):
    """Generic data formats"""
    i8  = (2, from_i8)
    i16 = (4, from_i16)
    f16 = (4, from_f16)
    f32 = (8, from_f32)
    f64 = (16, from_f64)

    def bytes(self):
        """Returns how many bytes each complex value takes"""
        return self.value[0]

    def read(self, samples):
        """Convert samples to format"""
        return self.value[1](samples)
