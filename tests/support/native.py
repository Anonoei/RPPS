"""The C++ kernels switched off and on, to compare them with the Python reference"""
from contextlib import contextmanager

from rpps import _accel

class Counting:
    """rpps._native, counting the kernel calls (a native run that silently fell back to Python would compare equal)"""
    def __init__(self, mod):
        self.mod, self.calls = mod, 0
    def __getattr__(self, name):
        self.calls += 1
        return getattr(self.mod, name)

@contextmanager
def kernels(on):
    """The kernels on or off inside the block: yields the counting spy when they are on and built, else None"""
    saved = _accel.native
    spy = Counting(saved) if on and saved is not None else None
    _accel.native = spy
    try:
        yield spy
    finally:
        _accel.native = saved

def both(fn):
    """fn() with the Python loops, then with the kernels: (python, native). Fails if no kernel ran"""
    with kernels(False):
        ref = fn()
    with kernels(True) as spy:
        got = fn()
        assert spy.calls > 0, "the native kernel never ran"
    return ref, got
