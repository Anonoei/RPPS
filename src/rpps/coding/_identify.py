from . import MAP

from .coding import Coding


def name(t: str, n: str, *arg, **kwargs) -> Coding:
    return MAP[t][n](*arg, **kwargs)
