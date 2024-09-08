"""Generic viz wrappers"""
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def show():
    """Show plots"""
    plt.show()


def figure() -> Figure:
    """Create a new figure"""
    return plt.figure()

def subplot(*args, **kwargs) -> Axes:
    """Create/get a subplot"""
    return plt.subplot(*args, **kwargs)

def subplots(*args, **kwargs):
    """Create subplots"""
    return plt.subplots(*args, **kwargs)

def ion():
    """Enable interactive mode"""
    plt.ion()

def ioff():
    """Disable interactive mode"""
    plt.ioff()

def cla():
    """Clear axes"""
    plt.cla()

def pause(interval: float):
    """Pause"""
    plt.pause(interval)
