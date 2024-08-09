import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def show():
    plt.show()


def figure() -> Figure:
    return plt.figure()

def subplot(*args, **kwargs) -> Axes:
    return plt.subplot(*args, **kwargs)

def subplots(*args, **kwargs):
    return plt.subplots(*args, **kwargs)

def ion():
    plt.ion()

def ioff():
    plt.ioff()

def cla():
    plt.cla()

def pause(interval: float):
    plt.pause(interval)
