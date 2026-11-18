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

def axes(ax=None, **kw) -> Axes:
    """ax, or the axes of a new figure when it is None (kw: for plt.subplots)"""
    return plt.subplots(**kw)[1] if ax is None else ax

def line(ax, x, y, title=None, left=None, right=None, xlabel=None, ylabel=None, grid=None, **style):
    """Plot y against x and return the line. ax: axes (None: a new figure), or the line an earlier call returned, which is
    updated in place (an animation). title/left/right: the centre, left and right titles; grid: its alpha"""
    from matplotlib.lines import Line2D
    if isinstance(ax, Line2D):
        ax.set_data(x, y)
        return ax
    ax = axes(ax)
    ln, = ax.plot(x, y, **style)
    for loc, text in (("center", title), ("left", left), ("right", right)):
        if text is not None:
            ax.set_title(text, loc=loc)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if grid is not None:
        ax.grid(True, alpha=grid)
    ax.set_xlim(x[0], x[-1])
    return ln
