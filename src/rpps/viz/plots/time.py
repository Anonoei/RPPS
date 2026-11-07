"""Time domain viz helpers"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def I(ax, samps):
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(ax, mpl.lines.Line2D):
        line = ax
        line.set_data(np.arange(0, len(samps)), samps.real)
    else:
        line, = ax.plot(samps.real, ".-", c="r", label="I")
        ax.set_xlim(0, len(samps)-1)
    return line

def Q(ax, samps):
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(ax, mpl.lines.Line2D):
        line = ax
        line.set_data(np.arange(0, len(samps)), samps.imag)
    else:
        line, = ax.plot(samps.imag, ".-", c="b", label="Q")
        ax.set_xlim(0, len(samps)-1)
    return line

def mag(ax, samps):
    if ax is None:
        fig, ax = plt.subplots()

    line_mag, = ax.plot(np.abs(samps), ".-", c="b", label="Mag")
    return line_mag

def pha(ax, samps):
    if ax is None:
        fig, ax = plt.subplots()

    line_pha, = ax.plot(np.angle(samps), ".-", c="b", label="Phase")
    return line_pha

def IQ(ax, samps):
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(ax, tuple):
        line_i, line_q = ax
        ax = line_i.axes
    else:
        line_i = ax
        line_q = ax
        ax.grid(True, alpha=0.5)
        ax.set_title("IQ")
        ax.set_title("Time", loc="left")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, len(samps)-1)
        ax.relim()
    ax.set_title(f"{len(samps)} N", loc="right")

    line_i = I(line_i, samps)
    line_q = Q(line_q, samps)
    if ax.get_xlim()[1] < len(samps)-1:
        ax.set_xlim(0, len(samps)-1)
    amax = np.max((samps.real, samps.imag, -samps.real, -samps.imag))
    if amax > ax.get_ylim()[1]:
        ax.set_xlim(-amax, amax)
        ax.set_ylim(-amax, amax)

    return (line_i, line_q)

def IQ3d(ax, samps):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    amp_max = np.max((-np.min(samps.real), np.max(samps.real), -np.min(samps.imag), np.max(samps.imag)))

    N = len(samps)

    r = np.repeat(amp_max, N)
    x = np.arange(N)

    if isinstance(ax, tuple):
        line_i, line_q, line_c = ax
        ax = line_i.axes
        line_i.set_data_3d(x, samps.real, -r)
        line_q.set_data_3d(x, r, samps.imag)
        line_c.set_data_3d(x, samps.real, samps.imag)
    else:
        line_i, = ax.plot(x, samps.real, -r, color="r", linestyle="solid", linewidth=1, alpha=0.5)
        line_q, = ax.plot(x, r, samps.imag, color="b", linestyle="solid", linewidth=1, alpha=0.5)
        line_c, = ax.plot(x, samps.real, samps.imag, color="g", linestyle="solid", marker=".", linewidth=3)

        ax.set_xlabel("Time")
        ax.set_ylabel("I")
        ax.set_zlabel("Q")
        ax.set_ylim(1.2, -1.2)
        ax.set_zlim(1.2, -1.2)
        ax.set_box_aspect((5,1,1))
        ax.grid(True, alpha=0.5)
    ax.set_xlim(0, len(samps))
    ax.set_ylim(-amp_max, amp_max)
    ax.set_zlim(-amp_max, amp_max)

    return (line_i, line_q, line_c)
