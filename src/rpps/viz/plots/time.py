"""Time domain viz helpers"""
import numpy as np
import matplotlib.pyplot as plt

def I(ax, samps):
    if ax is None:
        fig, ax = plt.subplots()

    line_i, = ax.plot(samps.real, ".-", c="r", label="I")
    return line_i

def Q(ax, samps):
    if ax is None:
        fig, ax = plt.subplots()

    line_q, = ax.plot(samps.imag, ".-", c="b", label="Q")
    return line_q

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

    line_i = I(ax, samps)
    line_q = Q(ax, samps)

    ax.grid(True)
    ax.set_title("Time IQ")
    ax.set_title("Time Domain", loc="left")
    ax.set_title(f"{len(samps)} samples", loc="right")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    return line_i, line_q

def IQ3d(samps, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    I = np.real(samps)
    Q = np.imag(samps)

    T = np.arange(len(I))
    E = np.zeros(len(I))

    ip = np.array([T,I,E])
    qp = np.array([T,E,Q])

    ax.plot(ip[0], ip[1], ip[2])
    ax.plot(qp[0], qp[1], qp[2])
    ax.scatter(ip[0], ip[1], qp[2])

    ax.set_xlabel("Time")
    ax.set_ylabel("I")
    ax.set_zlabel("Q")
    ax.set_ylim(1.2, -1.2)
    ax.set_zlim(1.2, -1.2)

    plt.grid(True)
    plt.legend()
    return ax
