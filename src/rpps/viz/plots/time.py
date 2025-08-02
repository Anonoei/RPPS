"""Time domain viz helpers"""
import numpy as np
import matplotlib.pyplot as plt

def time(ax, samps):
    if ax is None:
        fig, ax = plt.subplots()

    line_c = ax.plot(samps.real_sams.imag, c="g", label="IQ")

    ax.grid(True)
    ax.set_title("Time")
    ax.set_title("Time Domain", loc="left")
    ax.set_title(f"{len(samps)} samples", loc="right")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    return line_c

def timeI(ax, samps):
    if ax is None:
        fig, ax = plt.subplots()

    line_i, = ax.plot(samps.real, ".-", c="r", label="I")
    return line_i

def timeQ(ax, samps):
    if ax is None:
        fig, ax = plt.subplots()

    line_q, = ax.plot(samps.imag, ".-", c="b", label="Q")
    return line_q

def timeIQ(ax, samps):
    if ax is None:
        fig, ax = plt.subplots()

    line_i = timeI(ax, samps)
    line_q = timeQ(ax, samps)

    ax.grid(True)
    ax.set_title("Time IQ")
    ax.set_title("Time Domain", loc="left")
    ax.set_title(f"{len(samps)} samples", loc="right")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    return line_i, line_q

def time(samps, ax=None):
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
    # ax.set_aspect("equal", adjustable="box")

    plt.grid(True)
    plt.legend()
    return fig, ax

def ot_complex(samps, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    I = np.real(samps)
    Q = np.imag(samps)

    T = np.arange(len(I))
    E = np.zeros(len(I))

    ip = np.array([T,I,E])
    qp = np.array([T,E,Q])

    ax.set_xlabel("Time")
    ax.set_ylabel("I")
    ax.set_zlabel("Q")
    ax.set_ylim(1.2, -1.2)
    ax.set_zlim(1.2, -1.2)
    plt.grid(True)
    plt.legend()

    ip = []
    qp = []
    xp = []
    ep = []

    for x,y,z in zip(T,I,Q):
        ip.append(y/2)
        qp.append(z/2)
        xp.append(x)
        ep.append(0)
        ax.plot(xp,ip,ep, color="red")
        ax.plot(xp,ep,qp, color="blue")
        ax.scatter(x,y,z, color="green")
        plt.pause(0.1)

    return fig, ax
