"""Time domain viz helpers"""
import numpy as np
import matplotlib.pyplot as plt

from . import Meta


def phasor(symbols, ax=None):
    """Plot phasor diagram"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.scatter(np.real(symbols), np.imag(symbols), s=5, c="b")

    ax.grid(True)
    ax.set_title(f"Phasor Diagram")
    ax.set_title("Time Domain", loc="left")
    ax.set_title(f"{len(symbols)} symbols", loc="right")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_xlim(1.2, -1.2)
    ax.set_ylim(1.2, -1.2)
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def quadrature(symbols, ax=None):
    """Plot I and Q"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    real = np.real(symbols)
    imag = np.imag(symbols)

    base_i = np.cos(np.arange(len(symbols))) * 0.2
    base_q = np.sin(np.arange(len(symbols))) * 0.2

    ax.plot(base_i + base_q, "-", label="Carrier")
    ax.plot(real + imag, ".-", label="Constructed")

    plt.grid(True)
    ax.set_title("Quadrature Signal")
    ax.set_title("Time Domain", loc="left")
    ax.set_title(f"{len(symbols)} symbols", loc="right")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-5, 5)
    plt.legend()
    return fig, ax

def complex(symbols, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    I = np.real(symbols)
    Q = np.imag(symbols)

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

def ot_complex(symbols, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    I = np.real(symbols)
    Q = np.imag(symbols)

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
