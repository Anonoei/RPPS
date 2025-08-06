import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def IQ3d(samps, Fs=1, ax=None):
    if ax is None:
        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(projection="3d")
    if isinstance(ax, mpl.figure.Figure):
        ax = fig.add_subplot(projection="3d")

    num_show = 128
    num_cur = 0
    ax.figure.subplots_adjust(top=1.1, bottom=-.1)

    I = np.real(samps)
    Q = np.imag(samps)

    y_max = np.max(I)
    z_max = np.max(Q)

    T = np.arange(len(I)) / Fs
    E = np.zeros(len(I))

    ip = np.array([T,I,E])
    qp = np.array([T,E,Q])

    ax.set_xlabel("Time")
    ax.set_ylabel("I")
    ax.set_zlabel("Q")
    ax.set_ylim(-y_max, y_max)
    ax.set_zlim(-z_max, z_max)
    ax.set_box_aspect((3, 1, 1))
    plt.grid(True, alpha=0.2)

    ip = np.zeros(num_show)
    qp = np.zeros(num_show)
    xp = np.zeros(num_show)
    ep = np.zeros(num_show)

    lineI, = ax.plot(T[0], I[0], 1, "-", c="r", label="I")
    lineQ, = ax.plot(T[0], -1, Q[0], "-", c="b", label="Q")
    lineX, = ax.plot(T[0], I[0], Q[0], ".-", c="g", linewidth=5, label="IQ")

    for x,y,z in zip(T,I,Q):
        if num_cur >= num_show:
            num_cur -= 1
            ip = np.roll(ip, -1)
            qp = np.roll(qp, -1)
            xp = np.roll(xp, -1)
        xp[num_cur] = x
        ip[num_cur] = y
        qp[num_cur] = z
        lineI.set_data_3d(xp[:num_cur],ip[:num_cur],ep[:num_cur]-z_max)
        lineQ.set_data_3d(xp[:num_cur],ep[:num_cur]+y_max,qp[:num_cur])
        lineX.set_data_3d(xp[:num_cur],ip[:num_cur],qp[:num_cur])
        if num_cur > 1:
            ax.set_xlim(xp[0], xp[num_cur])
        plt.draw()
        plt.pause(0.01)
        num_cur += 1
        # print(f"Done {(x/T[-1])*100:.2f}%")

    return ax
