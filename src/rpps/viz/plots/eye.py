import numpy as np
import matplotlib.pyplot as plt

def get_eye(syms, sps):
    num_snip = len(syms)//(sps)
    x_len = sps+1
    snip = syms / np.real(np.max(syms))
    mesh = np.zeros((num_snip, x_len), dtype=syms.dtype)
    for i in range(sps, len(snip), sps):
        sym_snip = snip[i-(sps//2):i+(sps//2)+1]
        mesh[i//sps] = sym_snip
    return mesh

def eyeI(ax, eye_list):
    if ax is None:
        fig, ax = plt.subplots()
    x_len = eye_list.shape[1]
    x = np.arange(0, x_len)/(x_len-1) - 0.5
    for syms in eye_list:
        ax.plot(x, syms.real, "-", c="r", alpha=0.2)

    ax.grid(True)
    ax.set_xlabel("Time [Fm]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(1.2, -1.2)

def eyeQ(ax, eye_list):
    if ax is None:
        fig, ax = plt.subplots()
    x_len = eye_list.shape[1]
    x = np.arange(0, x_len)/(x_len-1) - 0.5
    for syms in eye_list:
        ax.plot(x, syms.imag, "-", c="b", alpha=0.2)

    ax.grid(True)
    ax.set_xlabel("Time [Fm]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(1.2, -1.2)

def eye(ax, syms, sps):
    if ax is None:
        fig, ax = plt.subplots(2,1)

    eye_list = get_eye(syms, sps)

    eyeI(ax[0], eye_list)
    eyeQ(ax[1], eye_list)

    ax[0].set_title(f"Eye Diagram")
    ax[0].set_title("Time Domain", loc="left")
    ax[0].set_title(f"{len(syms)/sps} symbols", loc="right")
    return
