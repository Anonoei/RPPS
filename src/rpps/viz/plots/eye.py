import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_eye(syms, sps):
    N = len(syms)
    num_snip = N//sps
    out = syms
    norm = np.max([out.real, out.imag])
    if not len(out) % sps == 0:
        out = np.zeros(N+(sps-(N%sps)), dtype=syms.dtype)
        num_snip += 1
        out[:N] = syms
        syms = out
    x_len = sps+1
    snip = out / norm
    mesh = np.zeros((num_snip, x_len), dtype=out.dtype)
    for i in range(sps, len(snip), sps):
        sym_snip = snip[i-(sps//2):i+(sps//2)+1]
        mesh[i//sps] = sym_snip
    return mesh

def I(ax, syms, sps):
    if ax is None:
        fig, ax = plt.subplots(1)
    eye_list = get_eye(syms, sps)
    x_len = eye_list.shape[1]
    alpha = (1/eye_list.shape[0]) * 30
    x = np.arange(0, x_len)/(x_len-1) - 0.5
    if isinstance(ax, list):
        for idx, syms in enumerate(eye_list):
            ax[idx].set_data(x, syms.real)
        lines = ax
    else:
        lines = []
        for syms in eye_list:
            line, = ax.plot(x, syms.real, "-", c="r", alpha=alpha)
            lines.append(line)
        ax.grid(True)
        ax.set_xlabel("Time [Fm]")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(1.2, -1.2)
    return lines

def Q(ax, syms, sps):
    if ax is None:
        fig, ax = plt.subplots(1)
    eye_list = get_eye(syms, sps)
    x_len = eye_list.shape[1]
    alpha = (1/eye_list.shape[0]) * 30
    x = np.arange(0, x_len)/(x_len-1) - 0.5
    if isinstance(ax, list):
        for idx, syms in enumerate(eye_list):
            ax[idx].set_data(x, syms.real)
        lines = ax
    else:
        lines = []
        for syms in eye_list:
            line, = ax.plot(x, syms.imag, "-", c="b", alpha=alpha)
            lines.append(line)
        ax.grid(True)
        ax.set_xlabel("Time [Fm]")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(1.2, -1.2)
    return lines

def IQ(ax, syms, sps):
    if ax is None:
        fig, ax = plt.subplots(1)
    sps = int(round(sps, 0))

    eye_list = get_eye(syms, sps)
    x_len = eye_list.shape[1]
    alpha = ((1/eye_list.shape[0])*2)+0.1
    print(f"Eye: {eye_list.shape}, alpha: {alpha}")
    # exit()
    x = np.arange(0, x_len)/(x_len-1) - 0.5
    if isinstance(ax, tuple):
        for idx, syms in enumerate(eye_list):
            ax[0][idx].set_data(x, syms.real)
            ax[1][idx].set_data(x, syms.imag)
        linesI, linesQ = ax
    else:
        linesI = []
        linesQ = []
        for syms in eye_list:
            lineI, = ax.plot(x, syms.real, "-", c="r", alpha=alpha)
            linesI.append(lineI)
            lineQ, = ax.plot(x, syms.imag, "-", c="b", alpha=alpha)
            linesQ.append(lineQ)
        ax.grid(True)
        ax.set_xlabel("Time [Fm]")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(1.2, -1.2)
        ax.set_title("Eye")
        ax.set_title("Time", loc="left")
        ax.set_title(f"{eye_list.shape[0]} N/{sps} sps", loc="right")
    return linesI, linesQ
