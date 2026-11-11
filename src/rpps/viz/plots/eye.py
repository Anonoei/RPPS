import numpy as np
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
    half = sps//2
    x_len = 2*half+1 # Odd sps has no sample on the symbol boundary, so it is sps wide, even sps+1
    snip = out / norm
    mesh = np.zeros((num_snip, x_len), dtype=out.dtype)
    for i in range(sps, len(snip), sps):
        sym_snip = snip[i-half:i+half+1]
        if len(sym_snip) == x_len: # Skip a snippet cut short by the end
            mesh[i//sps] = sym_snip
    return mesh

def sync_phase(st, n_in):
    """Where a timing sync samples within the block it just consumed, in samples (None: not running yet)

    st: the sync's state after consuming the block (rp.time.MM or rp.sync.time.Mat), n_in: that block's length.
    The sync's buffer ends at the end of the block, so the next sampling instant (buffer index) is
    that many samples into the block, and the rest of its lattice follows every sps.
    """
    buf = getattr(st, "buffer", None)
    if buf is None:
        return None
    nxt = st.pos + st.est if hasattr(st, "pos") else st.t_next # MM / Mat
    return nxt - (len(buf) - n_in)

def get_eye_at(syms, sps, phase, width=1.0, derotate=None):
    """Eye traces `width` symbols wide centred on the sampling instants phase + k*sps (fractional sps and phase ok)

    The sampling instant is the middle column of every trace.
    derotate: modulation order (2: BPSK, 4: QPSK...). Each trace is rotated by the carrier phase its own centre sample
        shows (angle(c**order)/order), so a carrier offset can't move the signal between I and Q.
    """
    isps = max(2, int(round(sps*width)))
    points = 2*(isps//2) + 1 # Odd, so there is a middle column
    t = (np.arange(points)/(points-1) - 0.5) * sps * width
    half = sps*width/2
    first = int(np.ceil((half - phase)/sps)) # First centre whose whole trace is inside
    last = int(np.floor((len(syms) - 1 - half - phase)/sps))
    centres = phase + sps*np.arange(first, last + 1)
    if len(centres) == 0:
        return np.zeros((0, points), dtype=syms.dtype)
    pos = centres[:, None] + t[None, :]
    n = np.arange(len(syms))
    mesh = np.interp(pos, n, syms.real) + 1j*np.interp(pos, n, syms.imag)
    if derotate:
        c = mesh[:, points//2]
        mesh = mesh * np.exp(-1j*np.angle(c**derotate)/derotate)[:, None]
    return mesh/np.max([mesh.real, mesh.imag])

def lock_phase(mesh, order):
    """Rotate the traces by the carrier frequency and phase found from the order'th power of their centre samples.
    Unlike derotate (each trace by its own centre, which pinches every trace to the axis at the centre) this keeps the
    spread of the symbols, but needs the carrier to turn only slowly (a residual frequency offset, not noise-like)"""
    c = mesh[:, mesh.shape[1]//2]
    k = np.arange(len(c))
    z = c**order # The data is gone, a tone is left: its frequency (peak of a padded FFT) and phase are the carrier's
    n = 8*len(z)
    f = np.fft.fftfreq(n)[np.argmax(np.abs(np.fft.fft(z, n)))] # Cycles per symbol, of the order'th power
    a = np.angle(np.sum(z*np.exp(-2j*np.pi*f*k)))
    return mesh * np.exp(-1j*(a + 2*np.pi*f*k)/order)[:, None]

def get_eye_mf(syms, sps, phase, beta, span=8, width=1.0, derotate=None, lock=None, max_traces=300):
    """Like get_eye_at, but through the RRC matched filter the sync uses (rp.sync.time.Mat), evaluated at each
    point of the trace. This is what the sync sees, unlike the raw samples, which are the unmatched pulses plus noise"""
    from ...sync.time.mat import matrix_mf, centres_in
    isps = max(2, int(round(sps*width)))
    points = max(17, 2*(isps//2) + 1) # Odd, and enough points for a smooth trace between samples
    S = span*sps
    m = np.ceil(sps*width/2) + 1 # The trace reaches this far past its centre
    t0 = phase + np.ceil((S + 5 + m - phase)/sps)*sps
    centres = centres_in(len(syms) - m, t0, sps, span)[-max_traces:]
    if len(centres) == 0:
        return np.zeros((0, points), dtype=syms.dtype)
    t = (np.arange(points)/(points-1) - 0.5) * sps * width
    mesh = np.stack([matrix_mf(syms, centres + d, sps, beta, span) for d in t], axis=1)
    if lock and len(centres) > 2:
        mesh = lock_phase(mesh, lock)
    if derotate:
        c = mesh[:, points//2]
        mesh = mesh * np.exp(-1j*np.angle(c**derotate)/derotate)[:, None]
    return mesh/np.max([mesh.real, mesh.imag])

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

def IQ(ax, syms, sps, phase=None, width=1.0, derotate=None, beta=None, lock=None):
    """phase: sampling instant [samples] in syms (see sync_phase), centres the eye on it. None: fixed grid
    width: how many symbols (sps) wide the eye is, derotate: modulation order, see get_eye_at (both need phase)
    lock: modulation order, see lock_phase (with beta only)
    beta: RRC rolloff, draws the eye through the matched filter (see get_eye_mf) instead of the raw samples"""
    if ax is None:
        fig, ax = plt.subplots(1)
    if phase is None:
        sps = int(round(sps, 0))
        eye_list = get_eye(syms, sps)
    elif beta is not None:
        eye_list = get_eye_mf(syms, sps, phase, beta, width=width, derotate=derotate, lock=lock)
    else:
        eye_list = get_eye_at(syms, sps, phase, width, derotate)
    x_len = eye_list.shape[1]
    alpha = ((1/eye_list.shape[0])*2)+0.1
    print(f"Eye: {eye_list.shape}, alpha: {alpha}")
    # exit()
    w = width if phase is not None else 1.0
    x = (np.arange(0, x_len)/(x_len-1) - 0.5)*w
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
        ax.set_xlim(-w/2, w/2)
        ax.set_ylim(1.2, -1.2)
        ax.set_title("Eye")
        ax.set_title("Time", loc="left")
        ax.set_title(f"{eye_list.shape[0]} N/{sps:.2f} sps", loc="right")
    return linesI, linesQ
