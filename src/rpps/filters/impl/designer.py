import numpy as np

def designer(taps, cut_off,
        window=np.hamming,
        pass_zero=True, scale=True, fs=None):
    fs = 2 if fs is None else fs
    nyq = 0.5*fs

    cutoff = np.asarray(cut_off, dtype=np.float32)
    cutoff = cutoff.reshape((-1)) / float(nyq)

    pass_nyq = np.size(cutoff) % 2 == 0 == pass_zero
    cutoff = np.concat((np.zeros(int(pass_zero)), cutoff, np.ones(int(pass_nyq))))

    bands = np.reshape(cutoff, (-1, 2))
    alpha = 0.5 * (taps-1)
    m = np.arange(0, taps, dtype=cutoff.dtype) - alpha
    h = 0
    for j in range(bands.shape[0]):
        left, right = bands[j,0], bands[j,1]
        h = h + (right * np.sinc(right * m))
        h = h - (left * np.sinc(left * m))

    h = h * window(taps)

    if scale:
        left, right = bands[0, ...]
        if left == 0:
            scale_freq = 0.0
        elif right == 1:
            scale_freq = 1.0
        else:
            scale_freq = 0.5 * (left + right)
        c = np.cos(np.pi*m*scale_freq)
        s = np.sum(h*c)
        h /= s
    return h
