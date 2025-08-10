import numpy as np

def _delta(taps, amp=1, shift=0):
    f = np.zeros(taps)
    f[taps//2+shift] = amp
    return f

def _first_dif(taps):
    f = np.zeros(taps)
    f[taps//2] = 1
    f[taps//2+1] = -1
    return f

def _exp(taps):
    f = _delta(taps, amp=1)
    for i in range(taps//2+1, taps, 1):
        f[i] = f[i-1]*0.75
    return f
def _sqr(taps):
    f = np.zeros(taps)
    f[taps//2:taps//2+(taps//4)-1] = 1
    return f
def _sinc(taps):
    t = np.arange(-taps//2,taps//2)/(taps/6)
    f = np.sinc(t)
    f = f * np.hanning(taps)
    return f

def lowpass(taps, name="sinc"):
    if name == "sinc":
        f = _sinc(taps)
    elif name == "exp":
        f = _exp(taps)
    elif name == "sqr":
        f = _sqr(taps)
    f = f / np.sum(f) # normalize values to 1
    return f

def highpass(taps, name="sinc"):
    f = lowpass(taps, name)
    if name == "sqr":
        d = _delta(taps, shift=taps//8-1)
    else:
        d = _delta(taps)
    f = d - f
    return f

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
