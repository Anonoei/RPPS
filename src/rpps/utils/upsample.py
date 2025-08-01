import numpy as np

def fft(samples, ratio):
    """
    samples: complex IQ samples
    ratio: ratio to upsample by
    """
    frq = np.fft.fft(samples)
    up = ratio*np.hstack([
        frq[:len(frq)//2],
        frq[len(frq)//2]/ratio,
        np.zeros((len(frq)*ratio)-len(frq)-1),
        frq[len(frq)//2]/ratio,
        frq[-(len(frq)//2-1):]
    ])

    up = np.fft.ifft(up)
    return up
