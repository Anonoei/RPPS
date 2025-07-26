import numpy as np

def rrc(taps, sps, beta=0.35):
    """
    taps: number of taps for FIR
    sps: samples per symbol (sample width for 0 ISI)
    beta: roll off factor [>=0,<=1]
    """
    t = np.arange(-taps//2,0)/sps # only generate half the pulse

    sin_arg = np.sin(np.pi*t*(1-beta))
    cos_arg = 4*beta*t*np.cos(np.pi*t*(1+beta))
    denom = np.pi*t*(1-(4*beta*t)**2)
    h = (sin_arg+cos_arg)/denom

    h = np.concat((h, [1+beta*(4/np.pi -1)], h[::-1]))
    return h


def pulse_rc(taps, sps, beta=0.35):
    """Raised Cosine shaping
    taps: number of taps for FIR
    sps: samples per symbol (sample width for 0 ISI)
    beta: roll-off factor [>=0,<=1]
    """
    t = np.arange(-taps//2,1)/sps # only generate half the pulse

    numer = np.cos((np.pi*beta*t))
    denom = 1 - (2*beta*t)**2
    h = np.sinc(t) * (numer/denom)

    h = np.concat((h[:-1], h[::-1]))
    return h

def pulse_rect(taps, sps, beta):
    return np.ones(taps)
