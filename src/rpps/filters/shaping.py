import numpy as np

def rrc(taps, sps, beta=0.35, gain=1.0):
    """Root-Raised Cosine shaping
    taps: number of taps for FIR
    sps: samples per symbol (sample width for 0 ISI)
    beta: roll off factor [0.35,0>=beta<=1]
    gain: filter gain [1]
    """
    t = np.arange(-taps//2,0)/sps # only generate half the pulse

    sin_arg = np.sin(np.pi*t*(1-beta))
    cos_arg = 4*beta*t*np.cos(np.pi*t*(1+beta))
    denom = np.pi*t*(1-(4*beta*t)**2)
    h = (sin_arg+cos_arg)/denom

    h = np.concat((h, [1+beta*(4/np.pi -1)], h[::-1]))
    return gain*h


def rc(taps, sps, beta=0.35, gain=1.0):
    """Raised Cosine shaping
    taps: number of taps for FIR
    sps: samples per symbol (sample width for 0 ISI)
    beta: roll off factor [0.35,0>=beta<=1]
    gain: filter gain [1]
    """
    t = np.arange(-taps//2+1,1)/sps # only generate half the pulse

    numer = np.cos((np.pi*beta*t))
    denom = 1 - (2*beta*t)**2
    h = np.sinc(t) * (numer/denom)

    h = np.concat((h[:-1], h[::-1]))
    return gain*h

def sinc(taps, sps, gain=1.0):
    t = np.arange(-taps//2,1)/sps
    h = np.sinc(t)
    h = np.concat((h[:-1], h[::-1]))
    return gain*h

def pulse_rect(taps, sps, beta):
    return np.ones(taps)
