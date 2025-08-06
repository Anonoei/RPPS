import numpy as np

def _delta(taps):
    h = np.zeros(taps)
    h[taps//2+1] = 1
    return h

def _kern_square(taps, cut_off):
    h = np.zeros(taps)
    h[taps//2:taps//2+cut_off*2] = 1
    return h

def _kern_exp(taps, cut_off):
    t = np.arange(cut_off*2, 0, -1)
    h = np.zeros(taps)
    h[taps//2:taps//2+cut_off*2] = t**2
    return h

def low_pass(taps, cut_off):
    t = np.arange(-taps//2+1,0)/cut_off
    h = np.sinc(t)
    h = np.concat((h, [np.sinc(0)], h[::-1]))
    h = h * np.hamming(taps)
    h = h/cut_off
    return h

def high_pass(taps, cut_off):
    t = np.arange(-taps//2,0)/cut_off
    h = np.sinc(t)
    h = np.concat((h[:-1], h[::-1]))
    h = h * np.hamming(taps)
    h = h/cut_off
    h = _delta(taps) - h
    return h
