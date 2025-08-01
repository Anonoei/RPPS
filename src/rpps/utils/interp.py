import numpy as np

def linear(x, xp, yp):
    return np.interp(x, xp, yp)
