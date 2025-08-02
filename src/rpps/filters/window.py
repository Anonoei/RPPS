"""
Window functions

Generalized properties
- (NB) Noise bandwidth,  / Spectral leakage
- (SL) Scalloping loss / Amplitude accuracy
- (FR) Frequency resolution
Ratings, (Best, Good, Fair, Poor) [5,3,1,0]

Window    | NB | SL | FR |
--------- | -- | -- | -- |
Blackman  |  5 |  3 |  1 | *NB
Flattop   |  3 |  5 |  0 | *SL
Gaussian  |  1 |  3 |  1 |
Rectangle |  0 |  0 |  5 | *FR
Hanning   |  3 |  1 |  3 |
Hamming   |  1 |  1 |  3 |
Kaiser    |  3 |  3 |  1 |
"""
import numpy as np

def rect(n: int): # 1st-order B-spline / 0th power-of-sine
    return np.ones(n)

# --- B-spline windows --- #
# def _bspline(N, k, L=None):
#     # w[n] = 1 - | (n-N/2)/(L/2)|, 0 <= n <= N
#     if L is None:
#         L = N
#     n = np.arange(0, N)
#     w = 1 - abs( (n-N/2)/(L/2) )
#     return w

def bartlett(N, Lp=0): # 2nd-order B-spline / triangular window
    # 0 <= Lp <= 2
    L = N + Lp
    n = np.arange(0, N)
    w = 1 - abs( (n-N/2)/(L/2) )
    return w

def parzen(N): # 4th-order B-spline
    L = N+1
    n0 = np.arange(0, N) - N/2
    n1i = (0 <= np.abs(n0)) & (np.abs(n0) <= L/4)
    n2i = (L/4 < np.abs(n0)) & (np.abs(n0) <= L/2)

    n1 = n0[n1i]
    n2 = n0[n2i]
    w = np.zeros(N)
    w[n1i] = 1-6*( n1/(L/2) )**2 * (1-np.abs(n1)/(L/2) )
    w[n2i] = 2 * ( 1-np.abs(n2)/(L/2) )**3
    return w

# --- Polynomial windows --- #
def welch(N):
    n = np.arange(0, N)
    return 1 - ( (n-(N/2)) / (N/2) )**2

def welch_2f(N):
    n = np.arange(0, N)
    w1 = 1 + ( n - (N/2) )/(N/2)
    w2 = 1 - ( n - (N/2) )/(N/2)
    return w1*w2

# --- Raised-cosine windows --- #
def _rc(N, a0):
    n = np.arange(0, N)
    return a0 - (1 - a0) * np.cos( (2*np.pi*n)/N )

def _rc_0p(N, a0): # zero phase
    n = np.arange(0, N) - N/2
    return a0 + (1-a0)*np.cos( (2*np.pi*n)/N )

def hann(N): # eq to power-of-sine 2
    return _rc(N, 0.5)

def hamming(N): # Optimal hamming
    return _rc(N, 0.53836) # a1 = 0.46164

def hamming_orig(N): # Proposed hamming
    return _rc(N, 0.54) # a1 = 0.46

# --- Cosine-sum windows --- #
def _cs_2f(N, n, a0, a1):
    return a0 - a1*np.cos( (2*np.pi*n)/N )
def _cs_3f(N, n, a0, a1, a2):
    return _cs_2f(N,n,a0,a1) + a2*np.cos( (4*np.pi*n)/N )
def _cs_4f(N, n, a0, a1, a2, a3):
    return _cs_3f(N,n,a0,a1,a2) - a3*np.cos( (6*np.pi*n)/N )
def _cs_5f(N, n, a0, a1, a2, a3, a4):
    return _cs_4f(N,n,a0,a1,a2,a3) + a4*np.cos( (8*np.pi*n)/N )

def blackman(N): # "not very serious proposal", alpha=0.16
    a0 = 0.42 # (1-alpha)/2
    a1 = 0.5  # (1/2)
    a2 = 0.08 # alpha/2
    return _cs_3f(N,np.arange(0, N),a0,a1,a2)

def blackman_exact(N):
    a0 = 0.426_59  # 7938/18608
    a1 = 0.496_56  # 9240/18608
    a2 = 0.076_849 # 1430/18606
    return _cs_3f(N,np.arange(0, N),a0,a1,a2)

def nuttall(N):
    a0 = 0.355_768
    a1 = 0.487_396
    a2 = 0.144_232
    a3 = 0.012_604
    return _cs_4f(N,np.arange(0, N),a0,a1,a2,a3)

def blackman_nuttall(N):
    a0 = 0.363_581_9
    a1 = 0.489_177_5
    a2 = 0.136_599_5
    a3 = 0.010_641_1
    return _cs_4f(N,np.arange(0, N),a0,a1,a2,a3)

def blackman_harris(N): # Minimize sidelobes
    a0 = 0.358_75
    a1 = 0.488_29
    a2 = 0.141_28
    a3 = 0.011_68
    return _cs_4f(N,np.arange(0, N),a0,a1,a2,a3)

def flattop(N): # Minimal scalloping loss
    a0 = 0.215_578_95
    a1 = 0.416_631_58
    a2 = 0.277_263_158
    a3 = 0.083_578_947
    a4 = 0.006_947_368
    return _cs_5f(N,np.arange(0, N),a0,a1,a2,a3,a4)

# --- Sine window --- #
# even-integer power-of-sine
def sine0(N):
    return rect(N)
def sine2(N): # eq. to Hann
    return _cs_2f(N, np.arange(0, N), 0.5, 0.5)
def sine4(N):
    return _cs_3f(N, np.arange(0, N), 0.375, 0.5, 0.125)
def sine6(N):
    return _cs_4f(N, np.arange(0, N), 0.3125, 0.46875, 0.1875, 0.03125)
def sine8(N):
    return _cs_5f(N, np.arange(0, N), 0.2734375, 0.4375, 0.21875, 0.0625, 7.8125e-3)

# --- Adjustable windows --- #
def _gaus(x, N, L, sigma, p=2):
    return np.exp(-( (x-N/2)/(2*L*sigma) )**p)
def _gausN(x, L, sigma, p=2):
    return np.array([_gaus(n,len(x),L,sigma,p) for n in x])

def gaussian(N, sigma=0.4):
    n = np.arange(0, N)
    w = np.exp( -(1/2)*( (n-N/2)/(sigma*N/2) )**2)
    return w

def acon_gaussian(N, sigma=0.1): # Approximate Confined Gaussian
    n = np.arange(0, N)
    L = N+1

    num = _gaus(-0.5,N,L,sigma)*(_gausN(n+L,L,sigma) + _gausN(n-L,L,sigma))
    den = _gaus(-0.5+L,N,L,sigma)+_gaus(-0.5-L,N,L,sigma)
    w = _gausN(n, L, sigma) - (num/den)
    return w

def gen_gaussian(N, sigma=0.2, p=2): # Generalized Gaussian
    n = np.arange(0, N)
    w = np.exp(-( (n-N/2)/(sigma*N/2) )**p)
    return w

def tukey(N, a=0.5):
    w = np.zeros(N//2)
    n = np.arange(0, N/2)

    n1i = (0 <= n) & (n < (a*N)/2)
    n2i = ((a*N)/2 <= n) & (n <= N/2)

    w1 = 0.5*(1-np.cos( (2*np.pi*n[n1i])/(a*N) ))
    w2 = np.repeat(1, np.count_nonzero(n2i))

    w[n1i] = w1
    w[n2i] = w2
    w = np.concat((w, w[::-1]))
    return w

def planck_taper(N, e=0.25):
    w = np.zeros(N//2)
    n = np.arange(0, N/2)

    n1i = (1 <= n) & (n < e*N)
    n2i = (e*N <= n) & (n <= N/2)

    w1 = (1 + np.exp( (e*N)/n[n1i] - (e*N)/(e*N-n[n1i]) ))**-1
    w2 = np.repeat(1, np.count_nonzero(n2i))

    w[n1i] = w1
    w[n2i] = w2
    w = np.concat((w, w[::-1]))
    return w

# def kaiser(N, a): # TODO
#     n = np.arange(0, N)
#     w = I*(np.pi*a*np.sqrt(1-((2*n)/N-1)**2))

def exponential(N, D=8.69):
    n = np.arange(0, N)
    t = (N/2) * (8.69/D)
    w = np.e**(-np.abs(n-N/2)*1/t)
    return w

s = [k for k in globals().keys() if not k.startswith("_") and not k == "np"]

def get(name):
    return globals()[name]
