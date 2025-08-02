import rpps as rp
import numpy as np
import matplotlib.pyplot as plt

def main():
    N = 128 # number of samples to simulate
    Fs = 300 # sample rate
    Ts = 1/Fs # sample period
    cf = 0

    t = np.arange(N)/Fs
    s = np.exp(2j*np.pi*cf*t)
    f = np.arange(N) - N//2

    n_pwr = 0.0000001
    _awgn = (np.random.randn(len(s)) + 1j*np.random.randn(len(s)))/np.sqrt(2)
    s += _awgn*np.sqrt(n_pwr)

    all(f, s)
    # ind(f, s)


def all(f, s):
    N = len(s)
    fig, ax = plt.subplots(1,2)
    fig.tight_layout()
    ax[0].grid(True)
    ax[1].grid(True)

    ax[0].set_xlim(f[0], f[-1])
    ax[1].set_xlim(f[0], f[-1])
    ax[1].set_ylim(-130, 0)

    for window in rp.filters.window.s:
        w = rp.filters.window.get(window)(N)
        sw = s * w
        sw /= 100
        psd = np.fft.fftshift(np.fft.fft(sw))
        psd = 10*np.log10(np.abs(psd)**2)
        ax[0].plot(f, w, label=window)
        ax[1].plot(f, psd, label=window)
    fig.legend()
    plt.show()
    exit()

def ind(f, s):
    N = len(s)
    # w = rp.filters.window.rect(N)
    # # B-Spline
    # w = rp.filters.winfow.bartlett(N)
    # w = rp.filters.window.parzen(N)
    # # Polynomial
    # w = rp.filters.window.welch(N)
    # # Raised-cosine
    # w = rp.filters.window.hann(N)
    w = rp.filters.window.hamming(N)
    # # Cosine-sum
    # w = rp.filters.window.blackman(N)
    # w = rp.filters.window.nuttall(N)
    # w = rp.filters.window.blackman_nuttall(N)
    # w = rp.filters.window.blackman_harris(N)
    # w = rp.filters.window.flattop(N)
    # # Adjustable
    # w = rp.filters.window.acon_gaussian(N, 0.1)
    # w = rp.filters.window.gen_gaussian(N, 0.2)
    # w = rp.filters.window.tukey(N, 0.5)
    # w = rp.filters.window.planck_taper(N,0.25)
    # w = rp.filters.window.exponential(N, 60)
    s *= w
    s /= 100

    psd = np.fft.fftshift(np.fft.fft(s))
    psd = 10*np.log10(np.abs(psd)**2)

    fig, ax = plt.subplots(1,2)
    fig.tight_layout()
    ax[0].grid(True)
    ax[1].grid(True)

    ax[0].set_xlim(f[0], f[-1])
    ax[1].set_xlim(f[0], f[-1])

    ax[0].fill_between(f, w, 0)
    ax[1].fill_between(f, psd, -130)


    ax[1].set_ylim(-130, 0)
    plt.show()

if __name__ == "__main__":
    main()
