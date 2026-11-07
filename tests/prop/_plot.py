import numpy as np

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

def gaussian(y, sigma):
    fwhm = sigma2fwhm(sigma)
    x = np.arange(-int(sigma)*4, int(sigma)*4+1)

    kernel = np.exp(-(x) ** 2 / (2 * sigma ** 2))

    kernel_above_thresh = kernel > 0.0001
    finite_kernel = kernel[kernel_above_thresh]
    finite_kernel = finite_kernel / finite_kernel.sum()

    kernel_n_below_0 = int((len(finite_kernel) - 1) / 2.)

    smi = int(fwhm)
    smin = np.min(y)
    ex_samps = np.concatenate((np.repeat(smin, smi), y, np.repeat(smin, smi)))
    convolved_y = np.convolve(ex_samps, finite_kernel)
    smoothed_by_convolving = convolved_y[kernel_n_below_0+smi:(len(y)+kernel_n_below_0+smi)]
    return smoothed_by_convolving

def vbw(samps, Fs, vbw):
    smooth = vbw/(Fs / len(samps))
    sigma = fwhm2sigma(smooth)
    return gaussian(samps, sigma)

def phasor(syms, ax):
    snip = syms / np.real(np.max(syms))
    ax.plot(snip.real, snip.imag, ".")
    ax.grid(True)
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.set_aspect("equal")

def psd(t, samples, ax, Fs=1, vbw_hz=None):
    _psd = np.abs(np.fft.fft(samples))**2 / (len(samples)*Fs)
    _psd = np.fft.fftshift(10*np.log10(_psd))
    if vbw_hz is not None:
        _psd = vbw(_psd, Fs, vbw_hz)
    ax.plot(t, _psd)

def time(t, samples, ax):
    ax.plot(t, samples.real+samples.imag)

def timeIQ(t, samples, ax):
    ax.plot(t, samples.real)
    ax.plot(t, samples.imag)
