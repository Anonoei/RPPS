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
