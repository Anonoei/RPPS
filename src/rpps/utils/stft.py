import numpy as np

from ..filters import Window
from . import psd as _psd

def stft(samples, nfft=1024, overlap=0.8, win="hamming"):
    win_func = Window.get(win)
    overlap = 1-overlap
    n_samp = len(samples)
    n_frames = int(n_samp/(nfft*overlap))
    out = np.zeros((nfft, n_frames), dtype=samples.dtype)

    # print(f"stft for {n_samp} samples / {out.shape[1]}*{nfft} FFT @ overlap {overlap}")

    for i in range(n_frames):
        s_idx = int(i*(nfft*overlap))
        e_idx = s_idx + nfft
        # print(f"snip {i} = [{s_idx}:{e_idx}]")
        if e_idx > n_samp:
            segment = np.zeros(nfft, dtype=samples.dtype)
            f_idx = n_samp-s_idx
            segment[:f_idx] = samples[s_idx:]
            # segment[f_idx:] = samples[-f_idx:]
        else:
            segment = samples[s_idx:e_idx]
        out[:,i] = np.fft.fft(segment*win_func(nfft))
    return out

def psd(samples, Fs=1, vbw_hz=None):
    out = np.zeros_like(samples, dtype=np.float32)
    for i in range(samples.shape[1]):
        y = np.abs(samples[:,i])**2 / (len(samples)*Fs)
        y = np.fft.fftshift(10.0*np.log10(y))
        if vbw_hz is not None:
            smooth = _psd.vbw_calc(samples.shape[0], Fs, vbw_hz) # type: ignore
            y = _psd.vbw(y, smooth) # type: ignore
        out[:,i] = y
    return out
