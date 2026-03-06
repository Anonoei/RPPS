import numpy as np

from .psd import psd

class Signal:
    def __init__(self, mid, width, bottom, height):
        self.mid = mid
        self.baud = width
        self.bottom = bottom
        self.height = height

def trim_psd(_psd):
    N = len(_psd)
    N1 = int((1/100)*N)
    return _psd[N1:-N1], N1

def find_noise_floor(_psd):
    return np.min(trim_psd(_psd))

def smooth_psd(_psd, window_size=None):
    """Smooth PSD amplitudes"""
    if window_size is None:
        window_size = len(_psd)//100
        print(f"PSD {len(_psd)} using window_size {window_size}")
    window = np.ones(window_size) / window_size
    out = np.convolve(_psd, window, mode='same')
    return out

def find_coarse(_psd, Fs: float = 1):
    pmin = np.min(_psd)
    pmax = np.max(_psd)
    pmed = np.mean(_psd)
    pstd = np.std(_psd)
    pdev = pmax - pmin
    pchk = pstd/2
    print(f"Noise floor is {pmin:0.3f}, max: {pmax:0.3f}, mean: {pmed:0.3f}, std: {pstd:0.3f}, dev: {pdev:0.3f}, chk: {pchk:0.3f}")
    check = np.where(_psd > (pmed - pchk))[0]
    coarse = []
    for i in check:
        if len(coarse) == 0:
            coarse.append([i])
        elif len(coarse[-1]) == 1:
            coarse[-1].append(i)
        else:
            if coarse[-1][1] + 3 > i:
                coarse[-1][1] = i
            else:
                coarse.append([i])
    return coarse

def find_signals(samps, Fs: float = 1):
    N = len(samps)
    _psd = psd(samps, Fs)
    _psd = smooth_psd(_psd)
    _psd, trimmed = trim_psd(_psd)
    # _psd = psd(samps, Fs=Fs, vbw_hz=(20/len(samps))*Fs)
    coarse = find_coarse(_psd, Fs)
    signals = []
    bin_hz = Fs/N
    print(f"{Fs} / {N} = bin_hz: {bin_hz}")
    for sig in coarse:
        il, ir = sig
        amp = _psd[il:ir+1]
        width = ir-il
        bottom = np.min(amp)
        height = np.max(amp) - bottom
        # sstd = np.std(amp)
        mid = int(il+(width/2))
        # print(f"signal peak: {peak}, std: {sstd}, l/r/m: {il}/{ir}/{mid}, width: {width}")
        # print(f"lpwr: {_psd[il]:0.3f}, rpwr: {_psd[ir]:0.3f}, mpwr: {_psd[mid]:0.3f}, height: {height}")
        left = il + trimmed
        right = ir + trimmed
        mid_hz = ((mid+trimmed)-N/2)*bin_hz
        signal = Signal(mid_hz, width*bin_hz, bottom, height)
        # print(f"Signal {len(signals)}: mid: {signal.mid}, baud: {signal.baud}")
        signals.append(signal)
        # signals.append((left, right, bottom, height))
    # print(f"Signals: {signals}")
    return signals
