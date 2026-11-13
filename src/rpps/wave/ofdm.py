from types import SimpleNamespace

import numpy as np

from ..base import Stage

class OFDMTx(Stage):
    """IFFT and cyclic prefix: n_fft carrier values per OFDM symbol in (see `rp.symbol.Carriers`), n_fft + n_cp samples out.
    A partial OFDM symbol waits in the stage for the rest"""
    def __init__(self, n_fft=64, n_cp=16):
        self.n_fft, self.n_cp = int(n_fft), int(n_cp)

    def new_state(self, meta):
        return SimpleNamespace(held=None)

    def step(self, samples, st, meta):
        x = np.asarray(samples) if st.held is None else np.concatenate((st.held, samples))
        n = len(x)//self.n_fft
        st.held = x[n*self.n_fft:]
        time = np.fft.ifft(x[:n*self.n_fft].reshape(n, self.n_fft), axis=1)*np.sqrt(self.n_fft)
        out = np.concatenate((time[:, -self.n_cp:], time), axis=1).reshape(-1) if n else np.zeros(0, np.complex64)
        ratio = (self.n_fft + self.n_cp)/self.n_fft
        meta.op.Fs = meta.Fs*ratio
        meta.op.sps = ratio # samples per carrier value: more than one, so this is samples
        return out.astype(np.complex64)

class OFDMRx(Stage):
    """Cyclic prefix off, then the FFT: samples starting on an OFDM symbol's prefix in, n_fft carrier values per OFDM symbol
    out (one value per carrier: symbols). Finding where the OFDM symbols start is sync's job"""
    def __init__(self, n_fft=64, n_cp=16):
        self.n_fft, self.n_cp = int(n_fft), int(n_cp)

    def new_state(self, meta):
        return SimpleNamespace(held=None)

    def step(self, samples, st, meta):
        L = self.n_fft + self.n_cp
        x = np.asarray(samples) if st.held is None else np.concatenate((st.held, samples))
        n = len(x)//L
        st.held = x[n*L:]
        blocks = x[:n*L].reshape(n, L)[:, self.n_cp:]
        out = (np.fft.fft(blocks, axis=1)/np.sqrt(self.n_fft)).reshape(-1) if n else np.zeros(0, np.complex64)
        meta.op.Fs = meta.Fs*self.n_fft/L
        meta.op.sps = 1
        return out.astype(np.complex64)
