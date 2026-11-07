from types import SimpleNamespace

import numpy as np

from ..base import Stage
from .impl.designer import designer

class Channelizer(Stage):
    """Polyphase filter bank: splits a wideband signal into `channels` equal bands, each at Fs/channels

        bands = wide + rp.filters.Channelizer(8)      # bands.data is (time, 8); column k is centred on k*Fs/8
        one = bands.data[:, 2]

    Channel k is centred on k*Fs/channels (channels above half the rate are the negative frequencies).
    taps_per: prototype filter taps per channel (longer: sharper channels, more delay)
    """
    __slots__ = ("M", "h")
    def __init__(self, channels, taps_per=12, window=np.hamming):
        self.M = int(channels)
        L = self.M*taps_per
        self.h = designer(L, 1.0/self.M, fs=2, window=window).astype(np.float32) # Cut off at half a channel (fs=2: Nyquist is 1)

    def new_state(self, meta):
        return SimpleNamespace(buf=np.zeros(len(self.h) - 1, dtype=np.complex64))

    def step(self, samples, st, meta):
        M, L = self.M, len(self.h)
        buf = np.concatenate((st.buf, samples.astype(np.complex64, copy=False)))
        frames = (len(buf) - (L - 1))//M # Outputs: one per M new samples
        meta.op.Fs /= M
        meta.op.sps /= M
        if frames <= 0:
            st.buf = buf
            return np.zeros((0, M), dtype=np.complex64)
        ends = (L - 1) + M*np.arange(frames) # Index of the newest sample of each window
        win = np.lib.stride_tricks.sliding_window_view(buf, L)[ends - (L - 1)][:, ::-1] # x[n - i], i = 0..L-1
        folded = (win * self.h).reshape(frames, L//M, M).sum(axis=1)
        out = np.fft.ifft(folded, axis=1)*M # y_k = sum_m s[m] e^{+j 2 pi k m / M}
        st.buf = buf[ends[-1] + M - (L - 1):] # From where the next window starts
        meta.delay += (L - 1)/2/(meta.Fs)
        return out.astype(np.complex64)
