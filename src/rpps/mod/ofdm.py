import numpy as np

from .modulation import Modulation
from ..base import Mod, Analog, pproc, dtype
from ..base.rpps import align

class OFDM(Modulation):
    """Orthogonal frequency-division multiplexing on constellation symbols

        ofdm = rp.mod.OFDM()                                    # 64 point FFT, 16 sample prefix, 48 data + 4 pilot carriers
        wave = (bits * rp.mod.QAM(16)) * ofdm                   # symbols in, time samples out (one OFDM symbol per 48 symbols)
        syms = wave / ofdm                                      # samples in (starting on a symbol), equalized symbols out

    The prefix covers a channel whose impulse response is shorter than it. Each OFDM symbol carries `pilots` (known
    values); decoding divides each carrier by the channel measured on them (interpolated), so a multipath channel
    and a constant phase offset are removed. It doesn't find the start of the signal or correct a frequency offset.

    n_fft, n_cp: FFT size and cyclic prefix length
    used: carrier numbers (-n_fft/2 .. n_fft/2 - 1, 0 is DC) that carry something. Default 52, the 802.11a layout
    pilots: carriers (a subset of used) with a known value, default -21 -7 7 21. More pilots allow a longer echo
    channel_taps: length of the channel impulse response fitted to the pilots (default 3/4 of the pilots, at most the prefix)
    """
    name = "OFDM"
    def __init__(self, n_fft=64, n_cp=16, used=None, pilots=(-21, -7, 7, 21), channel_taps=None):
        super().__init__()
        self.channel_taps = channel_taps
        self.n_fft, self.n_cp = n_fft, n_cp
        used = [k for k in range(-26, 27) if k != 0] if used is None else sorted(used)
        if n_fft == 64 and used is not None and max(map(abs, used)) > 31:
            raise ValueError("used carriers don't fit the FFT")
        self.pilots = tuple(pilots)
        self.used = tuple(used)
        self.data = tuple(k for k in used if k not in self.pilots)
        if not set(self.pilots) <= set(used):
            raise ValueError("pilots must be among the used carriers")
        self._pilot_vals = np.array([1, 1, 1, -1][:len(self.pilots)] + [1]*max(0, len(self.pilots) - 4), dtype=np.complex64)

    def __str__(self):
        return f"OFDM:{self.n_fft}+{self.n_cp}:{len(self.data)}d{len(self.pilots)}p"

    @property
    def data_per_symbol(self):
        return len(self.data)

    @property
    def symbol_len(self):
        return self.n_fft + self.n_cp

    def _bin(self, k):
        return np.asarray(k) % self.n_fft

    def encode(self, data):
        """Constellation symbols -> samples. A partial last OFDM symbol is held, or padded when flushed"""
        ctx = dict(data.ctx)
        syms = align(ctx, (self, True), np.asarray(data.data), self.data_per_symbol, pad=True)
        n = len(syms)//self.data_per_symbol
        grid = np.zeros((n, self.n_fft), dtype=np.complex64)
        grid[:, self._bin(self.data)] = syms.reshape(n, self.data_per_symbol)
        sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)[:, None] # Pilots alternate between symbols
        grid[:, self._bin(self.pilots)] = sign*self._pilot_vals[None, :]
        time = np.fft.ifft(grid, axis=1)*np.sqrt(self.n_fft)
        out = np.concatenate((time[:, -self.n_cp:], time), axis=1).reshape(-1) if n else np.zeros(0, dtype=np.complex64)
        ctx[(self, "count")] = ctx.get((self, "count"), 0) + n
        return data.derive(out.astype(np.complex64), pproc.MOD, dtype.SAMPLES, Mod, ctx)

    def decode(self, data):
        """Samples (starting on the prefix of an OFDM symbol) -> equalized constellation symbols"""
        ctx = dict(data.ctx)
        x = align(ctx, (self, False), np.asarray(data.data), self.symbol_len, pad=False, log=self.log)
        n = len(x)//self.symbol_len
        if not n:
            return data.derive(np.zeros(0, dtype=np.complex64), pproc.MOD, dtype.SYMBOLS, Mod, ctx)
        blocks = x.reshape(n, self.symbol_len)[:, self.n_cp:]
        grid = np.fft.fft(blocks, axis=1)/np.sqrt(self.n_fft)
        first = ctx.get((self, "count"), 0)
        sign = np.where((first + np.arange(n)) % 2 == 0, 1.0, -1.0)[:, None]
        pb = self._bin(self.pilots)
        order = np.argsort(self.pilots)
        pk = np.asarray(self.pilots)[order]
        h_p = grid[:, pb][:, order] / (sign*self._pilot_vals[None, :][:, order])
        db = self._bin(self.data)
        dk = np.asarray(self.data)
        # Least squares fit of a short impulse response (L taps) to the channel measured on the pilots, then its
        # frequency response on the data carriers: handles echoes as long as the pilots sample the response often enough
        L = self.channel_taps or max(1, min(self.n_cp, (3*len(pk))//4))
        lags = np.arange(L)
        fit = np.linalg.pinv(np.exp(-2j*np.pi*np.outer(pk, lags)/self.n_fft))
        h = (np.exp(-2j*np.pi*np.outer(dk, lags)/self.n_fft) @ (fit @ h_p.T)).T
        out = (grid[:, db]/np.where(np.abs(h) < 1e-9, 1e-9, h)).reshape(-1)
        ctx[(self, "count")] = first + n
        return data.derive(out.astype(np.complex64), pproc.MAP if False else pproc.MOD, dtype.SYMBOLS, Mod, ctx)
