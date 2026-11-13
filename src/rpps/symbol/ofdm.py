import numpy as np

from .modulation import Modulation
from ..base import Mod, dtype
from ..base.types import Category, Rep
from ..base.rpps import align

class Carriers(Modulation):
    """OFDM carrier allocation, the symbol-transform part of OFDM: data symbols <-> the carrier values of each OFDM symbol

        carriers = rp.symbol.Carriers()                                     # 64 carriers: 48 data, 4 pilots (802.11a layout)
        wave = bits * qam * carriers + rp.wave.OFDMTx(64, 16)             # transmit
        syms = wave + rp.wave.OFDMRx(64, 16) + rp.sync.PilotEqualizer(carriers) / carriers   # receive

    Encoding puts each `data_per_symbol` data symbols on the data carriers, the pilots' known values on the pilot carriers
    (their sign alternating from one OFDM symbol to the next), and zero on the rest: n_fft values per OFDM symbol, in FFT
    bin order. Decoding takes those n_fft values back to the data symbols. Equalizing them is sync's job
    (`rp.sync.PilotEqualizer`), and the FFT and cyclic prefix are the wave transform's (`rp.wave.OFDMTx`, `OFDMRx`).

    n_fft: carriers. used: carrier numbers (-n_fft/2 .. n_fft/2 - 1, 0 is DC) that carry something, default 52 (802.11a).
    pilots: carriers (a subset of used) with a known value, default -21 -7 7 21. More pilots follow a longer echo.
    """
    name = "Carriers"

    def __init__(self, n_fft=64, used=None, pilots=(-21, -7, 7, 21)):
        super().__init__()
        self.n_fft = int(n_fft)
        used = [k for k in range(-26, 27) if k != 0] if used is None else sorted(used)
        if max(map(abs, used)) > self.n_fft//2:
            raise ValueError("used carriers don't fit the FFT")
        self.pilots = tuple(pilots)
        self.used = tuple(used)
        self.data = tuple(k for k in used if k not in self.pilots)
        if not set(self.pilots) <= set(used):
            raise ValueError("pilots must be among the used carriers")
        self._pilot_vals = np.array([1, 1, 1, -1][:len(self.pilots)] + [1]*max(0, len(self.pilots) - 4), dtype=np.complex64)

    def __str__(self):
        return f"Carriers:{self.n_fft}:{len(self.data)}d{len(self.pilots)}p"

    def takes(self, encode):
        return {Rep.SYMBOLS}

    @property
    def data_per_symbol(self):
        return len(self.data)

    def bins(self, k):
        """FFT bins of carrier numbers"""
        return np.asarray(k) % self.n_fft

    def pilot_values(self, index):
        """Pilot values of OFDM symbols number `index` (an array): (len(index), pilots); the sign alternates by symbol"""
        sign = np.where(np.asarray(index) % 2 == 0, 1.0, -1.0)[:, None]
        return sign*self._pilot_vals[None, :]

    def encode(self, data):
        """Data symbols -> n_fft carrier values per OFDM symbol. A partial last OFDM symbol is held, or padded when flushed"""
        ctx = dict(data.ctx)
        syms = align(ctx, (self, True), np.asarray(data.data), self.data_per_symbol, pad=True)
        n = len(syms)//self.data_per_symbol
        first = ctx.get((self, "count"), 0)
        grid = np.zeros((n, self.n_fft), dtype=np.complex64)
        grid[:, self.bins(self.data)] = syms.reshape(n, self.data_per_symbol)
        grid[:, self.bins(self.pilots)] = self.pilot_values(first + np.arange(n))
        ctx[(self, "count")] = first + n
        return data.derive(grid.reshape(-1), Category.SYMBOL, dtype.SYMBOLS, Mod, ctx)

    def decode(self, data):
        """n_fft carrier values per OFDM symbol (equalized) -> the data symbols"""
        ctx = dict(data.ctx)
        x = align(ctx, (self, False), np.asarray(data.data), self.n_fft, pad=False, log=self.log)
        grid = x.reshape(-1, self.n_fft)
        out = grid[:, self.bins(self.data)].reshape(-1)
        return data.derive(out.astype(np.complex64), Category.SYMBOL, dtype.SYMBOLS, Mod, ctx)
