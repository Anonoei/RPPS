import numpy as np

from ..base import Mod, dtype
from ..base.types import Category, Rep
from ..base.rpps import align
from .modulation import Modulation
from ..spread.code import Code
from .despread import correlate

class Spread(Modulation):
    """Direct-sequence spreading as an operator: symbols (or bits) to chips and back, one sample per chip (shape and upsample them
    with the waveform stages)

        chips = syms * Spread(code)                 # one sequence (Barker, GPS C/A): every symbol times the sequence
        syms = chips / Spread(code)                 # correlate each code period
        chips = bits * Spread(set)                  # a set (802.15.4): every log2(count) bits, the first the least significant, pick a sequence
        llr = chips / Spread(set)                   # strongest correlations: one LLR per bit

    pair: put even chips on I and odd chips on Q (complex chips at half the chip rate), as O-QPSK carries 802.15.4's. Decoding a set
    through pair needs the carrier phase removed first: the I and Q parts are taken as the two rails of chips.
    A single-sequence code works on symbols that need a code period of chips each; a partial period is held until the rest arrives.
    """
    name = "Spread"

    def __init__(self, code: Code, pair=False):
        super().__init__()
        self.code, self.pair = code, bool(pair)
        if pair and code.length % 2:
            raise ValueError("pair needs an even number of chips per sequence")
        if code.count > 1:
            self.bits_per_symbol = int(np.log2(code.count))
            if 2**self.bits_per_symbol != code.count:
                raise ValueError(f"A set of {code.count} sequences is not a power of two")

    def __str__(self):
        return f"Spread:{self.code.name}"

    def takes(self, encode):
        """A set encodes channel bits (or data bits, uncoded); one sequence encodes symbols. Both decode chips (symbols)"""
        if encode and self.code.count > 1:
            return {Rep.CHANNEL, Rep.DATA}
        return {Rep.SYMBOLS}

    def pos_rate(self, decode):
        """Output items per input item (rpps.profile positions): chips per symbol or bits, or the other way round"""
        if self.code.count > 1:
            r = self.bits_per_symbol/(self.code.length//(2 if self.pair else 1))
        else:
            r = 1/self.code.length
        return r if decode else 1/r

    @property
    def chips_per_symbol(self):
        return self.code.length

    def _meta(self, data, factor):
        """Rates after changing the symbol rate by `factor`: chips are one sample each"""
        m = data.meta.fork()
        m.op.Fs = data.Fs*factor
        m.op.sps = 1
        return m.fork()

    # ---- encode

    def encode(self, data):
        if self.code.count > 1:
            return self._encode_set(data)
        x = np.asarray(data.data)
        chips = self.code.spread(x).astype(np.complex64)
        return data.derive(chips, Category.SYMBOL, dtype.SYMBOLS, Mod, dict(data.ctx), self._meta(data, self.code.length))

    def _encode_set(self, data):
        k = self.bits_per_symbol
        ctx = dict(data.ctx)
        bits = align(ctx, (self, True), np.asarray(data.bits(), dtype=np.uint8), k, pad=True)
        idx = (bits.reshape(-1, k) << np.arange(k)).sum(axis=1) if len(bits) else np.zeros(0, dtype=int)
        chips = self.code.spread(np.ones(len(idx)), index=idx)
        if self.pair:
            chips = chips.reshape(-1, 2)
            chips = chips[:, 0] + 1j*chips[:, 1]
        per = self.code.length//(2 if self.pair else 1)
        return data.derive(chips.astype(np.complex64), Category.SYMBOL, dtype.SYMBOLS, Mod, ctx, self._meta(data, per/k))

    # ---- decode

    def decode(self, data):
        ctx = dict(data.ctx)
        N = self.code.length
        x = np.asarray(data.data)
        if self.code.count > 1 and self.pair:
            x = np.stack((x.real, x.imag), axis=1).reshape(-1) if len(x) else x.real # Back to one rail of chips
        x = align(ctx, (self, False), x, N, pad=False, log=self.log)
        corr = correlate(x, self.code, 1) if len(x) else np.zeros((0, self.code.count), dtype=np.complex64)
        if self.code.count > 1:
            return self._decode_set(data, corr, ctx)
        syms = corr[:, 0].astype(np.complex64)
        return data.derive(syms, Category.SYMBOL, dtype.SYMBOLS, Mod, ctx, self._meta(data, 1/N))

    def _decode_set(self, data, corr, ctx):
        k = self.bits_per_symbol
        mag = np.abs(corr)
        labels = (np.arange(self.code.count)[:, None] >> np.arange(k)) & 1 # (count, k)
        llr = np.empty((len(mag), k), dtype=np.float32)
        for b in range(k):
            llr[:, b] = mag[:, labels[:, b] == 0].max(axis=1) - mag[:, labels[:, b] == 1].max(axis=1)
        return data.derive(llr.reshape(-1)*self.code.length, Category.SYMBOL, dtype.LLR, Mod, ctx, rep=Rep.CHANNEL)
