import numpy as np

from .modulation import Modulation
from ..base import Mod, dtype
from ..base.types import Category, Rep

class Differential(Modulation):
    """Differential phase coding of symbols: each one is sent as the phase change from the one before

        syms = bits * constellation * Differential()          # the constellation's points are phase changes
        bits = syms / Differential() / constellation

    The receiver needs no carrier phase, only that it holds over two symbols (a constant phase offset cancels). DBPSK is BPSK
    (points 1, -1: bit 1 turns the phase by 180 degrees); DQPSK is QPSK, the points on the axes for 802.11b (config/mod/constellation/dpsk/qpsk_axes)
    or at 45 degrees for pi/4-DQPSK, whose phase moves by an odd multiple of 45 degrees every symbol.

    The sent symbols have unit magnitude (the phase is what is coded). The phase before the first symbol is 0, so that one symbol of a
    stream decodes right only when the carrier phase is too; a decoder joined mid-stream is right from its second symbol.
    Received symbols are divided by the one before, so the receive gain cancels as well.
    """
    name = "Differential"

    def takes(self, encode):
        return {Rep.SYMBOLS}

    def pos_rate(self, decode):
        return 1.0

    def encode(self, data):
        ctx = dict(data.ctx)
        x = np.asarray(data.data)
        if not len(x):
            return data.derive(x.astype(np.complex64), Category.SYMBOL, dtype.SYMBOLS, Mod, ctx)
        ph = ctx.get((self, "tx"), 0.0) + np.cumsum(np.angle(x.astype(np.complex128)))
        ctx[(self, "tx")] = float(ph[-1] % (2*np.pi))
        return data.derive(np.exp(1j*ph).astype(np.complex64), Category.SYMBOL, dtype.SYMBOLS, Mod, ctx)

    def decode(self, data):
        ctx = dict(data.ctx)
        y = np.asarray(data.data).astype(np.complex128)
        if not len(y):
            return data.derive(y.astype(np.complex64), Category.SYMBOL, dtype.SYMBOLS, Mod, ctx)
        prev = np.concatenate(([ctx.get((self, "rx"), 1.0 + 0j)], y[:-1]))
        ctx[(self, "rx")] = y[-1]
        den = np.where(np.abs(prev) < 1e-12, 1.0, prev)
        return data.derive((y/den).astype(np.complex64), Category.SYMBOL, dtype.SYMBOLS, Mod, ctx)
