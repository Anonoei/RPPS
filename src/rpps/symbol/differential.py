import numpy as np

from .modulation import Modulation
from ..base import Mod, dtype
from ..base.types import Category, Rep

class Differential(Modulation):
    """Differential phase coding of symbols: each one is sent as the phase change from the one before

        syms = bits * constellation * Differential()          # the constellation's points are phase changes
        bits = syms / Differential() / constellation

    The receiver needs no carrier phase, only that it holds over two symbols (a constant phase offset cancels). DBPSK is BPSK
    (points 1, -1: bit 1 turns the phase by 180 degrees); DQPSK is QPSK, the points on the axes for 802.11b (config/constellation/psk/qpsk_axes)
    or at 45 degrees for pi/4-DQPSK (psk/qpsk itself), whose phase moves by an odd multiple of 45 degrees every symbol.

    The sent symbols have unit magnitude (the phase is what is coded). The phase before the first symbol is 0, so that one symbol of a
    stream decodes right only when the carrier phase is too; a decoder joined mid-stream is right from its second symbol.
    Each received symbol is multiplied by the conjugate of the one before (the noncoherent detector): the phase change, with
    each symbol weighted by its own reliability, as the demapper's LLRs want. Dividing instead (y/prev) gives the same hard
    decisions, but a weak `prev` blows the noise up: measured DQPSK into a K=7 Viterbi decoder, BER 5e-3 dividing against
    1.6e-5 at Eb/N0 7 dB (over 2 dB). The phase cancels; the signal comes out at the input's amplitude squared, so the input
    is at unit amplitude as for any demapper (the hard decisions don't depend on it).
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
        return data.derive((y*np.conj(prev)).astype(np.complex64), Category.SYMBOL, dtype.SYMBOLS, Mod, ctx)
