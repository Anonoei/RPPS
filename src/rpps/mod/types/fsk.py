from types import SimpleNamespace

import numpy as np

from ..constellation import ModConstellation
from ..utils import shapes
from ...base import Stage, Mod, Analog, pproc, dtype
from ...base.rpps import align

class FSK(ModConstellation):
    """Continuous-phase frequency-shift keying: `FSK(2)` binary, `FSK(4)` four tones. Gray coded

    M: tones. sps: samples per symbol of the waveform it makes
    h: modulation index, the spacing between neighbouring tones in cycles per symbol (1.0: orthogonal for noncoherent
       detection; 0.5: MSK-like)
    Demodulation is a quadrature discriminator averaged over each symbol (integrate and dump): the data must start on a
    symbol boundary (`rp.sync.time` first when it doesn't) at `sps` samples per symbol.
    """
    def __init__(self, M=2, sps=8, h=1.0, mapping=None):
        pts, lab = shapes.ask(M, bipolar=True)
        points, maps = self._generated(pts, lab)
        super().__init__(mapping, points, maps)
        self.M, self.sps, self.h = M, int(sps), h
        self.name = f"{M}FSK"
        lev = 2*np.arange(M) - (M - 1)
        self._norm = float(np.sqrt(np.mean(lev**2.0)))

    def _cycles(self, levels):
        """Tone (cycles per symbol) of unit-power ASK level points"""
        return levels * (self.h/2) * self._norm

    def encode(self, data):
        mod = super().encode(data) # Bits -> level points; a partial last symbol is held or padded as usual
        lev = np.real(mod.data)
        if not len(lev):
            return mod
        ctx = dict(mod.ctx)
        key = (self, "phase")
        phase0 = ctx.get(key, 0.0)
        inc = np.repeat(2*np.pi*self._cycles(lev)/self.sps, self.sps)
        ph = phase0 + np.cumsum(inc)
        ctx[key] = float(ph[-1] % (2*np.pi))
        meta = mod.meta.fork()
        meta.op.sps = self.sps
        return mod.derive(np.exp(1j*ph).astype(np.complex64), pproc.MOD, dtype.SAMPLES, Mod, ctx, meta.fork())

    def decode(self, data):
        sps = int(round(data.sps)) if data.sps >= 1 else self.sps
        ctx = dict(data.ctx)
        x = np.asarray(data.data)
        prev = ctx.get((self, "last"))
        if len(x):
            z = np.concatenate(([x[0] if prev is None else prev], x))
            disc = np.angle(z[1:] * np.conj(z[:-1])) # Phase advance per sample
            ctx[(self, "last")] = x[-1]
            if prev is None and len(disc) > 1:
                disc[0] = disc[1] # No sample before the first: borrow the next phase step
        else:
            disc = np.zeros(0)
        disc = align(ctx, (self, "disc"), disc, sps, pad=False, log=self.log)
        cyc = disc.reshape(-1, sps).mean(axis=1) * sps/(2*np.pi) # Tone, cycles per symbol
        lev = cyc / ((self.h/2) * self._norm)
        symbols = Analog(lev.astype(np.complex64), pproc.MOD, dtype.SYMBOLS, ctx, data.meta)
        out = super().decode(symbols)
        return out

class FMDemod(Stage):
    """Quadrature FM demodulator as a stream stage: the last sample carries across blocks"""
    __slots__ = ("gain",)
    def __init__(self, gain=0.5):
        self.gain = gain

    def new_state(self, meta):
        return SimpleNamespace(last=None)

    def step(self, samples, st, meta):
        if len(samples) == 0:
            return np.zeros(0, dtype=np.float32)
        last = samples[0] if st.last is None else st.last
        st.last = samples[-1]
        z = np.concatenate(([last], samples))
        return self.gain * np.angle(z[:-1] * np.conj(z[1:]))
