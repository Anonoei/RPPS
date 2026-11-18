from types import SimpleNamespace

from .sample import Sample, Meta
from ... import _accel

from ..filters import shaping
from ..filters.filter import FIR
from ..filters.filters import ConvFilter

import numpy as np

class Upsample(Sample):
    __slots__ = ("ratio")
    def __init__(self, ratio):
        self.ratio = int(ratio)

    def run(self, samples):
        raise NotImplementedError()

    def __str__(self):
        return f"{type(self).__name__}({self.ratio})"

    def step(self, samples, st, meta: Meta):
        meta.op.Fs *= self.ratio
        meta.op.sps *= self.ratio
        return self.run(samples)

class FFT(Upsample):
    """Upsample each block through the FFT (the block is treated as one period, so edges wrap)"""
    def run(self, samples):
        frq = np.fft.fft(samples)
        up = self.ratio*np.hstack([
            frq[:len(frq)//2],
            frq[len(frq)//2]/self.ratio,
            np.zeros((len(frq)*self.ratio)-len(frq)-1),
            frq[len(frq)//2]/self.ratio,
            frq[-(len(frq)//2-1):]
        ])

        up = np.fft.ifft(up)
        return up

class Interpolate(Upsample):
    """Zero stuff by `ratio` and filter with `filt` (an FIR stage, a pulse shape) in one step: the same output as
    `Zeros(ratio) + filt`, in polyphase form, so the zeros are never multiplied (ratio times less work)

        tx = symbols + rp.signal.sample.up.Interpolate(8, rp.signal.filters.load("pulse", "rrc/0_35"))

    Output sample n*ratio + p is phase p of the taps (taps[p::ratio]) run over the input, so each phase is an ordinary
    FIR on the input samples, with one history shared by all of them. A flush (the end of the stream) also sends the
    filter's tail, so the last symbols' pulses are complete."""
    __slots__ = ("filt",)
    def __init__(self, ratio, filt):
        super().__init__(ratio)
        self.filt = filt

    def __str__(self):
        return f"Interpolate({self.ratio}, {self.filt})"

    def new_state(self, meta):
        R = self.ratio
        m = meta.fork() # The filter runs at the output rate
        m.state.Fs, m.state.sps = m.state.Fs*R, m.state.sps*R
        m.op = m.state.copy()
        taps = np.asarray(self.filt.new_state(m).taps)
        phases = np.zeros((R, -(-len(taps)//R)), dtype=taps.dtype) # Short phases end in zeros: all keep one history
        for p in range(R):
            phases[p, :len(taps[p::R])] = taps[p::R]
        return SimpleNamespace(phases=phases, ntaps=len(taps), hist=None)

    def step(self, samples, st, meta: Meta):
        R = self.ratio
        L = st.phases.shape[1]
        if meta.flush: # The end of the transmission: the last pulses go out whole (their tail, ntaps - 1 samples, rounded
            # up to whole symbols). Cut off, the last half span of symbols arrives distorted: noiseless 32APSK short frames
            # failed their LDPC on it
            samples = np.concatenate((samples, np.zeros(-(-(st.ntaps - 1)//R), samples.dtype)))
        if _accel.native is not None and samples.dtype == np.complex64 and st.phases.dtype.kind == "f":
            if st.hist is None:
                st.hist = np.zeros(L - 1, np.complex64)
            buf = np.concatenate((st.hist, samples))
            out = _accel.native.fir_interpolate(buf, np.ascontiguousarray(st.phases, dtype=np.float64), L - 1, len(samples))
            st.hist = buf[len(buf) - (L - 1):]
        else: # Each phase an FIR over the input (the reference)
            hist, outs = st.hist, []
            for p in range(R):
                y, new = FIR.stream(samples, st.phases[p], hist)
                outs.append(y)
            st.hist = new
            out = np.empty(len(samples)*R, dtype=np.result_type(*outs))
            for p, y in enumerate(outs):
                out[p::R] = y
        meta.delay += (st.ntaps - 1)/2/(meta.Fs*R)
        meta.op.Fs *= R
        meta.op.sps *= R
        return out

class Sinc(Interpolate):
    """Zero stuff and interpolate with a windowed sinc. The filter history carries across blocks"""
    __slots__ = ("pulse",)
    def __init__(self, ratio, taps=None):
        ratio = int(ratio)
        taps = max(ratio*12, 31) if taps is None else int(taps)
        super().__init__(ratio, ConvFilter(np.sinc((np.arange(taps | 1) - (taps | 1)//2)/ratio))) # Streaming: symmetric, centre tap
        self.pulse = shaping.ShapingSinc(taps, ratio, 1) # One-shot kernel

    def __str__(self):
        return f"Sinc({self.ratio})"

    def run(self, samples):
        """One-shot (stateless) interpolation, same length as ratio*samples"""
        up = np.zeros(len(samples)*self.ratio, dtype=samples.dtype)
        up[::self.ratio] = samples
        up = self.pulse.run(up)
        return up

class Zeros(Upsample):
    def run(self, samples):
        up = np.zeros(len(samples)*self.ratio, dtype=samples.dtype)
        up[::self.ratio] = samples[::1]
        return up
