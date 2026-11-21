"""A phase-locked tone: the oscillator a pilot sets (FM stereo's 19 kHz, ATSC's pilot), and the signal mixed with it"""
from collections import deque
from types import SimpleNamespace

import numpy as np

from ..base.rpps import Stage


class ToneLock(Stage):
    """Lock an oscillator to a steady tone near `f0` Hz, and mix the stream with its `harmonic`

    A block-rate PLL: every `win` samples (windows at absolute positions, so the block cut does not matter) the tone's phase
    is measured against the oscillator, which then corrects its phase (kp) and frequency (ki). Output, from the oscillator
    as it was over each window:
        complex input: x exp(-j h phase), the tone (h = 1) moved to DC (ATSC: the pilot locks the carrier)
        real input:    2 x cos(h phase), a product detector at the harmonic (FM stereo: h = 2, the L-R subcarrier at 38 kHz)

    min_level: below this tone level the loop holds (a station with no pilot leaves noise there); lock_level and lock_err
    [rad] decide `locked`. min_part: at the stream's end, a last window this full still corrects. The state: `level` (the
    tone's amplitude: 2|z| for a real input, |z| for complex), `err` [rad], `locked`, and the last `history` windows'
    `errs` and `levels`.
    """
    def __init__(self, f0, win=16384, kp=0.4, ki=0.05, harmonic=1, min_level=0.0, lock_level=0.0, lock_err=0.5,
                 min_part=0.25, history=4096):
        if win < 1:
            raise ValueError("ToneLock: win is at least one sample")
        self.f0, self.win, self.kp, self.ki, self.harmonic = float(f0), int(win), kp, ki, harmonic
        self.min_level, self.lock_level, self.lock_err, self.min_part, self.history = min_level, lock_level, lock_err, min_part, history

    def __str__(self):
        return f"ToneLock({self.f0:g} Hz x{self.harmonic})"

    def new_state(self, meta):
        return SimpleNamespace(phase=0.0, freq=2*np.pi*self.f0/meta.Fs, k=0, acc=0j, level=0.0, err=0.0, locked=False,
                               errs=deque(maxlen=self.history), levels=deque(maxlen=self.history))

    def step(self, x, st, meta):
        x = np.asarray(x)
        real = x.dtype.kind != "c"
        out = np.empty(len(x), np.float32 if real else np.complex64)
        i, win = 0, self.win
        while i < len(x):
            n = min(len(x) - i, win - st.k)
            c = x[i:i + n]
            ph = st.phase + st.freq*(np.arange(n) if st.k == 0 else st.k + np.arange(n)) # st.phase: at the window's start
            rot = np.exp(-1j*ph)
            if real:
                st.acc += np.sum(c*rot)
                out[i:i + n] = 2*c*np.cos(self.harmonic*ph)
            elif self.harmonic == 1:
                d = c*rot
                st.acc += np.sum(d)
                out[i:i + n] = d
            else:
                st.acc += np.sum(c*rot)
                out[i:i + n] = c*np.exp(-1j*self.harmonic*ph)
            st.k += n
            i += n
            if st.k == win:
                self._correct(st, real)
        if meta.flush and st.k and st.k >= self.min_part*win:
            self._correct(st, real)
        return out

    def _correct(self, st, real):
        """End of a window of st.k samples: measure, correct, and move the oscillator to the next window's start"""
        n = st.k
        z = st.acc/n
        st.level = float((2 if real else 1)*abs(z))
        st.err = float(np.angle(z))
        st.errs.append(st.err)
        st.levels.append(st.level)
        if st.level > self.min_level:
            st.freq += self.ki*st.err/n
            st.phase += self.kp*st.err
        st.locked = st.level > self.lock_level and abs(st.err) < self.lock_err
        st.phase = (st.phase + st.freq*n) % (2*np.pi)
        st.k, st.acc = 0, 0j
