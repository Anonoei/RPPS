"""Automatic gain control, with no loop over samples

`AGC`: the level by block estimation (rpps.sync's first class): each window's level (the mean of log|x|, or of |x|),
the gain that brings it to `target` at the window's centre, a line between centres evaluated per sample. `PeakAGC`: the gain from a peak that is followed at once and decays, in closed form.
"""
from types import SimpleNamespace

import numpy as np

from ..base import Stage

class PeakAGC(Stage):
    """Gain from the peak: |x| is followed instantly upward and decays by `decay` per sample, and the signal is scaled so
    that peak sits at `target`. For bursts and on-off keying (Mode S pulses, OOK): the first pulse of a burst sets the level,
    where an average is still settling. decay 0.999 at 2 MS/s holds a peak for about a millisecond.

    The peak p[n] = max(|x[n]|, decay p[n-1]) is d^n max_(k<=n)(|x[k]| d^-k): in logs a running maximum
    (`np.maximum.accumulate`), no loop over the samples, the same peak to rounding."""
    __slots__ = ("target", "decay", "floor")
    CHUNK = 1 << 20 # Samples a running maximum spans: log d^-k grows with k, so long blocks go in pieces (precision)

    def __init__(self, target=1.0, decay=0.999, floor=1e-6):
        if not 0 < decay <= 1:
            raise ValueError(f"PeakAGC: decay in (0, 1], not {decay}")
        self.target, self.decay, self.floor = float(target), float(decay), float(floor)

    def new_state(self, meta):
        return SimpleNamespace(level=0.0)

    def step(self, samples, st, meta):
        x = np.asarray(samples)
        mag = np.abs(x).astype(np.float64)
        peak = np.empty(len(x))
        for a in range(0, len(x), self.CHUNK):
            peak[a:a + self.CHUNK] = self._peak(mag[a:a + self.CHUNK], st)
        return (x*(self.target/np.maximum(peak, self.floor))).astype(x.dtype)

    def _peak(self, m, st):
        if len(m) == 0:
            return m
        g = np.log(self.decay) # <= 0
        k = np.arange(len(m))
        with np.errstate(divide="ignore"):
            c = np.log(m) - k*g # log |x[k]| - k log d
            carried = np.log(st.level) + g if st.level > 0 else -np.inf # the level before, decayed once
        peak = np.exp(k*g + np.maximum(np.maximum.accumulate(c), carried))
        st.level = float(peak[-1])
        return peak


class AGC(Stage):
    """Gain that brings the signal's level to `target`, by block estimation

        block + rp.sync.AGC(1.0)

    First class (rpps.sync). Windows sit at absolute stream positions, `window` seconds (at least `min_samples`). Each
    window's level is exp(mean log|x|) (`log`, the default: a level that does not depend on the signal's shape) or
    mean |x|, and its gain log(target / level) is put at the window's centre; between centres the log gain is a straight
    line, evaluated per sample: no steps at window edges and no loop over the samples. A level step settles within a
    window, with no overshoot after it (a gain is not a drift: nothing carries a rate on). A window of zeros (a gap)
    keeps the gain before it. The gain is held to [min_gain, max_gain]. Until a window with signal has been seen, samples
    pass at `init`. Latency: a window and a half (a sample waits for the centre of the window after it). Estimates in
    stream.est[name]: gain, level, latency (seconds).
    """
    __slots__ = ("target", "window", "min_samples", "log", "min", "max", "init", "name")
    CONTROLS = {}

    def __init__(self, target=1.0, *, window=1e-4, min_samples=64, log=True, min_gain=0.0, max_gain=np.inf, init=1.0,
                 name="agc"):
        # (keywords only: a loop's `mu` given in second place would otherwise be taken for a window)
        self.target, self.window, self.min_samples, self.log = float(target), window, int(min_samples), bool(log)
        self.min, self.max, self.init, self.name = float(min_gain), float(max_gain), float(init), name

    def __str__(self):
        return "AGC"

    def new_state(self, meta):
        # buf: samples held from stream position `base`; k: the next window's index; knot: (centre, log gain) of the last
        # window measured; done: the stream position output so far
        return SimpleNamespace(buf=None, base=None, k=None, knot=None, done=None, level=None, W=None)

    def step(self, x, st, meta):
        x = np.asarray(x)
        cplx = x.dtype.kind == "c"
        if st.buf is None:
            W = max(int(round(self.window*meta.Fs)) if meta.Fs else self.min_samples, self.min_samples)
            st.buf, st.base, st.done, st.W, st.k = np.zeros(0, x.dtype if cplx else np.float64), meta.pos, meta.pos, W, meta.pos//W
        W = st.W
        st.buf = np.concatenate((st.buf, x))
        end = st.base + len(st.buf)
        # Every window ready, all at once: [k W, (k+1) W) clipped to what is held (the stream's first may start late,
        # its last, on a flush, end early)
        last = end//W if not meta.flush else -(-end//W)
        knots_c, knots_g = [], []
        if st.knot is not None:
            knots_c.append(st.knot[0]); knots_g.append(st.knot[1])
        if last > st.k:
            edges = np.clip(np.arange(st.k, last + 1)*W, st.base, end)
            seg = st.buf[edges[0] - st.base:edges[-1] - st.base]
            mag = np.abs(seg).astype(np.float64)
            on = mag > 0
            starts = edges[:-1] - edges[0]
            count = np.add.reduceat(on.astype(np.int64), starts) if len(seg) else np.zeros(0, np.int64)
            if self.log:
                with np.errstate(divide="ignore"):
                    tot = np.add.reduceat(np.where(on, np.log(np.where(on, mag, 1.0)), 0.0), starts)
                level = np.exp(tot/np.maximum(count, 1))
            else:
                level = np.add.reduceat(mag, starts)/np.maximum(edges[1:] - edges[:-1], 1)
            g = np.log(self.target/np.where(count > 0, level, 1.0))
            have = count > 0
            if have.any():
                st.level = float(level[have][-1])
            centres = (edges[:-1] + edges[1:] - 1)/2
            # a window of zeros keeps the gain before it: drop it as a knot
            knots_c += list(centres[have]); knots_g += list(g[have])
            st.k = last
        out = np.zeros(0, st.buf.dtype)
        if knots_c:
            # output up to the newest centre (the samples after it wait for the next window), or all of it on a flush
            upto = end if meta.flush else int(np.floor(knots_c[-1])) + 1
            n = np.arange(st.done, max(upto, st.done))
            gain = np.clip(np.exp(np.interp(n, knots_c, knots_g)), self.min, self.max)
            out = st.buf[st.done - st.base:n[-1] + 1 - st.base]*gain if len(n) else out
            st.done = max(upto, st.done)
            st.knot = (knots_c[-1], knots_g[-1])
        else: # nothing measured yet (no signal in any window so far): what the windows cover passes at `init`
            upto = end if meta.flush else min(end, st.k*W)
            out = st.buf[st.done - st.base:upto - st.base]*self.init
            st.done = max(upto, st.done)
        keep = st.done if st.knot is None else min(st.done, int(np.floor(st.knot[0])))
        st.buf, st.base = st.buf[keep - st.base:], keep
        meta.stream.est[self.name] = dict(gain=float(np.exp(st.knot[1])) if st.knot else self.init, level=st.level,
                                          latency=1.5*W/meta.Fs if meta.Fs else None)
        dt = x.dtype if cplx else np.float32
        return out.astype(dt)
