"""Per-burst acquisition: cut a stream of samples into the bursts that rise out of its noise floor

    for b in src(65536) + rp.sync.Burst(win=8, min_s=60e-6, max_s=3e-3) + rx:   # rx runs on each burst on its own
        print(b.meta.pos, len(b))

`data + Burst()` gives a list: the bursts that ended in this block, each a `Data` of its own stream (the stages after it
start over on every burst: loops, timing and frame sync acquire each packet fresh), placed at its absolute position in
the source (`meta.pos`, and `meta.time_of` when the source has a time). `Blocks` and `Pipeline` run what follows on every
burst. A burst still open at the end of a block waits for the blocks that end it; the stream's end closes it.

The detector: the mean power of windows of `win` samples, at absolute positions (independent of the block cut), against
`factor` times the noise floor. The floor is taken from the noise windows alone, of those seen so far (the last `history`):
their mean m is the fixed point of m = mean(windows below `quiet`*m), climbed to from the quietest window, so it stops at the
lowest cluster of window powers (the noise) however many windows the bursts fill; the floor is the `percentile` of the
windows below `quiet`*m. Bursts may follow one another as closely as the air allows: the floor needs some quiet windows,
not most of them (a signal with no gap of a window at all is one burst). Windows of exact zeros (padding, a clean synthetic
signal's gaps) join the noise windows as the quietest of them: a few do not move the floor, but once they are `percentile`%
of the noise windows the floor is 0, and any power is a burst (a clean signal's bursts are all found; a noisy capture padded
with that many zeros is one burst). For one flushed block of a whole capture this is a batch rule: every window of the capture sets the floor.
A stream's first `settle` windows are decided together, once they are all in (or the stream ends): a floor taken from the
first few windows alone would be a burst's own level.
"""
from types import SimpleNamespace

import numpy as np

from ..base.rpps import Pipe
from ..base.types import Category, Rep
from ..base.meta import Stream
from ..base.data import Data


class Burst(Pipe):
    """Bursts of a stream of samples, each a block of its own stream (see the module)

    factor: how far above the noise floor a window's power is to be part of a burst
    win: samples per window (the time resolution of the burst edges)
    min_s, max_s: shortest and longest burst kept [s] (max_s None: any length); a longer one is dropped, and its samples
        are not held past max_s
    percentile: of the noise windows' powers, the noise floor; quiet: how far above their mean a window may be and still
        count as noise; history: windows the floor is taken over (taken again once a 16th of them are new: one block is the
        batch rule exactly, a long stream does not sort its history every block); settle: windows at the stream's start held
        until the floor has them all
    Estimates go in the stream's `est["burst"]`: the floor, bursts emitted, too short, too long."""
    CATEGORY = Category.SYNC

    def __init__(self, factor=6.0, win=100, min_s=0.0, max_s=None, percentile=20, history=1 << 17, settle=512, quiet=3.0):
        if win < 1 or factor <= 0 or quiet <= 2:
            raise ValueError("Burst: win is at least 1 sample, factor positive, quiet above 2 (below it the floor sinks to 0)")
        self.factor, self.win, self.min_s, self.max_s = float(factor), int(win), float(min_s), max_s
        self.percentile, self.history, self.settle, self.quiet = percentile, int(history), int(settle), float(quiet)

    def __str__(self):
        return f"Burst(x{self.factor:g}, {self.win})"

    def takes(self, encode=True):
        return {Rep.SAMPLES}

    def new_state(self, meta):
        return SimpleNamespace(pos=None, rest=None, hist=np.zeros(0), new=0, floor=None, on=False, start=None, held=[], n_held=0, long=False,
                               est={"floor": None, "bursts": 0, "short": 0, "long": 0})

    def __radd__(self, data):
        from ..base.data import _Data, expect
        if not isinstance(data, _Data):
            return NotImplemented
        expect(self, data, self.takes(), True)
        return self.split(data)

    def split(self, data):
        """The bursts that end in this block (a list of Data)"""
        meta = data.meta
        st = meta.state_of(self)
        x = np.asarray(data.data)
        if st.pos is None:
            st.pos, st.rest = meta.pos, x[:0]
        buf = np.concatenate((st.rest, x)) if len(st.rest) else x
        pos0, win = st.pos, self.win # buf[0] is at absolute position pos0, a window boundary
        nw = len(buf)//win
        if len(st.hist) == 0 and nw < self.settle and not data.flush: # The start: wait for enough windows to set the floor
            st.rest = buf
            return []
        used = nw*win
        st.rest = buf[used:] if not data.flush else buf[:0] # (a partial window at the end of the stream is dropped)
        st.pos = pos0 + used
        out = []
        if nw:
            e = (np.abs(buf[:used].astype(np.complex128))**2).reshape(nw, win).mean(axis=1)
            st.hist = np.concatenate((st.hist, e))[-self.history:]
            st.new += nw
            if st.floor is None or st.new*16 >= len(st.hist):
                st.floor, st.new = self.floor(st.hist), 0
            floor = st.floor
            st.est["floor"] = floor
            on = e > self.factor*floor
            d = np.diff(np.r_[int(st.on), on.astype(int)])
            edges = sorted([(int(i)*win, True) for i in np.nonzero(d == 1)[0]] + [(int(i)*win, False) for i in np.nonzero(d == -1)[0]])
            for at, rise in edges:
                if rise:
                    st.start, st.held, st.n_held, st.long = pos0 + at, [], 0, False
                    st.on = True
                else:
                    self._take(st, buf, pos0, at, meta.Fs)
                    out += self._close(st, data, pos0 + at)
            if st.on: # Still open at the end of the block: hold what it has so far
                self._take(st, buf, pos0, used, meta.Fs)
        if data.flush and st.on: # The end of the stream ends the burst
            out += self._close(st, data, st.pos)
        meta.stream.est["burst"] = dict(st.est)
        return out

    def floor(self, e):
        """The noise floor of window powers `e` (see the module)"""
        zeros = int(np.count_nonzero(e <= 0))
        s = np.sort(e[e > 0])
        if not len(s):
            return 0.0
        cs = np.cumsum(s)
        m, n = s[0], 1
        for _ in range(1000): # m grows by at least quiet/2 a step while below the noise (quiet > 2), then settles
            n = max(int(np.searchsorted(s, self.quiet*m)), 1)
            m2 = cs[n - 1]/n
            if m2 <= m*(1 + 1e-9):
                break
            m = m2
        return float(np.percentile(np.r_[np.zeros(zeros), s[:n]], self.percentile))

    def _take(self, st, buf, pos0, end, Fs):
        """Hold the open burst's samples up to buf[end] (nothing more once it is past max_s)"""
        a = max(st.start - pos0, 0)
        if st.long or end <= a:
            return
        st.held.append(buf[a:end])
        st.n_held += end - a
        if self.max_s is not None and Fs and st.n_held/Fs > self.max_s:
            st.long, st.held = True, []

    def _close(self, st, data, end):
        st.on = False
        n = end - st.start
        Fs = data.meta.Fs
        dur = n/Fs if Fs else float(n)
        held, st.held = st.held, []
        if st.long or (self.max_s is not None and dur > self.max_s):
            st.est["long"] += 1
            return []
        if dur < self.min_s:
            st.est["short"] += 1
            return []
        st.est["bursts"] += 1
        samples = np.concatenate(held) if len(held) > 1 else held[0]
        stream = Stream(Fs)
        stream.t0 = data.meta.stream.t0
        stream.pos = st.start # Its first sample is where the burst starts in the source
        m = data.meta
        b = Data(np.ascontiguousarray(samples), flush=True, Fs=m.Fs, sps=m.sps, cf=m.cf, stream=stream, delay=m.delay)
        return [b]
