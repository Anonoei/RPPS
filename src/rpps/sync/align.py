from types import SimpleNamespace

import numpy as np

from ..base import BitStage

class Align(BitStage):
    """Align a bit stream on a marker that repeats every `period` units, and resolve its polarity

        aligned = coded_bits + rp.sync.Align(period=204, pattern=[0xB8] + [0x47]*7)   # DVB-S after Viterbi
        aligned = bits + rp.sync.Align(period=64, markers=(0x7E,))                     # any of a set, every period

    Takes channel or data bits, as LLRs or hard bits, and gives the same form back, starting on a marker and with the
    polarity put right (LLRs negated, bits inverted). This is sync on bits (ARCHITECTURE.md): block estimation, then
    tracking at the block rate.

    What starts each period is either
      markers: a set of unit values, any of which may start any period, or
      pattern: what starts each period in turn, repeating: a value, or a set of values, per period. DVB-S sends 0x47 and,
        every eighth packet, 0xB8. As a set those two cannot tell the polarity (inverted, one is the other); as a pattern
        they can, and the pattern's phase is found too.

    Acquisition: once enough periods are in, every bit offset, pattern phase and polarity is scored by the fraction of
    periods that start as expected, and the best one locks if it reaches `threshold`. Otherwise the oldest period is
    dropped and it tries again as more arrives. What comes before lock is dropped (counted).
    Tracking: each whole period goes out as it completes, and its start is checked. After `misses` periods in a row that
    don't start as expected, the lock is given up and acquisition starts again from there.

    period: units per period. unit: bits per unit (8: bytes, most significant bit first). acquire: periods to score, at
    least one whole pattern. start: "any" period, or "pattern": the output starts on the pattern's first position (a frame
    of periods: DVB-S energy dispersal restarts every 8 packets). Its estimates are in `stream.est[name]`.
    """
    def __init__(self, period, markers=None, pattern=None, unit=8, acquire=None, threshold=0.9, misses=3, polarity=True,
                 name="align", start="any"):
        if start not in ("any", "pattern"):
            raise ValueError(f"start is 'any' or 'pattern', not {start!r}")
        self.start = start
        if (markers is None) == (pattern is None):
            raise ValueError("give markers (any of a set) or pattern (one per period), not both")
        self.period, self.unit = int(period), int(unit)
        rows = [markers] if pattern is None else list(pattern)
        self._allowed = np.zeros((len(rows), 1 << self.unit), dtype=bool) # [pattern position, unit value]: expected?
        for i, r in enumerate(rows):
            self._allowed[i, np.atleast_1d(np.asarray(r, dtype=np.int64))] = True
        self.length = len(rows)
        acquire = max(8, 2*self.length) if acquire is None else int(acquire)
        self.acquire, self.threshold, self.misses = int(acquire), float(threshold), int(misses)
        self.polarity, self.name = bool(polarity), name
        self._w = 1 << np.arange(self.unit - 1, -1, -1) # Unit value from its bits, first bit most significant

    def __str__(self):
        return f"Align({self.name})"

    @property
    def span(self):
        """Bits in one period"""
        return self.period*self.unit

    def new_state(self, meta):
        return SimpleNamespace(pending=None, pend_pos=None, locked=False, flip=False, miss=0, index=0,
                               dropped=0, periods=0, fraction=0.0, boundary=None, locks=0)

    @staticmethod
    def _hard(x):
        return x < 0 if x.dtype.kind == "f" else x.astype(bool)

    def _units(self, bits):
        """Unit values of whole units of hard bits"""
        n = len(bits)//self.unit*self.unit
        return bits[:n].reshape(-1, self.unit).astype(np.int64) @ self._w

    def _expected(self, heads, first):
        """Which of these period starts are as expected, the first being pattern position `first`"""
        idx = (first + np.arange(len(heads))) % self.length
        return self._allowed[idx, heads]

    def _estimate(self, bits):
        """Best (fraction, bit offset, polarity flipped, pattern position of the first period) over a window of hard bits"""
        best = (0.0, 0, False, 0)
        P = self.span
        n = (len(bits) - P)//P # whole periods every offset sees
        for flip in ((False, True) if self.polarity else (False,)):
            b = ~bits if flip else bits
            for off in range(P):
                heads = b[off:off + n*P].reshape(n, P)[:, :self.unit].astype(np.int64) @ self._w
                for first in range(self.length):
                    frac = float(self._expected(heads, first).mean())
                    if frac > best[0]:
                        best = (frac, off, flip, first)
        return best

    def step(self, x, st, meta):
        x = np.asarray(x)
        if st.pending is None:
            st.pending, st.pend_pos = x[:0], meta.pos
        st.pending = np.concatenate((st.pending, x))
        P, out = self.span, []
        while True:
            if not st.locked:
                need = (self.acquire + 1)*P # One period more, so every offset sees `acquire` whole periods
                if len(st.pending) < need:
                    break
                frac, off, flip, first = self._estimate(self._hard(st.pending[:need]))
                if frac >= self.threshold:
                    st.locked, st.flip, st.miss, st.fraction, st.index = True, flip, 0, frac, first
                    st.locks += 1
                    self._drop(st, off)
                    if self.start == "pattern" and first: # On to the pattern's first position: a whole frame of periods
                        skip = self.length - first
                        self._drop(st, skip*P)
                        st.index = 0
                    st.boundary = st.pend_pos
                else:
                    self._drop(st, P) # Not there yet: forget the oldest period and wait for more
                continue
            n = len(st.pending)//P*P
            if not n:
                break
            chunk = st.pending[:n]
            if st.flip:
                chunk = -chunk if chunk.dtype.kind == "f" else ~chunk.astype(bool)
            heads = self._units(self._hard(chunk).reshape(-1, P)[:, :self.unit].reshape(-1))
            ok = self._expected(heads, st.index)
            k = len(ok)
            for i, good in enumerate(ok): # Track: count misses in a row; give the lock up after `misses`
                st.miss = 0 if good else st.miss + 1
                if st.miss >= self.misses:
                    k = max(0, i + 1 - st.miss) # Keep the periods before the misses started
                    break
            out.append(chunk[:k*P])
            st.periods += k
            st.index = (st.index + k) % self.length
            st.pending, st.pend_pos = st.pending[k*P:], st.pend_pos + k*P
            if k < len(ok):
                st.locked, st.miss = False, 0
                continue
            break
        meta.stream.est[self.name] = dict(locked=st.locked, boundary=st.boundary, polarity=-1 if st.flip else 1,
                                          fraction=st.fraction, periods=st.periods, dropped=st.dropped, locks=st.locks,
                                          index=st.index)
        return np.concatenate(out) if out else x[:0]

    @staticmethod
    def _drop(st, n):
        st.pending, st.pend_pos = st.pending[n:], st.pend_pos + n
        st.dropped += n
