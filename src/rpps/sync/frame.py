"""Frame sync: where frames start (ARCHITECTURE.md, sync)

    FrameSync   symbols: correlate a known preamble (DVB-S2 SOF), give each frame from its preamble to the next one,
                with the carrier phase the preambles measured taken off
    Delimiter   bits: find the word that comes before each frame (802.11b SFD), give the bits after it

Both publish {"frame": k} at each frame start (rp.profile.publish), so a profile can drive the steps after them frame by
frame (a header that reads the frame's start), and say where each frame came from (rp.profile.anchor).
"""
from types import SimpleNamespace

import numpy as np

from .. import _accel
from ..base import Stage, BitStage
from ..signal.filters.convolve import convolve
from . import track

NATIVE_CORRELATION = 256 # Longest preamble (symbols) correlated directly in C++; longer ones go through the FFT

class FrameSync(Stage):
    """Find frames by a known preamble among timed symbols, and give them whole and the right way round

        frames = symbols + rp.sync.FrameSync(preamble)          # preamble: the known symbols, complex
        frames = symbols + rp.sync.FrameSync(sof, words)        # then one of a set of known continuations (DVB-S2 PLS)

    Block estimation: the normalised correlation with the preamble, |c| / sqrt(E_window E_preamble) (1 for a perfect
    match), peaks above `threshold` and the largest within a preamble's length. With `words` (M, n): the n symbols after
    the preamble are one of M known words (a short header code), and the metric takes the best of them, in phase with the
    preamble, over the whole L + n symbols: random data then hardly ever passes, where a short preamble alone would every
    few frames (DVB-S2: 26 SOF symbols, then 64 PLS symbols from 128 words: random 8PSK data stays under 0.41, a header at
    -2 dB Es/N0 is over 0.5 99 times in 100; threshold 0.5 by default with words, 0.6 without). Each frame goes out when the next
    preamble is found (or the stream is flushed), from its preamble to the next, turned back by the carrier phase: a line
    from the phase its preamble measured to the next one's (block-rate tracking of a slowly turning carrier; a residual
    frequency is for the pilots to take off). A frame longer than `max_length` symbols is cut there.
    Estimates in stream.est[name]: frames, the last preamble's correlation and phase.
    """
    CONTROLS = {}

    def __init__(self, preamble, words=None, threshold=None, max_length=None, name="frame"):
        self.preamble = np.asarray(preamble, dtype=np.complex128)
        self.L = len(self.preamble)
        self.words = None if words is None else np.conj(np.asarray(words, dtype=np.complex128)) # (M, n), conjugated
        self.n = 0 if words is None else self.words.shape[1]
        self._words_ri = None if words is None else np.stack((self.words.real.T, -self.words.imag.T), axis=1).reshape(2*self.n, -1).astype(np.float32) # (re, -im) pairs
        self._words_t = None if words is None else np.ascontiguousarray(self._words_ri.T) # (words, 2n) for the C++ kernel
        self.threshold = float(threshold if threshold is not None else (0.6 if words is None else 0.5))
        self.name = name
        self.max_length = None if max_length is None else int(max_length)
        self._ep = float(np.sum(np.abs(self.preamble)**2))

    def __str__(self):
        return f"FrameSync({self.name})"

    def pos_rate(self, decode):
        return 1.0

    def new_state(self, meta):
        return SimpleNamespace(buf=np.zeros(0, np.complex128), base=None, searched=None, start=None, phase=0.0, k=0,
                               metric=0.0)

    def _peaks(self, y, lo, hi):
        """Preamble starts in y[lo:hi) (indices into y): above threshold and the largest within L"""
        L = self.L
        if hi - lo <= 0 or len(y) < L:
            return [], None
        a, b = max(lo - L, 0), min(hi + L, len(y) - L + 1)
        seg = y[a:b + L - 1]
        rest = y[a:]
        if _accel.native is not None and L <= NATIVE_CORRELATION: # C++: the direct correlation and its normalisation, on several threads
            c, m = _accel.native.preamble_metric(np.ascontiguousarray(seg, dtype=np.complex128), self.preamble, self._ep)
            cs = None
        else:
            c = convolve(seg, np.conj(self.preamble[::-1]), "valid") # = np.correlate(seg, preamble, "valid"), by FFT blocks when long
            cs = np.concatenate(([0.0], np.cumsum(rest.real**2 + rest.imag**2))) # energy so far, from index a on
            e = cs[L:L + len(c)] - cs[:len(c)] # the energy of every window of L
            m = np.abs(c)/np.sqrt(np.maximum(e, 1e-30)*self._ep)
        if self.words is not None:
            m = self._with_words(rest, c, m, cs)
        out = []
        for j in np.nonzero(m > self.threshold)[0]:
            i = a + j
            if not lo <= i < hi:
                continue
            w = m[max(j - L + 1, 0):j + L]
            if m[j] >= w.max():
                out.append((i, float(np.angle(c[j])), float(m[j])))
        return out, (c, a)

    def _with_words(self, y, c, m, cs):
        """The metric over the preamble and its best continuation, where the preamble alone is promising (> 0.3: random
        data passes that one time in ten, a DVB-S2 SOF at -4 dB Es/N0 nearly always). y: the samples from where c starts;
        cs: their cumulative energy (with a leading 0), for the numpy version (the C++ one needs none)"""
        L, n = self.L, self.n
        if _accel.native is not None: # C++: select the candidates, rank the words and normalise, in one call
            return _accel.native.frame_metric(np.ascontiguousarray(y, dtype=np.complex128), np.ascontiguousarray(c, dtype=np.complex128),
                                              np.ascontiguousarray(m, dtype=np.float64), L, self._words_t, 0.3)
        if cs is None:
            cs = np.concatenate(([0.0], np.cumsum(y.real**2 + y.imag**2)))
        out = np.zeros_like(m)
        js = np.nonzero(m > 0.3)[0]
        js = js[js + L + n <= len(y)] # (a continuation that is all there)
        if not len(js):
            return out
        ph = c[js]/np.maximum(np.abs(c[js]), 1e-30)
        best = np.empty(len(js))
        y32, ph32 = y.astype(np.complex64), np.conj(ph).astype(np.complex64)
        cols = np.arange(n)[None, :]
        for k in range(0, len(js), 4096): # the candidates in blocks that stay in cache
            sl = slice(k, k + 4096)
            tail = y32[js[sl][:, None] + L + cols]*ph32[sl, None] # (candidates, n), phase taken off
            # Re(tail . word) for every word: the (re, im) pairs of the tail are float32 rows, one real product ranks the words
            best[sl] = (tail.view(np.float32) @ self._words_ri).max(axis=1)
        e = cs[js + L + n] - cs[js]
        out[js] = (np.abs(c[js]) + best)/np.sqrt(np.maximum(e, 1e-30)*(L + n))
        return out

    def step(self, x, st, meta):
        from ..profile.control import publish, anchor
        if st.base is None:
            st.base = meta.pos
            st.searched = meta.pos
        st.buf = np.concatenate((st.buf, np.asarray(x, dtype=np.complex128)))
        end = st.base + len(st.buf)
        flush = meta.flush
        # Search the positions not searched yet whose neighbourhood is all in
        lo = st.searched - st.base
        hi = len(st.buf) - (self.L if flush else 2*self.L) - self.n + 1
        peaks, _ = self._peaks(st.buf, lo, hi)
        st.searched = max(st.searched, st.base + max(hi, lo))
        out, n_out = [], 0
        for i, ph, m in peaks:
            p = st.base + i
            if st.start is not None and p - st.start < self.L:
                continue
            if st.start is not None:
                n_out += self._emit(st, p, ph, out, meta, n_out, publish, anchor)
            st.start, st.phase, st.metric = p, ph, m
        if st.start is not None and self.max_length and end - st.start > self.max_length + self.L:
            n_out += self._emit(st, st.start + self.max_length, None, out, meta, n_out, publish, anchor)
            st.start = None
        if flush and st.start is not None:
            n_out += self._emit(st, end, None, out, meta, n_out, publish, anchor)
            st.start = None
        # Keep what may still hold a frame or a preamble
        keep_from = st.start if st.start is not None else max(st.searched - self.L, st.base)
        cut = keep_from - st.base
        st.buf, st.base = st.buf[cut:], st.base + cut
        meta.stream.est[self.name] = dict(frames=st.k, metric=st.metric, phase=st.phase)
        return np.concatenate(out).astype(np.complex64) if out else np.zeros(0, np.complex64)

    def _emit(self, st, until, next_phase, out, meta, n_out, publish, anchor):
        """Frame st.start .. until, phase turned back; returns its length"""
        a, b = st.start - st.base, until - st.base
        y = st.buf[a:b]
        c = (self.L - 1)/2
        if next_phase is None:
            turn = np.exp(-1j*st.phase)
        else: # a line from this preamble's phase to the next one's
            d = np.angle(np.exp(1j*(next_phase - st.phase)))
            rate = d/max(until - st.start, 1)
            turn = track.line_phasor(st.phase - rate*c, rate, len(y))
        out.append(y*turn)
        publish(meta, self, {"frame": st.k}, at=st.start, out=n_out)
        anchor(meta, self, st.start, n_out)
        st.k += 1
        return len(y)

class Delimiter(BitStage):
    """Find frames in a bit stream by the word that comes before each one (a start-of-frame delimiter), give the bits
    after it

        frames = bits + rp.sync.Delimiter(word=0xF3A0, bits=16, order="lsb")      # 802.11b SFD, sent LSB first

    word: the delimiter's value, `bits` long, sent most (order "msb") or least significant bit first. errors: bit errors
    a match may have. length (controlled: a header says how long its frame is): after that many bits the frame is over and
    the search starts again; unknown, the bits pass until the next delimiter. Takes hard bits or LLRs, gives the same.
    Estimates in stream.est[name]: frames found.
    """
    CONTROLS = {"length": "live"}

    def __init__(self, word, bits, order="msb", errors=0, name="delimiter"):
        v = int(word, 0) if isinstance(word, str) else int(word)
        idx = np.arange(bits) if order == "lsb" else np.arange(bits - 1, -1, -1)
        self.word = ((v >> idx) & 1).astype(bool)
        self.n, self.errors, self.name = int(bits), int(errors), name
        self.length = None

    def __str__(self):
        return f"Delimiter({self.name})"

    def control(self, length=None):
        self.length = None if length is None else int(length)

    def pos_rate(self, decode):
        return 1.0

    def new_state(self, meta):
        return SimpleNamespace(tail=None, tail_pos=None, cur=None, mode="seek", emitted=0, k=0, seek_from=None)

    def step(self, x, st, meta):
        from ..profile.control import publish, anchor
        x = np.asarray(x)
        n = self.n
        if st.tail is None:
            st.tail, st.tail_pos, st.cur, st.seek_from = x[:0], meta.pos, meta.pos, meta.pos
        buf = np.concatenate((st.tail, x))
        base, end = st.tail_pos, st.tail_pos + len(buf)
        hard = buf < 0 if buf.dtype.kind == "f" else buf.astype(bool)
        starts = self._starts(hard, base)
        cur, out, n_out = st.cur, [], 0
        def emit(a, b):
            nonlocal n_out
            out.append(buf[a - base:b - base])
            n_out += b - a
        while cur < end or (st.mode == "frame" and self.length is not None and st.emitted >= self.length):
            if st.mode == "frame" and self.length is not None:
                take = min(self.length - st.emitted, end - cur)
                if take > 0:
                    emit(cur, cur + take)
                    st.emitted += take
                    cur += take
                if st.emitted < self.length:
                    break
                st.mode, st.seek_from = "seek", cur # The frame is over: look for the next one
                continue
            frm = max(st.seek_from, cur)
            nxt = next((s for s in starts if s >= frm), None) # A word starting here or later
            if st.mode == "frame": # Length unknown: the bits pass up to the next delimiter, kept back until they can't be one
                stop = nxt if nxt is not None else (end if meta.flush else end - (n - 1))
                if stop > cur:
                    emit(cur, stop)
                    st.emitted += stop - cur
                    cur = stop
            if nxt is None:
                if st.mode == "seek":
                    cur = max(cur, end - (n - 1)) # What can't start a delimiter is dropped
                break
            cur = st.seek_from = nxt + n # A frame starts after the word
            st.mode, st.emitted = "frame", 0
            publish(meta, self, {"frame": st.k}, at=cur, out=n_out)
            anchor(meta, self, cur, n_out)
            st.k += 1
        keep = max(min(cur, end) - (n - 1), base)
        st.tail, st.tail_pos, st.cur = buf[keep - base:], keep, cur
        meta.stream.est[self.name] = dict(frames=st.k, mode=st.mode)
        return np.concatenate(out) if out else x[:0]

    def _starts(self, hard, base):
        """Absolute positions where the word starts in hard bits whose first one is at `base`"""
        n = self.n
        if len(hard) < n:
            return []
        w = np.lib.stride_tricks.sliding_window_view(hard, n)
        bad = np.count_nonzero(w != self.word[None, :], axis=1)
        return [base + int(j) for j in np.nonzero(bad <= self.errors)[0]]
