"""Frame sync: where frames start (rpps.sync)

    FrameSync   symbols: correlate a known preamble (DVB-S2 SOF), give each frame from its preamble to the next one,
                with the carrier phase the preambles measured taken off
    Delimiter   bits: find the word that comes before each frame (802.11b SFD, a CCSDS ASM), give the bits after it (or
                the frame from its word on), for a fixed or a controlled length, with the polarity put right

Both publish {"frame": k} at each frame start (rp.base.control.publish), so a profile can drive the steps after them frame by
frame (a header that reads the frame's start), and say where each frame came from (rp.base.control.anchor).
"""
from types import SimpleNamespace

import numpy as np

from .. import _accel
from ..base import Stage, BitStage
from ..base.kernel import device_functions, has_device_kernels, to_host
from ..math import bits as bitops
from ..signal.filters.convolve import convolve, correlate
from ..base.control import publish, anchor
from . import track

NATIVE_CORRELATION = 256 # Longest preamble (symbols) correlated directly in C++; longer ones go through the FFT
LONG_WORD = 64 # Delimiter: a longer word is matched by FFT correlation rather than window by window

# FrameSync's metric on a device: a thread a position, the correlation with the (conjugated) preamble and the energy of its
# window, |c| / sqrt(E_window E_preamble). Portable CUDA C
FRAME_SOURCE = r"""
extern "C" __global__ void pre_metric(const float2 *y, const long long npos, const float2 *pre, const int L, const float ep,
                                      float2 *c, float *m) {
    const long long j = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= npos) return;
    float cr = 0.0f, ci = 0.0f, e = 0.0f;
    for (int l = 0; l < L; l++) {
        const float2 v = y[j + l], p = pre[l];
        cr += v.x*p.x - v.y*p.y; ci += v.x*p.y + v.y*p.x;
        e += v.x*v.x + v.y*v.y;
    }
    c[j] = make_float2(cr, ci);
    m[j] = sqrtf(cr*cr + ci*ci)/sqrtf(fmaxf(e, 1e-30f)*ep);
}
// For candidate k at js[k]: the n symbols after the preamble turned back by its correlation's phase, as (re, im) floats (a
// row of the words' matrix product), and the energy of all L + n. A block a candidate, its threads over the symbols
extern "C" __global__ void pre_tails(const float2 *y, const long long *js, const float2 *c, const int L, const int n,
                                     float *tail, float *e) {
    __shared__ float part[128];
    const long long k = blockIdx.x, j = js[k];
    const float2 cc = c[j];
    const float a = fmaxf(sqrtf(cc.x*cc.x + cc.y*cc.y), 1e-30f), pr = cc.x/a, pi = -cc.y/a;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < L + n; i += blockDim.x) {
        const float2 v = y[j + i];
        acc += v.x*v.x + v.y*v.y;
        if (i >= L) {
            tail[k*2*n + 2*(i - L)] = v.x*pr - v.y*pi;
            tail[k*2*n + 2*(i - L) + 1] = v.x*pi + v.y*pr;
        }
    }
    part[threadIdx.x] = acc;
    __syncthreads();
    for (int w = blockDim.x/2; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) e[k] = part[0];
}
"""

def word_errors(n_bits, false_alarm=1e-6, positions=1, polarity=False):
    """The most bit errors a match of an `n_bits` word may have while random bits stay under `false_alarm` matches over
    `positions` places searched (a frame's bits, when the search restarts every frame): the largest e with
    positions * P(Binomial(n_bits, 1/2) <= e) (twice that with `polarity`, the inverted word matched too) under it.
    -1 when even an exact match is too likely (a short word over many places). A 32-bit ASM searched over a CCSDS frame
    (8160 places) allows 1 at 1e-3; an 800-bit word (100 bytes) 314 at 1e-9, so its own channel errors hardly ever matter"""
    from math import log
    from ..math.prob import log_binom_half_cdf
    n = int(n_bits)
    budget = log(false_alarm) - log(max(1, int(positions))) - (log(2) if polarity else 0.0)
    ok = np.nonzero(np.atleast_1d(log_binom_half_cdf(n, np.arange(n + 1))) <= budget)[0]
    return int(ok[-1]) if len(ok) else -1

class FrameSync(Stage):
    """Find frames by a known preamble among timed symbols, and give them whole and the right way round

        frames = symbols + rp.sync.FrameSync(preamble)          # preamble: the known symbols, complex
        frames = symbols + rp.sync.FrameSync(sof, words)        # then one of a set of known continuations (DVB-S2 PLS)
        frames = levels + rp.sync.FrameSync({"bs_voice": a, "bs_data": b}, before=54, length=132, scale=True)  # DMR

    Block estimation: the normalised correlation with the preamble, |c| / sqrt(E_window E_preamble) (1 for a perfect
    match), peaks above `threshold` and the largest within a preamble's length. With `words` (M, n): the n symbols after
    the preamble are one of M known words (a short header code), and the metric takes the best of them, in phase with the
    preamble, over the whole L + n symbols: random data then hardly ever passes, where a short preamble alone would every
    few frames (DVB-S2: 26 SOF symbols, then 64 PLS symbols from 128 words: random 8PSK data stays under 0.41, a header at
    -2 dB Es/N0 is over 0.5 99 times in 100; threshold 0.5 by default with words, 0.6 without). Each frame goes out when the next
    preamble is found (or the stream is flushed), from its preamble to the next, turned back by the carrier phase: a line
    from the phase its preamble measured to the next one's (block-rate tracking of a slowly turning carrier; a residual
    frequency is for the pilots to take off). A frame longer than `max_length` symbols is cut there.
    A unique word known as bits (a long one after demodulation, before the FEC) is found here rather than among bits when
    the carrier leaves an M-fold ambiguity: its phase is measured, so a frame turned by 90 degrees is put right too (a
    bit search only corrects an inversion). pending: the preamble comes later (`set_template`), as a profile makes it
    from such a word. Over L symbols random data correlates about 1/sqrt(L) (0.05 for a 400-symbol word), so a long word
    may use a threshold well under the default.

    preamble: the symbols, or alternatives of one length: {name: symbols} or {name: (symbols, {field: value})} (sync
        words that say what follows: DMR's voice / data, base / mobile; Zadoff-Chu roots and shifts). Each is
        correlated; the best names the frame, published as `word` with its fields, for controls to drive steps by.
    before, length: the frame starts `before` symbols ahead of its preamble (a sync word mid-burst) and is `length`
        symbols long (a controlled parameter too: a header says it); without a length a frame runs to the next one.
    segments: the preamble correlated in that many parts, added without their phases: a carrier offset turns a long
        word (a Zadoff-Chu sequence moves its peak with it) and a coherent sum over the whole word loses it (a burst's
        own offset, 3 kHz at 512 kBd, turns a 64-symbol word by 145 degrees). "auto": one part a 16 symbols, at most 4
        (1 with header words): what a profile's frame step on a word takes unless it says.
    count: frames to find in a stream, then no more searching (None: all; a burst's runner: 1, as data later in the burst
        can pass for the word).
    cfo: take the carrier frequency its preamble measured off each frame too (within +-1/L cycles a symbol: a larger
        offset is for a carrier step before), not only its phase.
    scale: divide each frame by the gain its preamble measured, so levels come out as the preamble's (4FSK +-1, +-3).
    Real levels (a discriminator's output) stay real: the phase is a polarity, a frame upside down is turned over.
    polarity: an inverted (turned) match counts (an unknown carrier phase, a discriminator of unknown sense); False:
        only matches in phase, where the sense is known and alternatives are each other inverted (DMR voice / data).
    classify: constellation names; each frame's symbols after its preamble are scored against them (rpps.sync.acquire)
        at the Es/N0 the preamble measured, and the best published as `modulation` (when it leads by `margin` nats a
        symbol over at least `min_symbols`; otherwise the last one stays, reported).

    Each frame start publishes {"frame": k, "metric", "phase", "scale", "esn0" (dB: the frame's power less the noise
    the residual after its preamble measured, over that noise), "cfo" (cycles a symbol, from the phase between the preamble's halves; complex input), "word" and its fields
    (alternatives), "modulation" (classify)}. Estimates in stream.est[name]: the same of the last frame.
    """
    CONTROLS = {"length": "live"}

    def __init__(self, preamble, words=None, threshold=None, max_length=None, name="frame", pending=False, device=False,
                 before=0, length=None, segments=1, scale=False, classify=None, margin=0.1, min_symbols=64, polarity=True,
                 cfo=False, count=None):
        if preamble is None and not pending:
            raise ValueError("FrameSync: a preamble, or pending=True (set_template gives it later)")
        if pending and words is not None:
            raise ValueError("FrameSync: words follow a preamble given now, not a pending one")
        self.preamble, self.L, self._ep = None, 0, 0.0
        self.pres, self.names, self.fields = None, None, None
        self._auto = segments == "auto" # (resolved when the preamble is known: its length says)
        self.segments = 1 if self._auto else int(segments)
        if self.segments < 1:
            raise ValueError("FrameSync: segments >= 1")
        if preamble is not None:
            self.set_template(preamble)
        self.words = None if words is None else np.conj(np.asarray(words, dtype=np.complex128)) # (M, n), conjugated
        self.n = 0 if words is None else self.words.shape[1]
        self._words_ri = None if words is None else np.stack((self.words.real.T, -self.words.imag.T), axis=1).reshape(2*self.n, -1).astype(np.float32) # (re, -im) pairs
        self._words_t = None if words is None else np.ascontiguousarray(self._words_ri.T) # (words, 2n) for the C++ kernel
        self.threshold = float(threshold if threshold is not None else (0.6 if words is None else 0.5))
        self.name = name
        self.max_length = None if max_length is None else int(max_length)
        self.device, self._xp = bool(device), None # device: the metric on the GPU of rpps._accel.device() (numpy without one)
        self.before = int(before)
        if self.before < 0:
            raise ValueError("FrameSync: before >= 0 (symbols of the frame ahead of its preamble)")
        self.length = None
        self.control(length=length)
        self.scale = bool(scale)
        self.classify = None if not classify else list(classify)
        self.margin, self.min_symbols = float(margin), int(min_symbols)
        self.polarity = bool(polarity)
        self.cfo = bool(cfo)
        self.count = None if count is None else int(count)
        self._check()

    def _check(self):
        plain = self.pres is None or self._plain
        if self.words is not None and not plain:
            raise ValueError("FrameSync: header words follow one preamble correlated whole (no alternatives, no segments)")
        if self.pres is not None and self.segments > self.L:
            raise ValueError(f"FrameSync: {self.segments} segments of a {self.L}-symbol preamble")

    def control(self, length=None):
        self.length = None if length is None else int(length)
        if self.length is not None and self.length <= self.before:
            raise ValueError(f"FrameSync: a frame of {self.length} symbols does not reach its preamble ({self.before} before it)")

    def set_template(self, preamble, first=0, span=None):
        """The known symbols that start each frame, or {name: symbols or (symbols, fields)} of one length (a profile makes
        them from words, through its own transmit side down to the symbol mapping: rpps.profile.Profile._template; first
        and span are unused, the symbols being one a sample)"""
        if isinstance(preamble, dict):
            items = list(preamble.items())
        else:
            items = [(None, preamble)]
        names, pres, fields = [], [], []
        for k, v in items:
            f = {}
            if isinstance(v, tuple):
                v, f = v
            names.append(k)
            pres.append(np.asarray(v, dtype=np.complex128).reshape(-1))
            fields.append(dict(f))
        if not pres or not len(pres[0]):
            raise ValueError("FrameSync: an empty preamble")
        if len({len(p) for p in pres}) != 1:
            raise ValueError(f"FrameSync: alternatives of one length, not {sorted({len(p) for p in pres})}")
        self.pres, self.names, self.fields = np.stack(pres), names, fields
        if getattr(self, "_auto", False):
            words = getattr(self, "words", None) is not None
            self.segments = 1 if words else min(4, max(1, len(pres[0])//16))
        self.preamble = self.pres[0]
        self.L = len(self.preamble)
        self._eps = np.sum(np.abs(self.pres)**2, axis=1)
        self._ep = float(self._eps[0])
        bounds = np.linspace(0, self.L, min(self.segments, self.L) + 1).round().astype(int)
        self._segs = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        if hasattr(self, "words"):
            self._check()

    def __str__(self):
        return f"FrameSync({self.name})"

    def pos_rate(self, decode):
        return 1.0

    def new_state(self, meta):
        return SimpleNamespace(buf=np.zeros(0, np.complex128), base=None, searched=None, start=None, phase=0.0, k=0,
                               metric=0.0, p=None, info=None, queue=[], floor=None, last=None, real=None, modulation=None,
                               est={}, found=0)

    @property
    def _plain(self):
        return len(self.pres) == 1 and self.segments == 1 and self.polarity

    def _peaks(self, y, lo, hi):
        """Preamble starts in y[lo:hi) (indices into y): above threshold and the largest within L, as (index, phase,
        metric, alternative)"""
        L = self.L
        if hi - lo <= 0 or len(y) < L:
            return [], None
        if not self._plain:
            return self._peaks_alt(y, lo, hi), None
        a, b = max(lo - L, 0), min(hi + L, len(y) - L + 1)
        if self.device:
            if self._xp is None:
                self._xp = _accel.device()[0]
            if has_device_kernels(self._xp):
                return [(i, ph, m, 0) for i, ph, m in self._peaks_device(self._xp, y, a, b, lo, hi)], None
        seg = y[a:b + L - 1]
        rest = y[a:]
        if _accel.native is not None and L <= NATIVE_CORRELATION: # C++: the direct correlation and its normalisation, on several threads
            c, m = _accel.native.preamble_metric(np.ascontiguousarray(seg, dtype=np.complex128), self.preamble, self._ep)
            cs = None
        else:
            c = correlate(seg, self.preamble)
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
                out.append((i, float(np.angle(c[j])), float(m[j]), 0))
        return out, (c, a)

    def _peaks_alt(self, y, lo, hi):
        """`_peaks` over alternatives and segments: per alternative the segments' correlations, their magnitudes added
        (their sum's phase is the frame's), normalised by the window's energy; the best alternative at each position"""
        L = self.L
        a, b = max(lo - L, 0), min(hi + L, len(y) - L + 1)
        npos = b - a
        if npos <= 0:
            return []
        seg = y[a:b + L - 1]
        cs = np.concatenate(([0.0], np.cumsum(seg.real**2 + seg.imag**2)))
        e = np.maximum(cs[L:L + npos] - cs[:npos], 1e-30)
        best_m = np.full(npos, -1.0)
        best_c = np.zeros(npos, np.complex128)
        best_a = np.zeros(npos, np.int64)
        for k, p in enumerate(self.pres):
            mag = np.zeros(npos)
            tot = np.zeros(npos, np.complex128)
            for s0, s1 in self._segs:
                cseg = correlate(seg[s0:s0 + npos + (s1 - s0) - 1], p[s0:s1])[:npos]
                mag += np.abs(cseg) if self.polarity else np.maximum(cseg.real, 0.0) # (the phase known: in phase only)
                tot += cseg
            m = mag/np.sqrt(e*self._eps[k])
            better = m > best_m
            best_m[better], best_c[better], best_a[better] = m[better], tot[better], k
        out = []
        for j in np.nonzero(best_m > self.threshold)[0]:
            i = a + j
            if not lo <= i < hi:
                continue
            if best_m[j] >= best_m[max(j - L + 1, 0):j + L].max():
                out.append((i, float(np.angle(best_c[j])), float(best_m[j]), int(best_a[j])))
        return out

    def _peaks_device(self, xp, y, a, b, lo, hi):
        """`_peaks` with the metric on a device, in single precision: the correlation and window energy of every position
        (a thread each), the words of the promising ones ranked in one matrix product; the positions over the threshold and
        their neighbourhoods (for the largest within L) come back, the rest stays there"""
        L, n = self.L, self.n
        fs = device_functions(xp, FRAME_SOURCE, ("pre_metric", "pre_tails"))
        f = fs["pre_metric"]
        rest = xp.asarray(np.ascontiguousarray(y[a:], dtype=np.complex64))
        npos = b - a
        c = xp.empty(npos, dtype=np.complex64)
        m = xp.empty(npos, dtype=np.float32)
        pre = self._dconst(xp, "pre", lambda: np.conj(self.preamble).astype(np.complex64))
        T = 256
        f(((npos + T - 1)//T,), (T,), (rest, np.int64(npos), pre, np.int32(L), np.float32(self._ep), c, m))
        if self.words is not None:
            js = xp.nonzero((m > 0.3) & (xp.arange(npos) + L + n <= len(rest)))[0]
            out = xp.zeros_like(m)
            if len(js):
                K = len(js)
                tail = xp.empty((K, 2*n), dtype=np.float32) # (re, im) pairs of each candidate's continuation, phase taken off
                e = xp.empty(K, dtype=np.float32)
                fs["pre_tails"]((K,), (128,), (rest, js.astype(np.int64), c, np.int32(L), np.int32(n), tail, e))
                best = (tail @ self._dconst(xp, "words", lambda: self._words_ri)).max(axis=1)
                out[js] = (xp.abs(c[js]) + best)/xp.sqrt(xp.maximum(e, 1e-30)*(L + n))
            m = out
        j = xp.nonzero(m > self.threshold)[0]
        if not len(j):
            return []
        nb = xp.clip(j[:, None] + xp.arange(-(L - 1), L)[None, :], 0, npos - 1) # (the neighbourhood, clipped as numpy slices)
        top = m[j] >= m[nb].max(axis=1)
        j = j[top]
        jj, ph, mm = to_host(j), to_host(xp.angle(c[j])), to_host(m[j])
        return [(a + int(k), float(p), float(v)) for k, p, v in zip(jj, ph, mm) if lo <= a + int(k) < hi]

    def _dconst(self, xp, key, make):
        cache = self.__dict__.setdefault("_dconsts", {})
        v = cache.get(key)
        if v is None:
            v = cache[key] = xp.asarray(make())
        return v

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
        if self.preamble is None:
            raise ValueError(f"{self}: no preamble yet (a profile makes it from its word: set_template)")
        x = np.asarray(x)
        if st.real is None:
            st.real = x.dtype.kind != "c" # Levels (a discriminator's): frames stay real, their phase a polarity
        if st.base is None:
            st.base = st.searched = st.floor = meta.pos
        st.buf = np.concatenate((st.buf, x.astype(np.complex128)))
        end = st.base + len(st.buf)
        flush = meta.flush
        L, before = self.L, self.before
        # Search the positions not searched yet whose neighbourhood is all in
        lo = st.searched - st.base
        hi = len(st.buf) - (L if flush else 2*L) - self.n + 1
        peaks, _ = self._peaks(st.buf, lo, hi)
        st.searched = max(st.searched, st.base + max(hi, lo))
        out, n_out = [], 0
        for i, ph, m, alt in peaks:
            p = st.base + i
            if st.last is not None and p - st.last < L:
                continue
            if self.count is not None and st.found >= self.count: # (a burst's one frame: no more searching in it)
                break
            st.found += 1
            f = p - before
            if f < st.floor: # Its start is gone: before the stream, or inside the last frame of a fixed length
                continue
            st.last = p
            info = self._measure(st.buf[i:i + L], alt, ph, m, st.real)
            if self.length is None:
                if st.start is not None:
                    n_out += self._emit(st, f, info["phase"], out, meta, n_out)
                st.start, st.p, st.info = f, p, info
                st.floor = f
            else:
                st.queue.append((f, f + self.length, p, info))
                st.floor = f + self.length
        while st.queue and (st.queue[0][1] <= end or flush): # Frames of a fixed length that are all in (or all there is)
            f, g, p, info = st.queue.pop(0)
            st.start, st.p, st.info = f, p, info
            n_out += self._emit(st, min(g, end), None, out, meta, n_out)
            st.start = None
        if st.start is not None and self.max_length and end - st.start > self.max_length + L + before:
            n_out += self._emit(st, st.start + self.max_length, None, out, meta, n_out)
            st.start = None
        if flush and st.start is not None:
            n_out += self._emit(st, end, None, out, meta, n_out)
            st.start = None
        # Keep what may still hold a frame or a preamble (and the symbols ahead of one)
        held = [st.start] if st.start is not None else []
        held += [q[0] for q in st.queue]
        keep_from = max(min(held + [st.searched - L - before]), st.base)
        cut = keep_from - st.base
        st.buf, st.base = st.buf[cut:], st.base + cut
        meta.stream.est[self.name] = dict(st.est, frames=st.k, delay=meta.delay) # (delay: its input's, for a time of its positions)
        if not out:
            return np.zeros(0, np.float32 if st.real else np.complex64)
        return np.concatenate(out).astype(np.float32 if st.real else np.complex64)

    def _measure(self, y, alt, ph, metric, real):
        """What the preamble at y (L symbols) says: {metric, phase, scale (the gain), esn0 (dB, the residual after the
        known symbols), cfo (cycles a symbol: the phase between its halves; complex symbols), word and its fields}"""
        p, ep, L = self.pres[alt], float(self._eps[alt]), self.L
        info = {"metric": float(metric)}
        if real:
            g = float(np.real(np.vdot(p, y)))/ep
            r = y.real - g*p.real
            info["phase"] = 0.0 if g >= 0 else float(np.pi)
        else:
            h = L//2
            cfo = 0.0
            if h:
                c1, c2 = np.vdot(p[:h], y[:h]), np.vdot(p[h:], y[h:])
                cfo = float(np.angle(c2*np.conj(c1)))/(2*np.pi*L/2)
                info["cfo"] = cfo
            q = p*np.exp(2j*np.pi*cfo*(np.arange(L) - (L - 1)/2))
            g = np.vdot(q, y)/ep
            r = y - g*q
            info["phase"] = float(np.angle(g))
        sig = abs(g)**2*ep/L
        info["scale"] = float(abs(g))
        info["_noise"] = max(float(np.sum(np.abs(r)**2))/max(L - 1, 1), sig*1e-12, 1e-300) # (per symbol, as received)
        if self.names[alt] is not None:
            info["word"] = self.names[alt]
            info.update(self.fields[alt])
        return info

    def _emit(self, st, until, next_phase, out, meta, n_out):
        """Frame st.start .. until, phase turned back (real: turned over when upside down), scaled; returns its length"""
        a, b = st.start - st.base, until - st.base
        y = st.buf[a:b]
        info = dict(st.info)
        noise = info.pop("_noise")
        es = float(np.mean(np.abs(y)**2)) - noise if len(y) else 0.0 # The frame's symbols: their power less the noise
        info["esn0"] = float(10*np.log10(max(es, noise*1e-12)/noise))
        if st.real:
            y = y.real*(1.0 if np.cos(info["phase"]) >= 0 else -1.0)
        else:
            c = (st.p - st.start) + (self.L - 1)/2 # The preamble's centre, where its phase was measured
            if self.cfo and "cfo" in info: # The frequency the preamble measured, from its centre on
                turn = np.exp(-1j*(info["phase"] + 2*np.pi*info["cfo"]*(np.arange(len(y)) - c)))
            elif next_phase is None:
                turn = np.exp(-1j*info["phase"])
            else: # a line from this preamble's phase to the next one's
                d = np.angle(np.exp(1j*(next_phase - info["phase"])))
                rate = d/max(until - st.p, 1)
                turn = track.line_phasor(info["phase"] - rate*c, rate, len(y))
            y = y*turn
        if self.scale:
            y = y/max(info["scale"], 1e-30)
        fields = {"frame": st.k, **info}
        if self.classify:
            mod = self._classify(y, st.p - st.start, info["esn0"], st, meta)
            if mod is not None:
                fields["modulation"] = mod
        out.append(y)
        publish(meta, self, fields, at=st.start, out=n_out)
        anchor(meta, self, st.start, n_out)
        st.est = fields
        st.k += 1
        return len(y)

    def _classify(self, y, off, esn0, st, meta):
        """The constellation of `classify` this frame's symbols after (and before) its preamble fit best, at the preamble's
        Es/N0; the last one when the frame is too short or no candidate leads by `margin`"""
        from .acquire import candidates, score
        from ..base.control import of
        pay = np.concatenate((y[:off], y[off + self.L:])).astype(np.complex128)
        if len(pay) < self.min_symbols:
            of(meta.stream).reports.append({"what": "unclassified", "step": self.name, "frame": st.k,
                                            "note": f"{len(pay)} symbols (classify needs {self.min_symbols})"})
            return st.modulation
        pay = pay/np.sqrt(max(np.mean(np.abs(pay)**2), 1e-30))
        s = {n: score(pay, pts, esn0=esn0) for n, pts in candidates(self.classify).items()}
        ranked = sorted(s, key=s.get, reverse=True)
        if len(ranked) == 1 or s[ranked[0]] - s[ranked[1]] >= self.margin:
            st.modulation = ranked[0]
        else:
            of(meta.stream).reports.append({"what": "unclassified", "step": self.name, "frame": st.k,
                                            "note": f"{ranked[0]} leads {ranked[1]} by {s[ranked[0]] - s[ranked[1]]:.3f}"})
        return st.modulation

class Delimiter(BitStage):
    """Find frames in a bit stream by the word that comes before each one (a start-of-frame delimiter), give the bits
    after it

        frames = bits + rp.sync.Delimiter(word=0xF3A0, bits=16, order="lsb")      # 802.11b SFD, sent LSB first
        frames = bits + rp.sync.Delimiter(0x1ACFFC1D, 32, length=8160, keep=True) # CCSDS: the ASM and its frame, whole

    word: the delimiter's value, `bits` long, sent most (order "msb") or least significant bit first; or the bits
    themselves (an array, in the order sent), or a `sequence` definition of type word (rp.sequence.load). errors: bit
    errors a match may have (`word_errors` gives the most a false-alarm rate allows; a word longer than 64 bits is matched
    by FFT correlation). length: bits in a frame after the word; after them the frame is over and the search starts
    again (also a controlled parameter: a header says how long its frame is); unknown, the bits pass until the next
    delimiter. keep: give each frame from its word on (so a `rp.code.Marker` after it can strip and check the word) rather
    than from the bit after it. polarity: match the inverted word too, and give such a frame inverted back (LLRs negated),
    for links whose sense an unknown carrier phase can flip. Takes hard bits or LLRs, gives the same.
    Estimates in stream.est[name]: frames found, polarity of the last.
    """
    CONTROLS = {"length": "live"}

    def __init__(self, word, bits=None, order="msb", errors=0, name="delimiter", length=None, keep=False, polarity=False):
        from ..sequence.word import Word, word_bits
        self.word = word.bits if isinstance(word, Word) else word_bits(word, bits, order)
        self.n, self.errors, self.name = len(self.word), int(errors), name
        self.length = None if length is None else int(length)
        self.keep, self.polarity = bool(keep), bool(polarity)

    def __str__(self):
        return f"Delimiter({self.name})"

    def control(self, length=None):
        self.length = None if length is None else int(length)

    def pos_rate(self, decode):
        return 1.0

    def new_state(self, meta):
        return SimpleNamespace(tail=None, tail_pos=None, cur=None, mode="seek", emitted=0, k=0, seek_from=None, flip=False)

    def _total(self):
        """Bits a frame gives out: its length, and the word too when kept"""
        return None if self.length is None else self.length + (self.n if self.keep else 0)

    def step(self, x, st, meta):
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
            seg = buf[a - base:b - base]
            if st.flip:
                seg = -seg if seg.dtype.kind == "f" else ~seg
            out.append(seg)
            n_out += b - a
        total = self._total()
        while cur < end or (st.mode == "frame" and total is not None and st.emitted >= total):
            if st.mode == "frame" and total is not None:
                take = min(total - st.emitted, end - cur)
                if take > 0:
                    emit(cur, cur + take)
                    st.emitted += take
                    cur += take
                if st.emitted < total:
                    break
                st.mode, st.seek_from = "seek", cur # The frame is over: look for the next one
                continue
            frm = max(st.seek_from, cur)
            nxt = next((s for s in starts if s[0] >= frm), None) # A word starting here or later
            if st.mode == "frame": # Length unknown: the bits pass up to the next delimiter, kept back until they can't be one
                stop = nxt[0] if nxt is not None else (end if meta.flush else end - (n - 1))
                if stop > cur:
                    emit(cur, stop)
                    st.emitted += stop - cur
                    cur = stop
            if nxt is None:
                if st.mode == "seek":
                    cur = max(cur, end - (n - 1)) # What can't start a delimiter is dropped
                break
            at, st.flip = nxt
            cur = at if self.keep else at + n # A frame starts at the word (kept) or after it
            st.seek_from = at + n
            st.mode, st.emitted = "frame", 0
            publish(meta, self, {"frame": st.k}, at=cur, out=n_out)
            anchor(meta, self, cur, n_out)
            st.k += 1
        keep = max(min(cur, end) - (n - 1), base)
        st.tail, st.tail_pos, st.cur = buf[keep - base:], keep, cur
        meta.stream.est[self.name] = dict(frames=st.k, mode=st.mode, polarity=-1 if st.flip else 1)
        return np.concatenate(out) if out else x[:0]

    def _starts(self, hard, base):
        """(absolute position, inverted) where the word starts in hard bits whose first one is at `base`"""
        n = self.n
        if len(hard) < n:
            return []
        if n > LONG_WORD: # Mismatches from the +-1 correlation, by FFT blocks: no window x word matrix
            pm = lambda b: 1.0 - 2.0*b
            bad = np.rint((n - convolve(pm(hard), pm(self.word)[::-1], "valid"))/2).astype(np.int64)
        else: # (a shift register and a popcount a bit, in C++)
            bad = bitops.distance(hard, self.word)
        found = {int(j): False for j in np.nonzero(bad <= self.errors)[0]}
        if self.polarity:
            for j in np.nonzero((n - bad) <= self.errors)[0]:
                found.setdefault(int(j), True)
        return [(base + j, flip) for j, flip in sorted(found.items())]
