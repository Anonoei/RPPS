"""A preamble at the start of a burst: where it is, and the carrier offset its repetition gives

    bursts = src(65536) + rp.sync.Burst(win=100, min_s=30e-6)
    for b in bursts + rp.sync.Preamble(ltf_symbol, repeat=64, offset=-192, correct=True):  # 802.11a/g: STF start, CFO off
        ...

Found once per stream: a stream is one burst after `rp.sync.Burst`. Until it is found the samples are held (up to `search`
samples of candidate starts, or the stream's end); then the output starts at the preamble (plus `offset`), placed at its
absolute position, and the rest of the stream passes. Not found: nothing comes out, and `est[name]["found"]` is False.
"""
from types import SimpleNamespace

import numpy as np

from ..base.rpps import Stage


class Preamble(Stage):
    """Find a burst's preamble: a known waveform, or a repeated one (Schmidl & Cox)

    template: the preamble's samples at the stream's rate (an 802.11 LTF symbol, an LTE PSS); its start is where the
        normalized correlation |sum x t*| / sqrt(sum |x|^2 sum |t|^2) (1: the template exactly, scaled and turned) is
        highest, if at least `gate`. A template that repeats in the preamble matches each copy: give all the copies
        (802.11's two LTF symbols). A carrier offset turns the preamble over its length and lowers the metric: about
        sinc(offset x length / Fs) (40 kHz over 128 samples at 20 MS/s: 0.89). None: the repetition alone finds it, Schmidl & Cox's C over `window` samples at lag
        `repeat`: among the starts where |C|/P reaches `gate`, the first where |C| is at its plateau (97% of its peak;
        |C|/P alone rises before the window is all in the preamble)
    repeat: the preamble repeats after this many samples (802.11 STF 16, LTF 64): the carrier offset is the angle of the
        repetition, -angle(sum x[n] x*[n + repeat]) Fs / (2 pi repeat), over `window` samples from the start found
        (default: the template's length). correct: take that offset off the output
    search: candidate starts examined (None: the whole stream, which then waits for its end). offset: the output starts
        this many samples after the preamble's start (negative: before it; 802.11 STF before the LTF)
    A real stream (an FSK discriminator's output) is matched by the correlation coefficient: the template and the stream's
    window with their means taken off (a carrier offset there is a constant), positive only. The offset, the level b of
    the fit x = a template + b over the match, is est["level"]; `correct` takes it off the output.
    first: the template's sample at its first symbol's peak; span: its samples up to the tail its pulses leave (its symbols
    times the samples a symbol; None: all of it); pending: the template comes later (`set_template`: a profile makes it
    from its own transmit side). `attach_tx()`: the transmit side, which sends this template (PreambleTx).
    Estimates in stream.est[name]: found, pos (absolute), metric, cfo [Hz] (with repeat), and where the symbols are:
    first (the absolute sample of the first symbol's peak), delay and Fs (the group delay and rate there; known timing
    with an `anchor` places its grid by them)."""
    CONTROLS = {}

    def __init__(self, template=None, repeat=None, window=None, search=None, gate=0.5, offset=0, correct=False, name="preamble",
                 first=0, pending=False, span=None):
        if template is None and not pending and (repeat is None or window is None):
            raise ValueError("Preamble: a template, or a repetition (repeat=lag, window=samples) to find")
        self.repeat, self.search, self.gate, self.offset = repeat, search, float(gate), int(offset)
        self._window = window
        self.correct, self.name = correct, name
        self.template, self.first, self.window, self.span = None, 0, None, None
        if template is not None:
            self.set_template(template, first, span)
        else:
            self.window = None if window is None else int(window)

    def set_template(self, template, first=0, span=None):
        """The waveform to find, its sample at the first symbol's peak, and its samples up to its pulses' tail"""
        t = np.asarray(template)
        if t.dtype.kind == "c" and np.max(np.abs(t.imag), initial=0.0) <= 1e-6*np.max(np.abs(t), initial=1.0):
            t = t.real # (real levels made from complex points: BPSK)
        self.template = t.astype(np.complex128) if t.dtype.kind == "c" else t.astype(np.float64)
        self.first = int(first)
        self.span = len(self.template) if span is None else int(span)
        if not 0 <= self.span <= len(self.template):
            raise ValueError("Preamble: span is within the template")
        self.window = int(self._window) if self._window is not None else len(self.template)

    def attach_tx(self):
        """The transmit side: a PreambleTx that sends this preamble's template (the one set now or later)"""
        return PreambleTx(self)

    def __str__(self):
        return f"Preamble({'template' if self.template is not None else f'repeat {self.repeat}'})"

    def new_state(self, meta):
        return SimpleNamespace(buf=None, base=None, found=None, start=None, w=0.0, est={"found": False})

    def _need(self):
        """Samples after the last candidate start a decision needs"""
        n = len(self.template) if self.template is not None else self.window + self.repeat
        if self.repeat is not None:
            n = max(n, self.window + self.repeat)
        return n

    def _locate(self, x):
        """(index, metric) of the preamble in x, or None"""
        if self.template is None and self.repeat is None:
            raise ValueError(f"{self}: no template yet (a profile makes it: set_template)")
        real = x.dtype.kind != "c" and (self.template is None or self.template.dtype.kind != "c")
        x = x.astype(np.float64 if real else np.complex128)
        if self.template is not None:
            t = self.template
            L = len(t)
            if len(x) < L:
                return None
            e = np.cumsum(np.r_[0.0, np.abs(x)**2])
            ex = e[L:] - e[:-L]
            if real: # Correlation coefficient: the means off (a carrier offset in a discriminator's output is one)
                t = t - t.mean()
                c = np.correlate(x, t, mode="valid")
                s1 = np.cumsum(np.r_[0.0, x])
                ex = ex - (s1[L:] - s1[:-L])**2/L
            else:
                c = np.abs(np.correlate(x, t, mode="valid")) # sum x[n+i] t*[i]
            m = c/np.sqrt(np.maximum(ex, 1e-30)*np.sum(np.abs(t)**2))
            if self.search is not None:
                m = m[:self.search]
            i = int(np.argmax(m))
            return (i, float(m[i])) if m[i] >= self.gate else None
        lag, W = self.repeat, self.window
        if len(x) < W + lag:
            return None
        prod = x[:-lag]*np.conj(x[lag:])
        cs = np.cumsum(np.r_[0, prod])
        cp = np.cumsum(np.r_[0.0, np.abs(x[lag:])**2])
        C, P = cs[W:] - cs[:-W], cp[W:] - cp[:-W]
        m = np.abs(C)/np.maximum(P, 1e-30)
        if self.search is not None:
            m, C = m[:self.search], C[:self.search]
        ok = m >= self.gate
        if not ok.any():
            return None
        a = np.abs(C)
        hit = np.nonzero(ok & (a >= 0.97*a[ok].max()))[0]
        return int(hit[0]), float(m[hit[0]])

    def _cfo(self, x, i, Fs):
        lag, W = self.repeat, self.window
        seg = x[i:i + W + lag].astype(np.complex128)
        if len(seg) < lag + 1:
            return 0.0
        acc = np.sum(seg[:-lag]*np.conj(seg[lag:]))
        return float(-np.angle(acc)*Fs/(2*np.pi*lag))

    def step(self, x, st, meta):
        x = np.asarray(x)
        if st.found is False: # Given up on: this stream has no preamble
            return x[:0]
        if st.found:
            return self._out(x, meta.pos, st, meta)
        if st.buf is None:
            st.buf, st.base = x[:0], meta.pos
        st.buf = np.concatenate((st.buf, x))
        enough = self.search is not None and len(st.buf) >= self.search + self._need()
        if not (enough or meta.flush):
            return x[:0]
        hit = self._locate(st.buf)
        if hit is None:
            st.found, st.buf = False, None
            st.est = {"found": False}
            meta.stream.est[self.name] = dict(st.est)
            return x[:0]
        i, metric = hit
        st.found = True
        st.est = {"found": True, "pos": st.base + i, "metric": metric, "first": st.base + i + self.first, "delay": meta.delay,
                  "Fs": meta.Fs}
        if self.repeat is not None:
            st.est["cfo"] = self._cfo(st.buf, i, meta.Fs)
        st.level = 0.0
        if self.template is not None and self.template.dtype.kind != "c" and st.buf.dtype.kind != "c":
            t = self.template
            w = st.buf[i:i + len(t)].astype(np.float64)
            a = np.dot(w - w.mean(), t - t.mean())/max(np.sum((t - t.mean())**2), 1e-30)
            st.est["level"] = float(w.mean() - a*t.mean())
            st.level = st.est["level"] if self.correct else 0.0
        st.start = st.base + max(i + self.offset, 0) # (a start before the stream's first sample is clamped)
        st.w = 2*np.pi*st.est.get("cfo", 0.0)/meta.Fs if self.correct else 0.0
        meta.stream.est[self.name] = dict(st.est)
        buf, st.buf = st.buf[st.start - st.base:], None
        meta.stream.out_pos[self] = st.start # Data.apply places the output at the preamble, not at the block
        return self._out(buf, st.start, st, meta)

    def _out(self, x, pos, st, meta):
        if getattr(st, "level", 0.0) and len(x):
            return (x - st.level).astype(x.dtype)
        if not st.w or not len(x):
            return x
        n = pos - st.start + np.arange(len(x)) # Samples since the output's start: the correction's phase is continuous
        return x*np.exp(-1j*st.w*n)


class PreambleTx(Stage):
    """The transmit side of a Preamble (`Preamble.attach_tx()`): its template put before a stream

    The stream starts with the template, the stream's own samples from its `span` on (the preamble's symbols), the tail its
    pulses leave added to them: for linear steps (pulse shaping) exactly the waveform of the preamble's symbols and the
    stream's sent as one. The template is the Preamble's when the stream starts (a profile sets it after building)."""
    CONTROLS = {}

    def __init__(self, preamble):
        self.preamble = preamble

    def __str__(self):
        return "PreambleTx"

    def new_state(self, meta):
        return SimpleNamespace(rest=None)

    def step(self, x, st, meta):
        t, span = self.preamble.template, self.preamble.span
        if t is None:
            raise ValueError(f"{self}: its Preamble has no template yet (a profile makes it)")
        x = np.asarray(x)
        dt = np.result_type(x.dtype, t.dtype)
        if st.rest is None: # The stream's first block: the preamble before it
            x = np.concatenate((t[:span].astype(dt), x.astype(dt)))
            st.rest, at = t[span:].astype(dt), span
        else:
            x, at = x.astype(dt), 0
        n = min(len(st.rest), len(x) - at)
        if n > 0:
            x = x.copy()
            x[at:at + n] += st.rest[:n]
            st.rest = st.rest[n:]
        if meta.flush and len(st.rest):
            x, st.rest = np.concatenate((x, st.rest)), st.rest[:0]
        return x
