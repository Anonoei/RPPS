"""AcquireFEC: which code timed symbols carry, found as the stream arrives (the scans are in rpps.analysis.identify)"""
from types import SimpleNamespace

import numpy as np

from ..base import Stage
from ..symbol.mapping import hard_bits
from ..code.coding import mismatch, cached

class AcquireFEC(Stage):
    """Identify the code the symbols carry, then turn them to the map its decoder uses

        rx = front + rp.sync.Acquire() + rp.sync.time.Mat() + rp.sync.phase.FFCarrier() + rp.sync.AcquireFEC()
        out = blk + rx
        fec = out.stream.est.get("fec")           # an rpps.analysis.identify.CodeDetection, once there is one
        dec = fec.rx()                             # once fec.complete: the decoder (a profile runner), bits = dec(out)

    Takes timed, carrier-corrected symbols (after `phase.FFCarrier`, with the modulation from `Acquire` in meta.det)
    and gives the same symbols, turned and conjugated so the decoder's map is the right one (the carrier phase has an
    M-fold ambiguity, and the IQ sense is unknown). Codes are the configured ones (`rpps.analysis.identify.identify`).

    Acquisition: symbols are held until `skip` + `window` have arrived, and the code is identified on those `window`
    (at absolute positions, so the result is the same however the stream was cut). While more symbols could tell more
    (a window shorter than the scans use, or Reed-Solomon behind a Viterbi, which needs ~100 codewords) the window doubles,
    up to `max_wait` symbols. A stream that ends first (meta.flush) is decided on every symbol held past `skip`, blind
    tools included, so a short capture gets the same look a long one does. Then what was held is released from the window on (from a codeword start for a block code;
    the `skip` symbols before it, where the sync before this was settling, are dropped).
    Tracking (convolutional codes): every `check` symbols the output is re-encoded through the code on the decoder's map.
    A mismatch above `relock` times the one at acquisition means a slip: the carrier by a turn, or the symbol timing by a
    symbol (the code's puncture phase moves). Every turn and alignment is tried on that window; it goes on with the one the
    code fits, turned, and with symbols dropped so the decoder after it keeps its alignment. `slips` counts them.

    Its estimate is `stream.est[name]`: the CodeDetection (found False when nothing was, with its report). While it is still
    acquiring it holds the answer so far with complete False (before any, found False and `n` the symbols held so far): a
    code found then may still gain the layers inside it, so a decoder is best built once complete, which it is once the stage
    stops looking.
    names: only these convolutional definitions. blind: also look for unknown codes (a `Guess`, never decoded), unless the
    rate came from a symbol-rate line weaker than `min_line` over its floor (Acquire took it anyway: a guess; None: always)
    """
    __slots__ = ("window", "skip", "max_wait", "names", "blind", "check", "relock", "name", "min_line")

    def __init__(self, window=2**14, skip=2048, max_wait=2**18, names=None, blind=False, check=4096, relock=4.0, name="fec",
                 min_line=5.0):
        self.window, self.skip, self.max_wait = int(window), int(skip), int(max_wait)
        self.names, self.blind, self.min_line = names, blind, min_line
        self.check, self.relock, self.name = int(check), float(relock), name

    def __str__(self):
        return "AcquireFEC()"

    def new_state(self, meta):
        return SimpleNamespace(held=[], count=0, length=self.window, done=False, fec=None, c=None, code=None, out=0,
                               drop=0, buf=[], buffered=0, shift=0, prior=None)

    def step(self, y, st, meta):
        det = meta.det
        if det is None:
            return y
        if st.done:
            return self._emit(y, st) if len(y) else y
        if len(y) == 0 and not (meta.flush and st.count): # (an empty last block still ends the acquisition)
            return y
        st.held.append(y)
        st.count += len(y)
        end = meta.flush # The stream ends with this block: decide on what there is
        if st.count < self.skip + st.length and not end:
            if self.name not in meta.stream.est: # Still acquiring: say so (found and complete False), and how far it got
                from ..analysis.identify import CodeDetection # The scans are blind analysis: loaded when first needed
                meta.stream.est[self.name] = CodeDetection(mod=det.mod)
            if meta.stream.est[self.name].report is None: # (an answer so far keeps the symbols it was found on)
                meta.stream.est[self.name].n = st.count
            return y[:0]
        x = np.concatenate(st.held)
        st.held = [x]
        win = x[self.skip:] if end else x[self.skip:self.skip + st.length] # (at the end: every symbol past `skip`)
        from ..analysis.identify import identify # (blind analysis, loaded when first needed)
        prior = st.prior # The previous window's answer: the configured scans saw the same start (identify(prior=))
        fec = identify(win, det.mod, self.names, blind=False, baud=det.baud, prior=prior)
        st.prior = fec
        last = end or 2*st.length > self.max_wait - self.skip
        if not last and not fec.complete:
            st.length *= 2
            meta.stream.est[self.name] = fec # The answer so far: complete False, more symbols may still change it
            return y[:0]
        # The blind tools take most of the time and don't decide completeness: once, on the final window; and not on a
        # rate Acquire only guessed (no symbol-rate line of min_line over its floor: symbols at a wrong rate fit no code)
        guessed = self.min_line is not None and getattr(det, "line", None) is not None and det.line < self.min_line
        if self.blind and not guessed:
            fec = identify(win, det.mod, self.names, blind=True, baud=det.baud, prior=fec)
        st.done, st.held, st.fec = True, None, fec
        fec.complete = True # Final: this stage looks no further (even if more symbols could have shown more)
        fec.n = len(win)
        fec.latency = (self.skip + len(win))/det.baud if det.baud else None # Held until the window was in
        meta.stream.est[self.name] = fec
        if not fec.found:
            return x
        from .. import symbol
        st.c = symbol.load_or_generate(fec.mod)
        bps = st.c.bits_per_symbol
        # Out from the window the code was identified on: before it the sync was settling (a symbol slipped there would
        # give a decoder that aligns once, on what comes first, another alignment than the one found and tracked here)
        st.drop = self.skip
        if fec.kind == "block" and fec.period: # And from a codeword start, if one falls on a symbol
            e = next((fec.offset + k*fec.period for k in range(bps) if (fec.offset + k*fec.period) % bps == 0), None)
            st.drop += 0 if e is None else e//bps
        if fec.kind == "conv" and not fec.differential:
            st.code = cached(f"conv/{fec.name}")
        st.out = st.drop
        return self._emit(x[st.drop:], st)

    # ---- tracking: the code still fits the symbols as turned and aligned

    def _mismatch(self, w, start, st, shift=0):
        """Re-encoding mismatch of symbols w (input symbol `start` on) on the decoder's map, the code's period starting
        where it did at acquisition, moved by the symbol slips seen since (st.shift) and `shift` more bits"""
        fec = st.fec
        bits = hard_bits(w, st.c.points, fec.target.arr)
        a = (fec.offset - (start - self.skip)*st.c.bits_per_symbol + st.shift + shift) % fec.period
        seg = bits[a:]
        seg = seg[:len(seg)//fec.period*fec.period]
        k, n = mismatch(st.code, seg)
        return k/max(n, 1)

    def _emit(self, y, st):
        """Symbols out, turned. With tracking they go out a window of `check` at a time, each checked (and put right if
        the carrier or the symbol timing slipped) before it goes, so a slip is put right where it was found whatever the cut"""
        fec = st.fec
        if not fec.found:
            return y
        if st.code is None:
            return fec.correct(y)
        st.buf.append(y)
        st.buffered += len(y)
        out = []
        while st.buffered >= self.check:
            w = np.concatenate(st.buf)
            raw, rest = w[:self.check], w[self.check:]
            st.buf, st.buffered = [rest], len(rest)
            start = st.out
            st.out += self.check
            win = fec.correct(raw)
            if start >= self.skip and self._mismatch(win, start, st) > self._limit(st): # (before skip: still settling)
                drop = self._relock(win, start, st)
                win = fec.correct(raw)[drop:]
            out.append(win)
        return np.concatenate(out) if out else y[:0]

    def _limit(self, st):
        """Mismatch above which the turn or alignment is wrong: well above lock, well below what a wrong hypothesis gets"""
        return min(self.relock*max(st.fec.mismatch, 2e-3), 0.5*st.fec.baseline)

    def _relock(self, win, start, st):
        """Try every turn of the constellation and every alignment in the code's period on this window, and go on with the
        one the code fits, if one does. A new alignment is a symbol the timing slipped: symbols are dropped from here so
        that what follows keeps the alignment the decoder after this has. Returns the symbols to drop"""
        from ..symbol.mapping import symmetries
        fec = st.fec
        bps, P = st.c.bits_per_symbol, fec.period
        tries = []
        for th, flip, _ in symmetries(st.c.points):
            w = np.conj(win)*np.exp(1j*th) if flip else win*np.exp(1j*th)
            tries += [(self._mismatch(w, start, st, d), th, flip, d) for d in range(P)]
        mm, th2, f2, d = min(tries, key=lambda t: t[0])
        if mm > self._limit(st):
            return 0
        if abs(np.angle(np.exp(1j*th2))) > 1e-6 or f2:
            th1, f1 = fec.correction # y' = conj^f1(y) e^(j th1), then y'' = conj^f2(y') e^(j th2)
            fec.correction = (float((th2 - th1 if f2 else th2 + th1) % (2*np.pi)), bool(f1 ^ f2))
        fec.slips += 1
        if d == 0:
            return 0
        st.shift += d
        return next((m for m in range(P) if (m*bps - d) % P == 0), 0) # m symbols move what follows back by d bits
