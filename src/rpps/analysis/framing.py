"""Blind symbol framing: the period a symbol stream repeats with, the unique word that marks it, and the frames cut on it

    fr = rp.analysis.framing.find(symbols, "qpsk")    # Frame(period, uw_start, uw_length, coherence, ...) or None
    frames = rp.analysis.framing.cut(symbols, fr)     # frames x period, each turned so its unique word matches the first
    bits = rp.analysis.framing.payload_bits(frames, fr, "qpsk")   # frames x payload bits, for rpps.analysis.block / fec.solve

A modem that frames its symbols (a unique word ahead of each block of coded data: TPC and LDPC SCPC modems, TDM framing,
a TDMA burst's preamble) sends the same few symbols every P symbols. The product y[n] conj(y[n+P]) is then coherent at
those positions and random elsewhere: averaged over many frames, it stands out where the unique word is, and P is the
lag whose average stands out most (`find`). The carrier phase may differ frame to frame by a multiple of 360/M degrees
(the M-fold ambiguity of a feed-forward carrier, or a slip), which the unique word itself resolves (`cut`).

Coded data under the unique word is random-looking; data that repeats (idle frames, fill) makes whole frames equal, and
`repeats` finds the period of that (a superframe, or how often a frame is sent again).
"""
import numpy as np

class Frame:
    """`find`: period (symbols), uw_start (position in the period), uw_length (symbols), coherence (of the unique word,
    1 = the same symbols every frame), floor (of the other positions), uw (the unique word's symbols, from the first
    frame), frames (how many the estimate is over)"""
    __slots__ = ("period", "uw_start", "uw_length", "coherence", "floor", "uw", "frames")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        return (f"frames of {self.period} symbols, a {self.uw_length}-symbol unique word at {self.uw_start} "
                f"(coherence {self.coherence:.2f}, other positions {self.floor:.2f}, over {self.frames} frames)")

    __repr__ = __str__

def _normalized(y):
    y = np.asarray(y, dtype=np.complex128)
    return y/np.sqrt(np.mean(np.abs(y)**2))

def periods(y, max_lag=20000, min_lag=16, top=8):
    """[(lag, correlation)] where the symbols correlate with themselves, strongest first: |mean y[n] conj(y[n+lag])|"""
    y = _normalized(y)
    n = len(y)
    max_lag = min(max_lag, n//4)
    if max_lag <= min_lag:
        return []
    Y = np.fft.fft(y, 2*n)
    r = np.abs(np.fft.ifft(np.abs(Y)**2)[:max_lag + 1])/np.arange(n, n - max_lag - 1, -1)
    lags = np.argsort(-r[min_lag:])[:4*top] + min_lag
    out = []
    for L in lags: # One per peak: a lag next to a stronger one is its shoulder
        if all(abs(L - k) > 2 for k, _ in out):
            out.append((int(L), float(r[L])))
        if len(out) == top:
            break
    return out

def coherence(y, period):
    """Per position in the period: |mean over frames of y[n] conj(y[n+period])|, 1 where every frame has the same symbol"""
    y = _normalized(y)
    c = y[:-period]*np.conj(y[period:])
    m = len(c)//period
    return np.abs(c[:m*period].reshape(m, period).mean(axis=0))

def find(y, mod=None, max_lag=20000, min_frames=8, threshold=0.5):
    """The frame structure of locked symbols y: a Frame, or None (no lag with a unique word standing out)

    The candidate periods are the lags the symbols correlate at (`periods`), shortest first among those near the
    strongest (a multiple of the frame is a superframe, kept for `repeats`); at a period, the unique word is the longest
    run of positions whose coherence passes `threshold` while the median position stays near chance. Data that itself
    repeats every few frames doesn't average out, which hides the unique word: then frame by hand and use `repeats`."""
    y = _normalized(y)
    cands = {L for L, _ in periods(y, max_lag)}
    # Frames that repeat (fill, a superframe) correlate most at the repeat: the frame is a divisor of it
    cands |= {L//k for L in list(cands) for k in range(2, 17) if L % k == 0 and L//k >= 16}
    best = None
    for L in sorted(cands):
        if len(y)//L < min_frames:
            continue
        prof = coherence(y, L)
        floor = float(np.median(prof))
        hot = prof > max(threshold, 4*floor)
        if not hot.any():
            continue
        # The longest run of coherent positions (circular: the word may straddle the fold)
        h = np.concatenate((hot, hot))
        runs, start = [], None
        for i, v in enumerate(h):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i - start))
                start = None
        if start is not None:
            runs.append((start, len(h) - start))
        s0, ln = max(runs, key=lambda t: t[1])
        ln = min(ln, L)
        s0 %= L
        coh = float(np.mean(prof[(s0 + np.arange(ln)) % L]))
        f = Frame(period=L, uw_start=int(s0), uw_length=int(ln), coherence=coh, floor=floor,
                  uw=y[s0:s0 + ln].copy(), frames=len(y)//L)
        if best is None or ln*coh > best.uw_length*best.coherence*1.5: # A shorter period wins unless much weaker
            best = f
    return best

def cut(y, fr, M=None, grid=4):
    """Frames x period symbols starting at the unique word, each turned onto the others by its unique word

    The reference is the word averaged over all frames (one frame's own word is as noisy as any symbol), found in two
    passes: frames turned onto the first frame's word by multiples of 90 degrees, averaged, then every frame onto that
    average. M=None: by the angle measured (also takes off a carrier phase drifting frame to frame); M: by the nearest
    multiple of 360/M degrees only (the ambiguity alone). Then the whole set is turned so the payload sits on a
    `grid`-fold symmetric constellation (QPSK points at 45 + 90k degrees, as rpps.symbol's), by its grid-th power."""
    y = _normalized(y)
    F = (len(y) - fr.uw_start)//fr.period
    f = y[fr.uw_start:fr.uw_start + F*fr.period].reshape(F, fr.period)
    U = fr.uw_length
    first = f[0, :U]
    a0 = np.angle(np.sum(f[:, :U]*np.conj(first), axis=1))
    ref = np.mean(f[:, :U]*np.exp(-1j*(np.pi/2)*np.round(a0/(np.pi/2)))[:, None], axis=0)
    a = np.angle(np.sum(f[:, :U]*np.conj(ref), axis=1))
    if M:
        a = 2*np.pi/M*np.round(a/(2*np.pi/M))
    f = f*np.exp(-1j*a)[:, None]
    if grid:
        pay = f[:, U:].ravel()
        g = (np.angle(-np.mean(pay**grid)) if grid == 4 else np.angle(np.mean(pay**grid)))/grid # QPSK: (e^{j pi/4})^4 = -1
        f = f*np.exp(-1j*g)
    return f

def payload_bits(frames, fr, constellation):
    """Frames x bits of the symbols after each unique word, hard-decided on `constellation` (a name or a Constellation)"""
    from .. import symbol
    c = symbol.load(constellation) if isinstance(constellation, str) else constellation
    pay = frames[:, fr.uw_length:]
    pts = np.asarray(c.points)/np.sqrt(np.mean(np.abs(c.points)**2))
    labels = np.asarray(c.mapping.arr if hasattr(c, "mapping") else np.arange(len(pts)))
    idx = np.argmin(np.abs(pay[..., None] - pts[None, None, :]), axis=-1)
    lab = labels[idx]
    k = c.bits_per_symbol
    bits = (lab[..., None] >> np.arange(k - 1, -1, -1)) & 1
    return bits.reshape(len(frames), -1).astype(np.uint8)

def repeats(frame_bits, max_period=64, agree=0.99):
    """(period, fraction of frames equal to the one `period` before): the smallest such period, or (None, 0.0). Frames
    that come again (idle, fill, a fixed superframe) are content, not code"""
    b = np.asarray(frame_bits, dtype=np.uint8)
    for p in range(1, min(max_period, len(b)//2) + 1):
        same = np.mean(np.mean(b[p:] == b[:-p], axis=1) >= agree)
        if same >= 0.5:
            return p, float(same)
    return None, 0.0

# --- A known word in samples: preambles, unique words, burst sync ------------------------------------------------------

def word_waveform(word, sps, beta=0.35, span=8):
    """The symbols `word` shaped by an RRC pulse at a fractional `sps` samples per symbol"""
    from ..signal.filters import rrc_pulse
    word = np.asarray(word, dtype=np.complex128)
    n = int(np.ceil((len(word) + 2*span)*sps))
    t = np.arange(n)/sps - span
    out = np.zeros(n, dtype=np.complex128)
    for k, s in enumerate(word):
        d = t - k
        m = np.abs(d) <= span
        out[m] += s*rrc_pulse(d[m], beta)
    return out

class Hit:
    """`find_word`: at (sample of the word's first symbol), score (0..1, 1 = the word alone, noiseless), baud, sps"""
    __slots__ = ("at", "score", "baud", "sps")
    def __init__(self, at, score, baud, sps):
        self.at, self.score, self.baud, self.sps = at, score, baud, sps

    def __repr__(self):
        return f"<word at sample {self.at}, score {self.score:.2f}, {self.baud/1e3:.3f} kBd>"

def find_word(x, Fs, word, bauds, beta=0.35, segments=8, threshold=0.5, top=16):
    """Where a known symbol word (a preamble, a burst's unique word) is in baseband samples x at Fs: [Hit] best first

    bauds: the symbol rates to try. The word's shaped waveform is cut into `segments` pieces, each correlated on its own
    and their magnitudes added, so a carrier offset that turns the phase across the word (up to about baud/len(word)*
    segments/4) and an unknown phase cost nothing. score: that sum over what the samples' energy allows (Cauchy-Schwarz),
    near 1 where the word is, about 1/sqrt(samples per segment) elsewhere."""
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    hits = []
    e = np.concatenate(([0.0], np.cumsum(np.abs(x)**2)))
    for baud in np.atleast_1d(bauds):
        sps = Fs/baud
        w = word_waveform(word, sps, beta)
        L = len(w)
        if L >= n:
            continue
        edges = np.linspace(0, L, segments + 1).astype(int)
        acc = np.zeros(n - L + 1)
        X = np.fft.fft(x, n + L)
        for a, b in zip(edges[:-1], edges[1:]):
            seg = np.zeros(L, dtype=np.complex128)
            seg[a:b] = w[a:b]
            c = np.fft.ifft(X*np.conj(np.fft.fft(seg, n + L)))[:n - L + 1] # sum x[t+i] conj(seg[i])
            win = e[np.arange(n - L + 1) + b] - e[np.arange(n - L + 1) + a]
            acc += np.abs(c)/np.sqrt(np.maximum(win, 1e-30)*np.sum(np.abs(w[a:b])**2))
        acc /= segments
        for k in np.argsort(-acc)[:top]:
            if acc[k] >= threshold:
                hits.append(Hit(int(k + round(8*sps)), float(acc[k]), float(baud), float(sps))) # (the pulse's lead)
    hits.sort(key=lambda h: -h.score)
    out = []
    for h in hits: # One per burst: a hit within the word's length of a better one is that one
        if all(abs(h.at - o.at) > len(word)*o.sps for o in out):
            out.append(h)
    return out[:top]

# --- Bursts: gate, acquire each on its own, and the preamble they share --------------------------------------------------

class Burst:
    """`bursts`: start, stop (samples), det (rpps.analysis.Detection of this burst alone, None if too short), symbols
    (matched-filtered at det's rate and timing, carrier offset and phase taken off)"""
    __slots__ = ("start", "stop", "det", "symbols")
    def __init__(self, start, stop, det=None, symbols=None):
        self.start, self.stop, self.det, self.symbols = start, stop, det, symbols

    def __repr__(self):
        d = f", {self.det.mod} {self.det.baud/1e3:.1f} kBd, Es/N0 {self.det.esn0:.1f} dB" if self.det is not None else ""
        return f"<burst {self.start}-{self.stop}{d}>"

def gate(x, Fs, smooth=4e-6, contrast=10.0, min_len=64e-6, hang=8e-6):
    """[(start, stop)] where the power is on: above the geometric mean of its low and high levels (`contrast` apart at
    least), bursts closer than `hang` merged, those shorter than `min_len` dropped"""
    p = np.abs(np.asarray(x))**2
    w = max(2, int(Fs*smooth))
    e = np.convolve(p, np.ones(w)/w, mode="same")
    lo, hi = np.percentile(e, 5), np.percentile(e, 99)
    if hi < contrast*lo:
        return []
    on = (e > np.sqrt(lo*hi)).astype(np.int8)
    d = np.diff(np.concatenate(([0], on, [0])))
    spans = list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < hang*Fs:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(int(a), int(b)) for a, b in merged if b - a >= min_len*Fs]

def bursts(x, Fs, names=None, baud=None, min_symbols=128, **kw):
    """The bursts of x (`gate`), each acquired on its own: rate (or `baud` for all), modulation, offset, timing (as a
    TDMA receiver must: every burst comes from another transmitter, with its own clock and carrier). [Burst]"""
    from .modulation import detect, correct, candidates, symmetry
    from ..sync.time.mat import matrix_mf, centres_in, best_tau
    x = np.asarray(x, dtype=np.complex128)
    out = []
    for a, b in gate(x, Fs, **kw):
        seg = x[a:b]
        bu = Burst(a, b)
        try:
            det = detect(seg, Fs, names) if baud is None else None
            if baud is not None:
                det = detect(seg, Fs, names)
                sps = Fs/baud
            else:
                sps = det.sps
            if len(seg)/sps >= min_symbols:
                # The matched filter needs the pulse's span either side: widen the segment by it, so the burst's first
                # symbols (the preamble) are taken too; the timing found on the gated part carries over
                pad = int(np.ceil(10*sps))
                a2, b2 = max(0, a - pad), min(len(x), b + pad)
                seg = x[a2:b2]
                t0 = det.tau + (a - a2)
                t0 -= np.floor((t0 - 8*sps - 5)/sps)*sps
                c = centres_in(len(seg), t0, sps, 8)
                y = matrix_mf(seg, c, sps, det.beta, 8)
                y = y/np.sqrt(np.mean(np.abs(y)**2))
                pts = candidates([det.mod])[det.mod]
                bu.symbols = correct(y, pts, symmetry(pts))[0]
                bu.det = det
        except Exception: # A burst too short or too weak to acquire is kept, without symbols
            pass
        out.append(bu)
    return out

def preamble(burst_list, max_len=512, agree=0.8, search=8):
    """What the bursts' first symbols share: (length, differential symbols d[n] = s[n] conj(s[n-1]) of it, where it starts
    in each burst's symbols) or (0, None, None). Differential, so each burst's own carrier phase drops out; each burst is slid by up to `search`
    symbols onto the first (the gate finds a start to within a few symbols), and the preamble is the leading run where
    the bursts' d[n] agree (|mean| over bursts above `agree`)"""
    S = [b.symbols for b in burst_list if b.symbols is not None and len(b.symbols) > 2*search + 32]
    if len(S) < 3:
        return 0, None, None
    D = [s[1:]*np.conj(s[:-1])/(np.abs(s[1:]*s[:-1]) + 1e-12) for s in S]
    ref = D[0][search:search + 64]
    offs = []
    for d in D: # Burst i's symbol k + offs[i] is the first burst's symbol k
        sc = [np.abs(np.vdot(ref, d[k:k + 64])) for k in range(2*search + 1)]
        offs.append(int(np.argmax(sc)) - search)
    c = max(0, -min(offs)) # From the first symbol every burst has
    L = min(max_len, min(len(d) - o - c for d, o in zip(D, offs)))
    rows = [d[o + c:o + c + L] for d, o in zip(D, offs)]
    m = np.abs(np.mean(np.array(rows), axis=0))
    s0 = next((i for i in range(min(len(m), 2*search + 2)) if m[i] > agree), None) # (the gate may open a little early)
    if s0 is None:
        return 0, None, None
    n = 0
    while s0 + n < len(m) and m[s0 + n] > agree:
        n += 1
    return n, np.mean(np.array(rows), axis=0)[s0:s0 + n], [o + c + s0 for o in offs]

# --- The unique word as pilots: symbol timing and carrier phase from it, frame by frame ----------------------------------

def _refine(x, at, uw, sps, beta, span=8, step=0.05):
    """Fractional timing of a word found near sample `at`: the offset in [-sps/2, sps/2] (by `step` symbols) whose matched
    filter outputs best match the word. Returns (time of its first symbol, complex correlation)"""
    from ..sync.time.mat import matrix_mf
    best = (None, 0.0)
    k = np.arange(len(uw))
    for d in np.arange(-0.5, 0.5 + 1e-9, step)*sps:
        c = at + d + k*sps
        if c[0] - span*sps - 2 < 0 or c[-1] + span*sps + 2 >= len(x):
            continue
        r = np.vdot(uw, matrix_mf(x, c, sps, beta, span))
        if best[0] is None or abs(r) > abs(best[1]):
            best = (at + d, r)
    return best

def coarse_cfo(x, Fs, M=4, pad=4):
    """Carrier offset [Hz] from the line of x^M (the modulation's M-fold symmetry taken off)"""
    from ..math.interp import peak
    n = 1 << int(np.ceil(np.log2(min(len(x), 2**21))))
    z = np.asarray(x[:n], dtype=np.complex128)**M
    S = np.abs(np.fft.fft(z*np.hanning(len(z)), pad*n))
    k, d = peak(S, circular=True)
    f = (k + d)/(pad*n)
    return (f - (1 if f > 0.5 else 0))*Fs/M

def pilot_frames(x, Fs, uw, baud, period, beta=0.35, span=8, threshold=0.45, M=4, dropout=0.6, track=64, smooth=8):
    """Frames of symbols synchronized on their unique word as pilots: (frames x period symbols from each word on, the words'
    sample times)

    Each word is found in the samples (`find_word`) and timed to a fraction of a sample; the symbols of a frame lie evenly
    between its word and the next one, which follows the transmitter's clock frame by frame. The carrier phase is measured
    on each word and taken linearly across the frame to the next (a frequency per frame), so no loop has to settle and a
    slip can't carry over. uw: the word's symbols (from `cut`: the first frame's, or their mean). The coarse carrier offset
    comes off first, from the M-th power (`coarse_cfo`). threshold: the first word's match (`find_word`; a short word's
    pulse tails overlap random neighbours, which caps it near 0.6); dropout: a word under this times the running level is
    taken as missing, and the clock carried over it. track: symbols over which the phase left between the words (LNB
    phase noise) is taken off by the M-th power (0: none). smooth: the words' times are fitted with a line over this many
    frames either side (a short word times one frame noisily; 0: as measured)."""
    from ..sync.time.mat import matrix_mf
    x = np.asarray(x, dtype=np.complex128)
    # The coarse carrier offset off first: across a frame the words' phases may only turn by less than half a turn
    cfo = coarse_cfo(x, Fs, M)
    x = x*np.exp(-2j*np.pi*cfo/Fs*np.arange(len(x)))
    uw = np.asarray(uw, dtype=np.complex128)
    uw = uw/np.sqrt(np.mean(np.abs(uw)**2))
    sps = Fs/baud
    # A short word matches random data now and then: anchor on the strongest match, then walk frame by frame, each word
    # looked for only near where the last one and the clock put it (the clock taken from the words found so far)
    first = find_word(x[:int(min(len(x), 4*period*sps))], Fs, uw, [baud], beta, threshold=threshold, top=4,
                      segments=1 if len(uw) < 32 else 8)
    if not first:
        return np.zeros((0, period), complex), np.zeros(0)
    t0, r0 = _refine(x, first[0].at, uw, sps, beta, span)
    while t0 is not None and t0 - period*sps > span*sps + 2: # Back to the first word in x
        t0 -= period*sps
    t0, r0 = _refine(x, t0, uw, sps, beta, span) if t0 is not None else (None, 0)
    times, phases = [], []
    T, t = period*sps, t0
    floor = None
    while t is not None and t + T + (span + 2)*sps < len(x):
        best = (None, 0.0)
        for d in range(-2, 3): # Within two symbols of where the word should be
            tt, r = _refine(x, t + d*sps, uw, sps, beta, span)
            if tt is not None and abs(r) > abs(best[1]):
                best = (tt, r)
        tt, r = best
        q = abs(r)/len(uw)
        if floor is None:
            floor = q
        if tt is None or q < dropout*floor: # Not there (a dropout): carry the clock over it
            times.append(None)
            phases.append(None)
        else:
            times.append(tt)
            phases.append(np.angle(r))
            known = [(i, v) for i, v in enumerate(times) if v is not None][-16:]
            if len(known) >= 2:
                T = (known[-1][1] - known[0][1])/(known[-1][0] - known[0][0])
            floor = 0.9*floor + 0.1*q
        t = (tt if tt is not None else t) + T
    # A short word times each frame noisily, the transmitter's clock is steady: each word's time from a straight line
    # through the words within `smooth` frames of it
    if smooth:
        idx = np.array([i for i, t in enumerate(times) if t is not None])
        tv = np.array([times[i] for i in idx])
        fitted = list(times)
        for j, i in enumerate(idx):
            m = np.abs(idx - i) <= smooth
            if m.sum() >= 3:
                a, b = np.polyfit(idx[m] - i, tv[m], 1)
                fitted[i] = b
        times = fitted
    frames, starts = [], []
    for i in range(len(times) - 1):
        if times[i] is None or times[i + 1] is None:
            continue
        T = times[i + 1] - times[i]
        if abs(T/period - sps) > 0.01*sps: # Not the next frame's word (one missed between): no clock to take from it
            continue
        c = times[i] + np.arange(period)*T/period
        if c[0] - span*sps - 2 < 0 or c[-1] + span*sps + 2 >= len(x):
            continue
        y = matrix_mf(x, c, T/period, beta, span)
        dphi = np.angle(np.exp(1j*(phases[i + 1] - phases[i]))) # The turn over one frame (a carrier offset under half a turn a frame)
        y = y*np.exp(-1j*(phases[i] + dphi*np.arange(period)/period))
        if track and M:
            # Phase noise between the words: what is left, a small turn, by the M-th power over `track` symbols (on the
            # words' reference already, so no ambiguity: only the part under 180/M degrees is taken)
            ref = np.mean(uw**M)
            for a in range(0, period, track):
                seg = y[a:a + track]
                y[a:a + track] = seg*np.exp(-1j*np.angle(np.mean(seg**M)*np.conj(ref))/M)
        frames.append(y)
        starts.append(times[i])
    f = np.array(frames)
    return (f/np.sqrt(np.mean(np.abs(f)**2)) if len(f) else f), np.array(starts)
