import numpy as np

from collections import namedtuple

Channel = namedtuple("Channel", "front Fs decimation delay settle")
Channel.__doc__ = """What brings a Signal to baseband: `block + channel.front` is the signal at Fs `Fs` (the capture's Fs / `decimation`)

delay: the front's group delay [s] (what a block's `meta.delay` gains through it)
settle: output samples a new stream starts with before every filter is full of signal (its start-up transient: the
    filters are linear phase, so twice the delay); a display drops these from a stream's first block"""

class Signal:
    """A signal found in a spectrum

    mid: centre [Hz, from the capture's centre], freq: centre [Hz] absolute (cf + mid)
    width: occupied bandwidth [Hz] (the span holding `occupied` of its power), lo/hi: its edges [Hz from the capture's centre]
    bottom: noise floor [dB] (global), height: peak above the floor [dB], snr: in-band power over noise power in that band [dB]
    power: total in-band power [dB], peak_freq: strongest bin [Hz from the capture's centre]
    duty: fraction of the time it is on (1: continuous, less: bursty)
    resolved: there were enough bins across it to trust the edges. False means read more samples
    clipped: it runs into the edge of the capture's band, so its width and centre are only what is inside
    baud_est / baud_strength: symbol rate from the |x|^2 line of the channelized signal (None: no clear line)
    holds: how many other signals stand on it (a transponder's pedestal, or a carrier with a narrower signal on it: which
        one, the spectrum can't tell; its own symbol-rate line can). over: the one it stands on, when it stands on one
    lines: a comb (lines side by side at one spacing, one box): how many. t0/t1, bursts: when it is on
    score: dB over the threshold that found it; edge, rough: its shape (see rpps.analysis.spectrum.find)
    id: its number, kept from survey to survey by `Survey.match`
    Fs, cf, bin_hz: the capture's rates and the spectrum's resolution
    """
    def __init__(self, mid, width, bottom, height, snr=None, *, lo=None, hi=None, power=None, peak_freq=None,
                 duty=None, resolved=True, Fs=None, cf=0.0, bin_hz=None, clipped=False):
        self.mid = mid
        self.width = width
        self.bottom = bottom
        self.height = height
        self.snr = snr
        self.lo = mid - width/2 if lo is None else lo
        self.hi = mid + width/2 if hi is None else hi
        self.power = power
        self.peak_freq = mid if peak_freq is None else peak_freq
        self.duty = duty
        self.resolved = resolved
        self.clipped = clipped
        self.Fs, self.cf, self.bin_hz = Fs, cf, bin_hz
        self.baud_est = None
        self.baud_strength = 0.0
        self.holds = 0
        self.id = None
        self.t0 = self.t1 = None # when it is on [s from the capture's start]; bursts: [(t0, t1)], None when steady
        self.bursts = None
        self.over = None         # the signal it stands on (a pedestal, a carrier it jams), or None
        self.lines = 0           # a comb: how many lines it holds
        self.score = None        # dB over the threshold that found it
        self.edge = None         # its edges' 25-75% rise over its width, the wider side
        self.rough = None        # its top's roughness beyond the noise's [dB]

    @property
    def freq(self):
        return self.cf + self.mid

    def channel(self, oversample=4.0, guard=1.3, atten_db=50.0):
        """The filters that bring this signal to baseband at about `oversample` times its width

        Shift + multi-stage decimation (rp.signal.channelize), then a sharp filter at the low rate to drop the neighbours
        the decimation lets through: passband to `guard`*width/2 (the pulse skirts reach past the 99% width).
        Returns Channel(front, Fs, decimation, delay, settle): `block + channel.front`.
        """
        from ...signal.filters import low_pass_kaiser
        from ...signal import channelize as _channel
        from ...base import Pipeline
        if self.Fs is None:
            raise ValueError("This signal doesn't know its capture's Fs: build it from rp.analysis.survey or find_signals")
        front, Fs, D = _channel(self.mid, self.width, self.Fs, oversample)
        if self.width*guard < Fs:
            sel = low_pass_kaiser(0.5*self.width*guard, 0.2*self.width, atten_db, Fs)
            front = Pipeline(front, sel)
        from ...base import Data
        delay = (Data(np.zeros(1, np.complex64), Fs=self.Fs) + front).meta.delay # (a stream of its own: front holds no state)
        return Channel(front, Fs, D, delay, int(np.ceil(2*delay*Fs)))

    def esn0_db(self, baud=None):
        """Es/N0 [dB] from the in-band SNR: the noise in the signal's width (snr) is that density times width, the
        symbol energy is the power over the symbol rate, so Es/N0 = snr + 10log10(width/baud). `baud` defaults to
        baud_est. The spectrum can't tell signal from noise inside the band, so this is the guide, the measured one
        (rp.sync.quality.measure on the locked symbols) the answer"""
        baud = baud or self.baud_est
        if baud is None or self.snr is None:
            return None
        return self.snr + 10*np.log10(self.width/baud)

    def estimate_baud(self, samps, min_strength=4.0):
        """Fill baud_est from samples of the capture (this signal is channelized first)"""
        from ...sync.rate import find_baud
        from ...base import Data
        ch = self.channel()
        x = (Data(np.asarray(samps), Fs=self.Fs) + ch.front).data
        if len(x) < 512:
            return None
        baud, strength = find_baud(x, ch.Fs)
        self.baud_strength = strength
        self.baud_est = baud if strength >= min_strength else None
        return self.baud_est

    def describe(self):
        bits = [f"{self.freq/1e6:+.4f} MHz", f"{self.width/1e3:.1f} kHz wide"]
        if self.snr is not None:
            bits.append(f"snr {self.snr:.1f} dB")
        if self.baud_est:
            bits.append(f"baud ~{self.baud_est/1e3:.1f} k")
        if self.duty is not None and self.duty < 0.9:
            bits.append(f"bursty ({self.duty*100:.0f}% on)")
        if not self.resolved:
            bits.append("unresolved")
        if self.clipped:
            bits.append("cut by band edge")
        if self.holds:
            bits.append(f"holds {self.holds} other signal{'s' if self.holds > 1 else ''}")
        if self.lines:
            bits.append(f"a comb of {self.lines} lines")
        if self.over is not None:
            bits.append(f"on the {self.over.width/1e3:.0f} kHz signal at {self.over.freq/1e6:+.4f} MHz")
        return ", ".join(bits)

    def __repr__(self):
        snr = "" if self.snr is None else f", snr {self.snr:.1f} dB"
        return f"<Signal {self.mid/1e6:+.4f} MHz, {self.width/1e3:.1f} kHz wide{snr}>"

class Survey(list):
    """The signals found in a capture (a list of Signal, by frequency), and the spectrum they came from

    Fs, cf: the capture's rates. nfft, K, bin_hz: the spectrum's size, averages and resolution
    freq, db: the spectrum [Hz from the centre, dB], floor: its noise floor [dB]
    head: the samples the survey looked at (a Device can't rewind: feed these first)
    """
    def __init__(self, signals=(), Fs=1.0, cf=0.0, nfft=0, K=0, freq=None, db=None, floor=0.0, head=None):
        super().__init__(signals)
        self.Fs, self.cf, self.nfft, self.K = Fs, cf, nfft, K
        self.bin_hz = Fs/nfft if nfft else 0.0
        self.freq, self.db, self.floor, self.head = freq, db, floor, head
        self.lost = []       # [(Signal, surveys since seen)]: what `match` remembers of signals that went missing
        self.next_id = None  # The id `match` gives the next new signal (ids are never reused)

    def pick(self, near=None, freq=None, index=None, strongest=False):
        """One signal. near: closest to this offset from the capture's centre [Hz], freq: closest to this absolute
        frequency [Hz], index: by position, strongest: highest in-band SNR, not total power, which a wide pedestal under the
        carriers would win (also the default for one argument less)"""
        if not len(self):
            raise ValueError("No signals to pick from")
        if index is not None:
            return self[index]
        if freq is not None:
            return min(self, key=lambda s: abs(s.freq - freq))
        if near is not None:
            return min(self, key=lambda s: abs(s.mid - near))
        return max(self, key=lambda s: s.snr if s.snr is not None else s.height)

    def match(self, previous, near=0.5, ratio=(0.75, 4/3), hold=8):
        """Carry signal ids over from the survey before this one of the same band (frame to frame): ids are a signal's
        own, stable however others come and go.

        A signal is the earlier one whose centre is within `near` of its width and whose width is within `ratio` of its own
        (the closest such); each earlier signal is matched once. One that went missing (not found for a survey or a few:
        too weak for a moment) is remembered for `hold` surveys and gets its id back when it returns. A new signal gets an
        id never used before in the chain of surveys, so a signal appearing never renumbers the others. Sets Signal.id in
        place (and `lost`, `next_id` for the next match) and returns self"""
        known = [(p, 0) for p in previous if p.id is not None] + list(getattr(previous, "lost", []))
        nxt = getattr(previous, "next_id", None)
        if nxt is None:
            nxt = max((p.id for p, _ in known), default=-1) + 1
        taken = set()
        for s in sorted(self, key=lambda s: -s.snr if s.snr is not None else -s.height): # Strongest choose first
            best = None
            for p, age in known:
                if p.id in taken:
                    continue
                d = abs(p.freq - s.freq)
                if d <= near*s.width and ratio[0] <= s.width/p.width <= ratio[1] and (best is None or (age, d) < best[0]):
                    best = ((age, d), p) # (one seen in the last survey before one remembered)
            if best is None:
                s.id, nxt = nxt, nxt + 1
            else:
                s.id = best[1].id
                taken.add(s.id)
        self.lost = [(p, age + 1) for p, age in known if p.id not in taken and age + 1 <= hold]
        self.next_id = nxt
        return self

    def plot(self, ax=None, chosen=None):
        """Draw the spectrum with every signal shaded and numbered. Returns the axes"""
        from ...viz.plots.survey import survey as _plot
        return _plot(ax, self, chosen)

    def choose(self):
        """Show the survey and wait for a click (or a number key) on the signal to process. Returns the Signal

        Without an interactive backend (CI, a script run headless) it can't ask: the strongest signal is taken and said so.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        if not len(self):
            raise ValueError("No signals to choose from")
        backend = matplotlib.get_backend().lower()
        try:
            from matplotlib.backends import backend_registry, BackendFilter
            headless = backend in backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)
        except ImportError: # Matplotlib before 3.9
            headless = backend in matplotlib.rcsetup.non_interactive_bk
        if headless:
            sig = self.pick(strongest=True)
            print(f"No interactive backend ({backend}): taking the strongest signal, {sig.describe()}")
            return sig
        fig, ax = plt.subplots(layout="constrained")
        self.plot(ax)
        ax.set_title("Click a signal (or press its number) to process it")
        picked = []
        def hit(sel):
            picked.append(sel)
            self.plot(ax, sel)
            fig.canvas.draw_idle()
        def on_click(ev):
            if ev.inaxes is ax and ev.xdata is not None:
                x = ev.xdata*1e6 - self.cf # The axis is MHz, absolute
                inside = [s for s in self if s.lo <= x <= s.hi]
                hit(min(inside or self, key=lambda s: abs(s.mid - x)))
        def on_key(ev):
            if ev.key and ev.key.isdigit() and int(ev.key) < len(self):
                hit(self[int(ev.key)])
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show(block=False)
        while not picked and plt.fignum_exists(fig.number):
            plt.pause(0.05)
        if not picked:
            raise RuntimeError("The survey was closed without choosing a signal")
        plt.pause(0.3) # Show the choice
        plt.close(fig)
        return picked[0]

    def __repr__(self):
        return f"<Survey {len(self)} signals, {self.Fs/1e6:g} MS/s at {self.cf/1e6:g} MHz, {self.bin_hz:.0f} Hz bins>"

def _segments(samps, nfft, overlap):
    """Windowed periodogram of each segment (rows)"""
    samps = np.asarray(samps)
    step = max(1, int(nfft*(1-overlap)))
    starts = np.arange(0, len(samps) - nfft + 1, step)
    if len(starts) == 0:
        raise ValueError(f"Need at least {nfft} samples")
    win = np.hanning(nfft)
    return np.stack([np.abs(np.fft.fft(samps[s:s+nfft]*win))**2 for s in starts]), np.sum(win**2)

def welch(samps, Fs: float = 1, nfft: int = 4096, overlap: float = 0.5):
    """Averaged periodogram (Hann window), fftshifted. Returns (power per bin, number of averages)"""
    seg, wp = _segments(samps, nfft, overlap)
    return np.fft.fftshift(seg.sum(axis=0)/(len(seg)*wp*Fs)), len(seg)

def noise_floor(db, iters=4, margin=3.0, start=None):
    """Robust noise floor [dB]: median, then median of what is within `margin` dB above it (signals drop out)

    start: where to begin instead of the median. A band mostly full of signal has a median inside the signals; starting
    from a low level makes the floor the quiet part, wherever the signals are
    """
    floor = np.median(db) if start is None else start
    for _ in range(iters):
        sel = db < floor + margin
        floor = np.median(db[sel])
    return floor



def measure_band(samps, Fs, freq, bw, cf=0.0, nfft=None, edge=0.005):
    """A band given by hand (a channel plan, a band a user typed) measured as find_signals measures what it finds: a
    Signal with power, SNR, height, peak and duty filled in, so typed and found signals compare alike

    freq: the band's centre [Hz, absolute: cf + offset], bw: its width [Hz], cf: the capture's centre frequency.
    The noise is the capture's floor (find's, per bin); the band is cut to the capture's edges (clipped)."""
    from .find import _nfft_for, fit_floor, spectrogram
    samps = np.asarray(samps)
    nfft = _nfft_for(len(samps)) if nfft is None else int(nfft)
    P, hop = spectrogram(samps, Fs, nfft, 0.5)
    fit = fit_floor(P, hop, edge)
    A = P.mean(axis=0, dtype=np.float64)
    bin_hz = Fs/nfft
    mid = freq - cf
    e0 = int(np.clip(np.floor((mid - bw/2)/bin_hz + nfft/2), 0, nfft - 1))
    e1 = int(np.clip(np.ceil((mid + bw/2)/bin_hz + nfft/2) - 1, e0, nfft - 1))
    clipped = mid - bw/2 < -Fs/2 or mid + bw/2 > Fs/2
    z = A[e0:e1+1]/fit.floor[e0:e1+1]
    n = e1 - e0 + 1
    snr = 10*np.log10(max(float(np.sum(z - 1.0)), 1e-30)/n)
    # Duty: the band's power frame by frame over the noise; halfway from the noise to its mean divides on from off. Too
    # weak or narrow to tell the swing from the estimate's own scatter: no answer
    on = (P[:, e0:e1+1]/fit.floor[None, e0:e1+1]).mean(axis=1)
    mean_p = float(on.mean())
    duty = float(np.mean(on > (1.0 + mean_p)/2)) if mean_p - 1.0 > 4*mean_p*np.sqrt(2/n) else None
    band = A[e0:e1+1]
    return Signal(((e0 + e1 + 1)/2 - nfft/2)*bin_hz, n*bin_hz, float(np.median(10*np.log10(fit.floor[e0:e1+1]))),
                  float(10*np.log10(max(float(z.max()), 1e-30))), float(snr), lo=(e0 - nfft/2)*bin_hz,
                  hi=(e1 + 1 - nfft/2)*bin_hz, power=float(10*np.log10(band.sum()*bin_hz + 1e-30)),
                  peak_freq=float((e0 + int(np.argmax(band)) - nfft/2)*bin_hz), duty=duty, resolved=n >= 8,
                  clipped=clipped, Fs=Fs, cf=cf, bin_hz=bin_hz)

_RETIRED = ("min_snr_db", "gap", "occupied", "min_band_snr_db", "edges", "edge_db")

def find_signals(samps, Fs: float = 1, nfft=None, cf=0.0, pfa=1e-3, edge=0.005, split_db=4.0, drop_db=10.0, min_bins=1,
                 min_across=8, nested=True, **retired):
    """The signals in a capture: rpps.analysis.spectrum.find.find (how it works is there), as a Survey of Signal sorted
    by frequency, ids 0, 1... in that order

    pfa: expected false boxes in a capture of pure noise; nfft: FFT size [N/16, up to 65536]; cf: the capture's centre
    frequency [Hz], for Signal.freq; edge: fraction of the band at each end not looked at (the receiver's roll-off is
    also found); split_db: least notch or step between two signals side by side [dB, 3/4 of it for a step]; drop_db: a
    carrier's edges are this far under its level; min_bins: narrowest signal; min_across: bins across under which its
    edges aren't trusted (Signal.resolved); nested: keep the boxes that hold other signals (a pedestal, a carrier a
    narrower one jams: Signal.holds). False: only what stands on them
    The thresholds of the rules this replaced (min_snr_db, gap, occupied, min_band_snr_db, edges, edge_db) are taken
    for one release with a DeprecationWarning, and do nothing: `pfa` sets what the detection lets through."""
    import warnings
    from .find import find
    bad = [k for k in retired if k not in _RETIRED]
    if bad:
        raise TypeError(f"find_signals() got unexpected keyword arguments {bad}")
    if retired:
        warnings.warn(f"find_signals: {sorted(retired)} no longer do anything (the finder's thresholds come from pfa); "
                      "they go in the next release", DeprecationWarning, stacklevel=2)
    sv = find(samps, Fs, cf=cf, nfft=nfft, pfa=pfa, edge=edge, split_db=split_db, drop_db=drop_db, min_bins=min_bins,
              min_across=min_across)
    if not nested:
        kept = [s for s in sv if not s.holds]
        for t in kept:
            if t.over is not None and t.over.holds:
                t.over = None
        sv[:] = kept
        for i, s in enumerate(sv):
            s.id = i
    return sv

def survey(source, Fs=None, cf=None, samples=2**17, more=2**20, baud=True, **kw):
    """Look at a source (rp.source.File, Device) or an array, and find the signals in it

    samples: how many samples to read. More means a finer spectrum, so close neighbours stay apart and the edges
        are right, whatever block size the receive chain will use later
    more: if some signal is too narrow for its bins, read up to this many (once) and look again
    baud: also estimate each signal's symbol rate (Signal.baud_est)
    The rest (pfa, split_db, ...) is for find_signals. Returns a Survey; Survey.head is what was read.
    """
    if hasattr(source, "peek"):
        Fs = source.Fs if Fs is None else Fs
        cf = source.cf if cf is None else cf
        head = source.peek(samples)
        Fs, cf = (source.Fs if Fs is None else Fs), (source.cf if cf is None else cf) # A device settles on its own
    else:
        if Fs is None:
            raise ValueError("An array needs Fs")
        head, cf = np.asarray(source), (0.0 if cf is None else cf)
    sv = find_signals(head, Fs, cf=cf, **kw)
    if more and hasattr(source, "peek") and len(head) < more and any(not s.resolved for s in sv):
        bigger = source.peek(more)
        if len(bigger) > len(head):
            head = bigger
            sv = find_signals(head, Fs, cf=cf, **kw)
    sv.head = head
    if baud:
        for s in sv:
            try:
                s.estimate_baud(head)
            except Exception: # No estimate is not a failed survey
                pass
        reject_spur_bauds(sv)
    return sv

def reject_spur_bauds(signals, tol=0.005, min_count=3, spread=1.5):
    """Drop baud estimates that are really a line in the capture, not the signal's symbol rate

    A symbol rate above the signal's width can't be (width is at least the symbol rate). A rate found again and again
    on signals of different widths is a spur of the capture (a clock coupling into every signal's |x|^2); it, and
    its half multiples, are dropped from every signal of this capture. Signals sharing a rate and width are carriers
    of one service and keep it.
    """
    sigs = [s for s in signals if s.baud_est]
    for s in sigs:
        if s.baud_est > 1.1*s.width:
            s.baud_est = None
    sigs = [s for s in sigs if s.baud_est]
    spurs = []
    for s in sigs:
        like = [t for t in sigs if abs(t.baud_est - s.baud_est) <= tol*s.baud_est]
        if len(like) >= min_count and max(t.width for t in like)/min(t.width for t in like) > spread:
            spurs.append(s.baud_est)
    if spurs:
        base = float(np.median(spurs))
        for s in sigs:
            n = s.baud_est/(base/2)
            if abs(n - round(n)) <= 2*tol*n and round(n) >= 1:
                s.baud_est = None
    return signals
