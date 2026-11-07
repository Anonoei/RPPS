import numpy as np

from collections import namedtuple

Channel = namedtuple("Channel", "front Fs decimation")
Channel.__doc__ = """What brings a Signal to baseband: `block + channel.front` is the signal at Fs `Fs` (the capture's Fs / `decimation`)"""

class Signal:
    """A signal found in a spectrum

    mid: centre [Hz, from the capture's centre], freq: centre [Hz] absolute (cf + mid)
    width: occupied bandwidth [Hz] (the span holding `occupied` of its power), lo/hi: its edges [Hz from the capture's centre]
    bottom: noise floor [dB] (global), height: peak above the floor [dB], snr: in-band power over noise power in that band [dB]
    power: total in-band power [dB], peak_freq: strongest bin [Hz from the capture's centre]
    duty: fraction of the time it is on (1: continuous, less: bursty)
    resolved: there were enough bins across it to trust the edges. False means read more samples
    baud_est / baud_strength: symbol rate from the |x|^2 line of the channelized signal (None: no clear line)
    Fs, cf, bin_hz: the capture's rates and the spectrum's resolution
    baud: an alias of width (older callers)
    """
    def __init__(self, mid, width, bottom, height, snr=None, *, lo=None, hi=None, power=None, peak_freq=None,
                 duty=None, resolved=True, Fs=None, cf=0.0, bin_hz=None):
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
        self.Fs, self.cf, self.bin_hz = Fs, cf, bin_hz
        self.baud_est = None
        self.baud_strength = 0.0

    @property
    def freq(self):
        return self.cf + self.mid

    @property
    def baud(self):
        return self.width

    def channel(self, oversample=4.0, guard=1.3, atten_db=50.0):
        """The filters that bring this signal to baseband at about `oversample` times its width

        Shift + multi-stage decimation (rp.filters.channel), then a sharp filter at the low rate to drop the neighbours
        the decimation lets through: passband to `guard`*width/2 (the pulse skirts reach past the 99% width).
        Returns Channel(front, Fs, decimation): `block + channel.front`.
        """
        from ..filters import channel as _channel, low_pass_kaiser
        from ..base import Pipeline
        if self.Fs is None:
            raise ValueError("This signal doesn't know its capture's Fs: build it from rp.dsp.survey or find_signals")
        front, Fs, D = _channel(self.mid, self.width, self.Fs, oversample)
        if self.width*guard < Fs:
            sel = low_pass_kaiser(0.5*self.width*guard, 0.2*self.width, atten_db, Fs)
            front = Pipeline(front, sel)
        return Channel(front, Fs, D)

    def esn0_db(self, baud=None):
        """Es/N0 [dB] from the in-band SNR: the noise in the signal's width (snr) is that density times width, the
        symbol energy is the power over the symbol rate, so Es/N0 = snr + 10log10(width/baud). `baud` defaults to
        baud_est. The spectrum can't tell signal from noise inside the band, so this is the guide, the measured one
        (rp.analysis.quality.measure on the locked symbols) the answer"""
        baud = baud or self.baud_est
        if baud is None or self.snr is None:
            return None
        return self.snr + 10*np.log10(self.width/baud)

    def estimate_baud(self, samps, min_strength=4.0):
        """Fill baud_est from samples of the capture (this signal is channelized first)"""
        from .rate import find_baud
        from ..base import Data
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

    def pick(self, near=None, freq=None, index=None, strongest=False):
        """One signal. near: closest to this offset from the capture's centre [Hz], freq: closest to this absolute
        frequency [Hz], index: by position, strongest: highest in-band power (also the default for one argument less)"""
        if not len(self):
            raise ValueError("No signals to pick from")
        if index is not None:
            return self[index]
        if freq is not None:
            return min(self, key=lambda s: abs(s.freq - freq))
        if near is not None:
            return min(self, key=lambda s: abs(s.mid - near))
        return max(self, key=lambda s: s.snr if s.snr is not None else s.height)

    def plot(self, ax=None, chosen=None):
        """Draw the spectrum with every signal shaded and numbered. Returns the axes"""
        from ..viz.plots.survey import survey as _plot
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

def noise_floor(db, iters=4, margin=3.0):
    """Robust noise floor [dB]: median, then median of what is within `margin` dB above it (signals drop out)"""
    floor = np.median(db)
    for _ in range(iters):
        sel = db < floor + margin
        floor = np.median(db[sel])
    return floor



def _split(sm, lo, hi, depth):
    """Split the run lo..hi (inclusive) of the smoothed spectrum `sm` [dB] at valleys at least `depth` dB below the peaks
    on both sides. Returns [(lo, hi), ...], neighbours sharing their valley bin"""
    seg = sm[lo:hi+1]
    cuts = []
    pk, vmin, vi = seg[0], None, 0
    for i, v in enumerate(seg):
        if vmin is None:
            if v > pk:
                pk = v
            elif v < pk - depth:
                vmin, vi = v, i
        elif v < vmin:
            vmin, vi = v, i
        elif v > vmin + depth: # Back up the other side of a valley
            cuts.append(lo + vi)
            pk, vmin = v, None
    edges = [lo] + cuts + [hi]
    return [(edges[i], edges[i+1]) for i in range(len(edges) - 1)]

def find_signals(samps, Fs: float = 1, nfft=None, min_snr_db=None, gap=None, occupied=0.99, min_bins=4, edge=0.02,
                 cf=0.0, split_db=3.0, min_across=8):
    """Find signals in a spectrum

    Averages periodograms (more samples means a finer, quieter spectrum), finds the noise floor, and takes every
    run of bins above floor + threshold as a signal. The threshold is a few standard deviations of the averaged
    estimate, so long captures can see weaker and narrower signals. A run with a valley between peaks is split into
    the signals either side of it.

    nfft: FFT size [largest power of two giving at least ~16 averages, up to 65536]
    min_snr_db: detection threshold above the floor [4 sigma of the estimate, at least 1 dB]
    gap: bins of dip tolerated inside one signal [nfft/256]
    occupied: fraction of the signal's power that defines its width (edges are found from the cumulative power)
    min_bins: narrowest signal reported
    edge: fraction of the band at each end to ignore (receiver filter roll-off)
    cf: the capture's centre frequency [Hz], for Signal.freq
    split_db: depth of a valley that separates two signals [dB]
    min_across: bins across a signal below which its edges are not trusted (Signal.resolved False)
    Returns a Survey (a list of Signal sorted by frequency).
    """
    samps = np.asarray(samps)
    N = len(samps)
    if nfft is None:
        nfft = 1 << int(np.clip(np.floor(np.log2(max(N//16, 64))), 6, 16))
    seg, wp = _segments(samps, nfft, 0.5)
    K = len(seg)
    pw = np.fft.fftshift(seg.sum(axis=0)/(K*wp*Fs))
    segs = np.fft.fftshift(seg/(wp*Fs), axes=1)
    db = 10*np.log10(pw + 1e-30)
    sigma = 4.34/np.sqrt(K) # Standard deviation of a K-average periodogram, in dB
    floor = noise_floor(db)
    thr = floor + (min_snr_db if min_snr_db is not None else max(4*sigma, 1.0))
    bin_hz = Fs/nfft
    freq = (np.arange(nfft) - nfft/2)*bin_hz

    # Light smoothing so noise peaks inside a signal don't split it
    k = max(3, nfft//1024*2 + 1)
    sm = 10*np.log10(np.convolve(pw, np.ones(k)/k, mode="same") + 1e-30)
    above = sm > thr
    gap = max(2, nfft//256) if gap is None else gap
    idx = np.nonzero(above)[0]
    runs = []
    for i in idx:
        if runs and i - runs[-1][1] <= gap:
            runs[-1][1] = i
        else:
            runs.append([i, i])

    # Valleys are looked for in a spectrum smoothed a little more, so ripple doesn't split a signal
    kv = max(5, nfft//512*2 + 1)
    sv = 10*np.log10(np.convolve(pw, np.ones(kv)/kv, mode="same") + 1e-30)
    depth = max(split_db, 4*4.34/np.sqrt(K*kv))
    parts = []
    for lo, hi in runs:
        parts.extend(_split(sv, lo, hi, depth) if hi - lo + 1 >= 3*kv else [(lo, hi)])

    floor_lin = 10**(floor/10)
    quiet = ~above
    out = []
    for lo, hi in parts:
        if hi - lo + 1 < min_bins:
            continue
        # Widen a little: the smoothed edge sits inside the skirts
        lo, hi = max(0, lo - k//2), min(nfft-1, hi + k//2)
        w = hi - lo + 1
        flank = np.concatenate((pw[max(0, lo - w):lo][quiet[max(0, lo - w):lo]], pw[hi+1:hi+1+w][quiet[hi+1:hi+1+w]]))
        local = np.median(flank) if len(flank) >= 8 else floor_lin # The noise right beside this signal
        excess = np.clip(pw[lo:hi+1] - local, 0, None)
        if excess.sum() <= 0:
            continue
        c = np.cumsum(excess)/excess.sum()
        e0 = lo + int(np.searchsorted(c, (1-occupied)/2))
        e1 = lo + int(np.searchsorted(c, 1-(1-occupied)/2))
        if e0 < edge*nfft or e1 > (1-edge)*nfft:
            continue
        n = max(e1 - e0 + 1, 1)
        band = pw[e0:e1+1]
        snr = 10*np.log10(max(np.sum(band - local), 1e-30)/(local*len(band)))
        on = segs[:, e0:e1+1].sum(axis=1) # Band power in each time segment, against the noise in the same band
        duty = float(np.mean(on > 2*local*len(band)))
        out.append(Signal(
            ((e0 + e1)/2 - nfft/2)*bin_hz, n*bin_hz, floor, float(db[lo:hi+1].max() - floor), float(snr),
            lo=(e0 - nfft/2)*bin_hz, hi=(e1 + 1 - nfft/2)*bin_hz, power=float(10*np.log10(band.sum()*bin_hz + 1e-30)),
            peak_freq=float(freq[e0 + int(np.argmax(band))]), duty=duty, resolved=n >= min_across,
            Fs=Fs, cf=cf, bin_hz=bin_hz))
    return Survey(out, Fs, cf, nfft, K, freq, db, float(floor))

def survey(source, Fs=None, cf=None, samples=2**17, more=2**20, baud=True, **kw):
    """Look at a source (rp.serial.File, Device) or an array, and find the signals in it

    samples: how many samples to read. More means a finer spectrum, so close neighbours stay apart and the edges
        are right, whatever block size the receive chain will use later
    more: if some signal is too narrow for its bins, read up to this many (once) and look again
    baud: also estimate each signal's symbol rate (Signal.baud_est)
    The rest (min_snr_db, occupied, ...) is for find_signals. Returns a Survey; Survey.head is what was read.
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
    return sv
