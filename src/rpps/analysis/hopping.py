"""Frequency hopping: the hops, their dwell, the hop set and its spacing, and the signal de-hopped

    h = rp.analysis.hopping.track(x, Fs)        # Hops: dwells (start, end, frequency), dwell time, channels, spacing
    y = rp.analysis.hopping.dehop(x, Fs, h)     # every dwell mixed to 0 Hz and joined: one channel's signal to analyse

A hopper is a narrow signal that jumps: over the capture the spectrum looks wide and flat, at any instant it is one
narrow peak. A short-time spectrum (frames much shorter than a dwell, bins much narrower than the hop set) has, frame by
frame, its strongest bin over the frame's floor where the signal is; a run of frames at one frequency (within a channel
spacing) is a dwell. The dwell time is the median of whole dwells (the first and last are cut by the capture); the hop
set is the frequencies the dwells visit, clustered; the spacing is the greatest common step between them (the median of
the smallest differences, then refined over all of them).
"""
import numpy as np

from .cyclic import noise_floor

class Dwell:
    __slots__ = ("start", "end", "freq", "power")
    def __init__(self, start, end, freq, power):
        self.start, self.end, self.freq, self.power = start, end, freq, power

    def __repr__(self):
        return f"Dwell({self.start*1e3:.3f}..{self.end*1e3:.3f} ms at {self.freq:+.1f} Hz)"

class Hops:
    """Result of `track`: dwells (Dwell: start, end s, freq Hz), dwell (s, median of whole dwells), rate (hops/s),
    channels (Hz, the hop set's frequencies), spacing (Hz), active (share of frames with a signal), hopping (more than
    one channel visited, dwells shorter than a quarter of the capture, the signal at an instant under half as wide as over
    the capture)"""
    __slots__ = ("dwells", "dwell", "rate", "channels", "spacing", "active", "hopping")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        if not self.hopping:
            return f"not hopping ({len(self.channels)} channel(s), {len(self.dwells)} dwell(s))"
        return (f"hopping: {len(self.channels)} channels {self.spacing/1e3:.3f} kHz apart over "
                f"{(max(self.channels) - min(self.channels))/1e3:.1f} kHz, dwell {self.dwell*1e3:.3f} ms "
                f"({self.rate:.1f} hops/s), {len(self.dwells)} dwells, active {100*self.active:.0f}% of the time")

    __repr__ = __str__

def _spectrogram(x, nfft, hop):
    n = (len(x) - nfft)//hop + 1
    idx = np.arange(nfft)[None, :] + hop*np.arange(n)[:, None]
    S = np.abs(np.fft.fftshift(np.fft.fft(x[idx]*np.hanning(nfft), axis=1), axes=1))**2
    return S

def track(x, Fs, nfft=256, hop=None, snr_db=6.0, merge=None, width=None):
    """Dwells of a frequency-hopping signal (see the module). nfft: frame length (shorter than a dwell, bins narrower than
    the channel spacing), hop: frame step (default nfft/2). snr_db: a frame's peak over its median bin to count.
    merge: frequency (Hz) within which consecutive frames are one dwell (default 2 bins). width: bins each side of a
    frame's peak its frequency is the power centroid of (default nfft/32)"""
    x = np.asarray(x, dtype=np.complex128)
    hop = hop or nfft//2
    S = _spectrogram(x, nfft, hop)
    f = (np.arange(nfft) - nfft//2)*Fs/nfft
    k = np.argmax(S, axis=1)
    peak = S[np.arange(len(S)), k]
    floor = np.median(S, axis=1)
    on = peak > floor*10**(snr_db/10)
    # each frame's frequency: the power centroid within `width` bins of its peak (an FSK's tones, a PSK's spectrum,
    # sit around the dwell's frequency: the peak bin alone jumps between them)
    width = width or max(2, nfft//32)
    b = np.arange(-width, width + 1)
    idx = np.clip(k[:, None] + b[None, :], 0, nfft - 1)
    P = S[np.arange(len(S))[:, None], idx] - floor[:, None]
    P = np.maximum(P, 0)
    fr = (np.sum(P*idx, axis=1)/np.maximum(P.sum(axis=1), 1e-300) - nfft//2)*Fs/nfft
    t = (np.arange(len(S))*hop + nfft/2)/Fs
    T = len(x)/Fs
    # 1: the channels: active frames' frequencies clustered (a histogram's peaks, a bin apart at least)
    merge = merge or 2*Fs/nfft
    fa = np.sort(fr[on])
    chans = []
    for fq in fa:
        if chans and fq - chans[-1][-1] < merge:
            chans[-1].append(fq)
        else:
            chans.append([fq])
    chans = [c for c in chans if len(c) >= 2] # a frame alone is noise or a transition
    channels = [float(np.median(c)) for c in chans]
    spacing = 0.0
    if len(channels) > 1:
        diffs = np.diff(channels)
        s0 = float(np.median(diffs[diffs < 1.5*np.min(diffs)])) # the smallest step, as a start
        steps = np.round((np.array(channels) - channels[0])/s0) # every channel a whole number of steps from the first
        ok = steps > 0
        spacing = float(np.sum((np.array(channels)[ok] - channels[0])*steps[ok])/np.sum(steps[ok]**2)) if ok.any() else s0
    # 2: every active frame to its nearest channel; a dwell is a run of frames on one channel
    ch = np.full(len(S), -1)
    if channels:
        cf = np.array(channels)
        near = np.argmin(np.abs(fr[:, None] - cf[None, :]), axis=1)
        close = np.abs(fr - cf[near]) < max(merge, 0.4*spacing if spacing else merge)
        ch = np.where(on & close, near, -1)
    dwells = []
    i = 0
    while i < len(S):
        if ch[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < len(S) and ch[j + 1] == ch[i]:
            j += 1
        if j > i: # a dwell is two frames or more
            dwells.append(Dwell(float(t[i] - hop/(2*Fs)), float(t[j] + hop/(2*Fs)), float(channels[ch[i]]),
                                float(np.median(peak[i:j + 1]))))
        i = j + 1
    whole = [d.end - d.start for d in dwells[1:-1]] or [d.end - d.start for d in dwells]
    dwell = float(np.median(whole)) if whole else 0.0
    # a hopper is narrow at any instant and wide over time: a wide signal's peak bin wanders too, without hopping
    long = S.mean(axis=0)
    wide_bins = np.count_nonzero(long > noise_floor(long)*10**(snr_db/10))
    frame_bins = np.median([np.count_nonzero(S[i] > floor[i]*10**(snr_db/10)) for i in np.flatnonzero(on)[:400]]) \
        if on.any() else wide_bins
    narrow = frame_bins < 0.5*wide_bins
    hopping = len(channels) > 1 and dwell < T/4 and narrow
    return Hops(dwells=dwells, dwell=dwell, rate=1/dwell if dwell else 0.0, channels=channels, spacing=spacing,
                active=float(np.mean(on)), hopping=hopping)

def dehop(x, Fs, hops, guard=0.0):
    """The signal with every dwell mixed to 0 Hz, the dwells joined in order (a `guard` share of each dwell's ends
    dropped: the synthesizer settling); between dwells nothing. Returns complex samples at Fs"""
    x = np.asarray(x, dtype=np.complex128)
    out = []
    for d in hops.dwells:
        a = int(round(d.start*Fs))
        b = int(round(d.end*Fs))
        g = int(guard*(b - a))
        a, b = max(a + g, 0), min(b - g, len(x))
        if b <= a:
            continue
        n = np.arange(a, b)
        out.append(x[a:b]*np.exp(-2j*np.pi*d.freq/Fs*n))
    return np.concatenate(out) if out else np.zeros(0, complex)
