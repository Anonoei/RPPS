"""What kind of waveform a signal is, when it isn't a single linearly modulated carrier: tones, OFDM, bursts

    w = rp.analysis.waveform.classify(x, Fs)    # x: the signal at baseband (rp.signal.baseband / Signal.channel)
    print(w)                                    # e.g. "OFDM 15 kHz subcarriers ... LTE / NR 15 kHz, normal CP; bursts ..."

Each test is cheap and says why it holds:
  tones: nearly all the power in a few lines narrower than a few Hz (a beacon, a pilot, an unmodulated carrier; two
      tones beat, and the beat looks like a symbol rate to the cyclostationary search)
  ofdm: the cyclic prefix repeats the end of each symbol, so the signal correlates with itself at the useful symbol length
      Tu (1/subcarrier spacing), about Tcp/(Tu+Tcp) of its power; x(t)x*(t+Tu) then pulses once a symbol, at 1/(Tu+Tcp).
      Many subcarriers add up to a Gaussian (E|x|^4/E|x|^2^2 near 2), which tells OFDM from a single carrier whose frame
      repeats (a preamble and pilots correlate at the frame length too). (Tu, Tcp) names the system: `SYSTEMS`
  bursts: the power switches on and off; the period of the switching (a TDD frame, a TDMA slot)
  envelope: E|x|^4/E|x|^2^2: 1 for a constant envelope (FM, FSK, CPM, a carrier), 1.3-1.7 for filtered PSK, 2 for noise
      or OFDM
"""
import numpy as np

# (name, useful symbol us, cyclic prefix us): one row per prefix length of a system
SYSTEMS = []

def _add(name, tu, cps):
    for label, cp in cps:
        SYSTEMS.append((f"{name}{label}", tu, cp))

_add("LTE / NR 15 kHz", 1e3/15, [(", normal CP", 144/2048*1e3/15), (", extended CP", 512/2048*1e3/15)])
_add("LTE MBSFN 7.5 kHz", 1e3/7.5, [(", extended CP", 1024/4096*1e3/7.5)])
_add("NR 30 kHz", 1e3/30, [(", normal CP", 144/2048*1e3/30)])
_add("NR 60 kHz", 1e3/60, [(", normal CP", 144/2048*1e3/60), (", extended CP", 512/2048*1e3/60)])
_add("NR 120 kHz", 1e3/120, [(", normal CP", 144/2048*1e3/120)])
_add("Wi-Fi 802.11a/g/n/ac (312.5 kHz)", 3.2, [(", long GI", 0.8), (", short GI", 0.4)])
_add("Wi-Fi 802.11ax (78.125 kHz)", 12.8, [(", GI 0.8 us", 0.8), (", GI 1.6 us", 1.6), (", GI 3.2 us", 3.2)])
_add("WiMAX (10.94 kHz)", 1e3/10.9375, [(f", CP {n}", 1e3/10.9375*f) for n, f in (("1/4", 1/4), ("1/8", 1/8), ("1/16", 1/16), ("1/32", 1/32))])
_add("DAB mode I", 1000, [("", 246)])
_add("DAB mode II", 250, [("", 62)])
_add("DAB mode III", 125, [("", 31)])
_add("DAB mode IV", 500, [("", 123)])
for _bw, _T in ((8, 7/64), (7, 1/8), (6, 7/48), (5, 7/40)): # DVB-T/T2: elementary period by channel width (MHz)
    for _k in (1, 2, 4, 8, 16, 32):
        _tu = 1024*_k*_T
        if _k in (2, 8):
            _add(f"DVB-T/T2 {_bw} MHz {_k}k", _tu, [(f", CP {n}", _tu*f) for n, f in (("1/4", 1/4), ("1/8", 1/8), ("1/16", 1/16),
                                                                                    ("1/32", 1/32), ("19/128", 19/128), ("19/256", 19/256), ("1/128", 1/128))])
        else:
            _add(f"DVB-T2 {_bw} MHz {_k}k", _tu, [(f", CP {n}", _tu*f) for n, f in (("1/4", 1/4), ("1/8", 1/8), ("1/16", 1/16), ("1/32", 1/32),
                                                                                 ("19/128", 19/128), ("19/256", 19/256), ("1/128", 1/128))])
for _m, _tu in ((1, 252), (2, 504), (3, 1008)): # ISDB-T, 6 MHz
    _add(f"ISDB-T mode {_m}", _tu, [(f", CP {n}", _tu*f) for n, f in (("1/4", 1/4), ("1/8", 1/8), ("1/16", 1/16), ("1/32", 1/32))])
for _k in (8, 16, 32): # ATSC 3.0, 6 MHz (6.912 MS/s)
    _add(f"ATSC 3.0 {_k}k", _k*1024/6.912, [(f", GI {g} samples", g/6.912) for g in (192, 384, 512, 768, 1024, 1536, 2048, 2432,
                                                                                      3072, 3648, 4096, 4864)])
_add("DOCSIS 3.1 (50 kHz)", 20, [(f", CP {n}/4096", 20*n/4096) for n in (192, 256, 512, 768, 1024)])
_add("DOCSIS 3.1 (25 kHz)", 40, [(f", CP {n}/8192", 40*n/8192) for n in (192, 256, 512, 768, 1024)])

def envelope(x):
    """E|x|^4/E|x|^2^2: 1 constant envelope, about 1.3-1.7 filtered PSK, 2 Gaussian (noise, OFDM, many carriers)"""
    a2 = np.abs(np.asarray(x))**2
    return float(np.mean(a2**2)/np.mean(a2)**2)

def autocorr(x):
    """|sum x(n+L) x*(n)| / sum |x|^2 for every lag L >= 0 (by FFT)"""
    n = len(x)
    X = np.fft.fft(x, 2*n)
    r = np.fft.ifft(np.abs(X)**2)[:n]
    return np.abs(r)/np.real(r[0])

class Tones:
    """`tones`: lines [(Hz from the centre, share of the power)], strongest first; share: of all of it, in them"""
    __slots__ = ("lines", "share", "resolution")
    def __init__(self, lines, share, resolution):
        self.lines, self.share, self.resolution = lines, share, resolution

    def __str__(self):
        if len(self.lines) == 1:
            what = f"an unmodulated carrier at {self.lines[0][0]:+.1f} Hz"
        else:
            sp = np.diff(sorted(f for f, _ in self.lines))
            what = (f"{len(self.lines)} tones at " + ", ".join(f"{f:+.1f}" for f, _ in sorted(self.lines)) + " Hz"
                    + (f" ({sp[0]:.1f} Hz apart)" if len(sp) == 1 else ""))
        return f"{what}: {self.share*100:.1f}% of the power in lines under {self.resolution*4:.1f} Hz wide"

    __repr__ = __str__

def tones(x, Fs, share=0.9, max_lines=4, width_bins=2):
    """Lines holding `share` of the power, at most `max_lines` of them, each within `width_bins` bins: Tones, or None.
    The resolution is Fs/len(x): a long stretch tells a carrier from a narrow modulated signal"""
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    if n < 256:
        return None
    P = np.abs(np.fft.fft(x*np.hanning(n)))**2
    total = P.sum()
    f = np.fft.fftfreq(n, 1/Fs)
    left, lines = P.copy(), []
    for _ in range(max_lines):
        k = int(np.argmax(left))
        idx = np.arange(k - width_bins - 1, k + width_bins + 2) % n # The Hann window spreads a line over 3 bins
        lines.append((float(f[k]), float(left[idx].sum()/total)))
        left[idx] = 0.0
        got = sum(s for _, s in lines)
        if got >= share:
            keep = [(fr, s) for fr, s in lines if s > 0.02*got]
            return Tones(keep, got, Fs/n)
        if lines[-1][1] < 0.02: # The rest is spread: not lines
            break
    return None

class OFDM:
    """`ofdm`: tu, tcp (s), strength (the correlation at Tu: about tcp/(tu+tcp) at a high SNR), ratio (over other lags),
    line (the symbol-rate line over its floor), envelope, systems [(name, misfit)] that fit, best first. comb: when the
    lag found repeats (correlating at 2x and 3x it too), it isn't the symbol but a pilot comb's period inside symbols:
    every comb-th subcarrier used, so the symbol is comb x tu (systems then fit that); None for plain OFDM"""
    __slots__ = ("tu", "tcp", "strength", "ratio", "line", "envelope", "systems", "comb")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def scs(self):
        return 1/self.tu

    def __str__(self):
        sy = f": {self.systems[0][0]}" if self.systems else ": no system in the table has these"
        if self.comb:
            return (f"OFDM pilot comb: the signal repeats every {self.tu*1e6:.2f} us within symbols, every "
                    f"{_nth(self.comb)} subcarrier of a {self.scs/self.comb/1e3:.4g} kHz grid (sync / reference symbols){sy}")
        return (f"OFDM {self.scs/1e3:.4g} kHz subcarriers (Tu {self.tu*1e6:.2f} us), CP {self.tcp*1e6:.2f} us "
                f"({self.tcp/self.tu:.3f} of Tu){sy}")

    __repr__ = __str__

def _nth(m):
    return f"{m}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(m, 'th')}"

COMB = 0.3 # The correlation at twice the lag, over the peak's, of a comb

def ofdm(x, Fs, tu_min=2e-6, tu_max=5e-3, min_ratio=8.0, min_envelope=1.8, max_samples=2**21, max_rate_line=8.0):
    """OFDM in baseband samples x at Fs: OFDM, or None (no lag stands out, no symbol period, or not Gaussian)

    tu_min/tu_max: useful symbol lengths searched. min_ratio: the correlation at Tu over the median of the other lags.
    max_rate_line: a |x|^2 line this far over its floor (rpps.analysis.find_baud) is a single carrier's symbol rate"""
    x = np.asarray(x[:max_samples], dtype=np.complex128)
    x = x - x.mean()
    n = len(x)
    env = envelope(x)
    lo, hi = max(int(tu_min*Fs), 8), min(int(tu_max*Fs), n//4)
    if env < min_envelope or hi <= lo + 8:
        return None
    # A single carrier has a symbol-rate line in |x|^2 (near its bandwidth); OFDM's subcarriers have none there
    from .rate import find_baud
    if find_baud(x[:2**18], Fs)[1] >= max_rate_line:
        return None
    r = autocorr(x)
    floor = float(np.median(r[lo:hi]))
    # The band's own correlation falls from 1 at lag 0 over about 1/bandwidth, longer for a sharp edge (a signal cut by
    # the capture's band edge rings): search from where it has fallen to the floor
    near = np.nonzero(r[lo:hi] < 3*floor)[0]
    if not len(near):
        return None
    lo += int(near[0])
    k = int(np.argmax(r[lo:hi])) + lo
    peak = float(r[k])
    if peak < min_ratio*floor:
        return None
    # A prefix peak is isolated; the band shape's own correlation near zero lag falls smoothly
    side = np.concatenate((r[max(lo, int(k*0.75)):max(lo, int(k*0.9))], r[int(k*1.1):min(n, int(k*1.25))]))
    if len(side) and peak < min_ratio*np.median(side):
        return None
    a, b, c = r[k - 1], r[k], r[k + 1]
    L = k + (0.5*(a - c)/(a - 2*b + c) if a - 2*b + c else 0.0)
    tu = L/Fs
    # The symbol period: the line of x(t) x*(t+Tu) between 1/(1.5 Tu) and 1/(1.003 Tu). Its phase is constant (the
    # carrier offset over Tu), so the line is at +-1/Ts: both sides are added
    Li = int(round(L))
    g = x[Li:]*np.conj(x[:-Li])
    m = 4*len(g)
    G = np.abs(np.fft.fft(g - g.mean(), m))
    G = G + G[(-np.arange(m)) % m]
    f = np.fft.fftfreq(m, 1/Fs)
    band = np.nonzero((f > 1/(1.5*tu)) & (f < 1/(1.003*tu)))[0]
    if not len(band):
        return None
    kk = band[np.argmax(G[band])]
    line = float(G[kk]/np.median(G[band]))
    if line < 6:
        return None
    tcp = 1/f[kk] - tu
    # A cyclic prefix correlates at Tu only; a comb (every m-th subcarrier used) repeats m times within its symbol, and
    # correlates at 2x and 3x the lag too. Its symbol is m x the lag, and the prefix found is no prefix
    comb = None
    if 2*Li + 2 < n and r[2*Li - 2:2*Li + 3].max() > COMB*peak:
        for m in range(2, 9):
            if any(abs(stu*1e-6 - m*tu)/(m*tu) < 0.01 for _, stu, _ in SYSTEMS):
                comb = m
                break
        else:
            comb = 0
    if comb is None and not 0.005 < tcp/tu < 0.4: # Not a prefix: a frame that repeats
        return None
    sy = []
    for name, stu, scp in SYSTEMS:
        if comb:
            e = abs(stu*1e-6 - comb*tu)/(comb*tu)
            if e < 0.01:
                sy.append((name.split(",")[0] + f", pilots on every {_nth(comb)} subcarrier", e))
            continue
        e_tu, e_cp = abs(stu*1e-6 - tu)/tu, abs(scp*1e-6 - tcp)/tu
        if e_tu < 0.01 and e_cp < 0.012:
            sy.append((name, e_tu + e_cp))
    sy = sorted(dict(sorted(sy, key=lambda t: -t[1])).items(), key=lambda t: t[1])
    return OFDM(tu=tu, tcp=tcp, strength=peak, ratio=peak/floor, line=line, envelope=env, systems=sy,
                comb=comb if comb else None)

class Bursts:
    """`bursts`: duty (fraction of the time on), period (s, None if the switching isn't periodic), on (s a period)"""
    __slots__ = ("duty", "period", "on")
    def __init__(self, duty, period, on):
        self.duty, self.period, self.on = duty, period, on

    def __str__(self):
        per = f", every {self.period*1e3:.3f} ms ({self.on*1e3:.3f} ms on)" if self.period else ", not periodic"
        return f"bursts: on {self.duty*100:.0f}% of the time{per}"

    __repr__ = __str__

def bursts(x, Fs, smooth=2e-6, contrast=10.0, min_off=0.03):
    """Whether the power switches on and off: Bursts, or None (steady, or too weak to tell). smooth: the power is averaged
    over this long; contrast: the on level over the off level needed (power ratio); min_off: the least time off"""
    p = np.abs(np.asarray(x))**2
    w = max(2, int(Fs*smooth))
    e = np.convolve(p, np.ones(w)/w, mode="valid")
    lo, hi = np.percentile(e, 2), np.percentile(e, 98)
    if hi < contrast*lo:
        return None
    on = e > np.sqrt(lo*hi)
    duty = float(on.mean())
    if duty > 1 - min_off:
        return None
    dec = max(1, int(Fs*10e-6)) # The pattern at 100 kS/s is enough to find a period of a millisecond or more
    o = on[:len(on)//dec*dec].reshape(-1, dec).mean(1)
    o = o - o.mean()
    fs2 = Fs/dec
    r = autocorr(o) if len(o) > 8 else np.zeros(1)
    lo_, hi_ = int(0.5e-3*fs2), min(len(r)//2, int(50e-3*fs2))
    period = None
    if hi_ > lo_ + 2:
        k = int(np.argmax(r[lo_:hi_])) + lo_
        if r[k] > 0.5:
            period = k/fs2
    return Bursts(duty, period, duty*period if period else None)

class Waveform:
    """`classify`: tones, ofdm, bursts (each None when not), envelope (E|x|^4/E|x|^2^2)"""
    __slots__ = ("tones", "ofdm", "bursts", "envelope")
    def __init__(self, tones=None, ofdm=None, bursts=None, envelope=None):
        self.tones, self.ofdm, self.bursts, self.envelope = tones, ofdm, bursts, envelope

    @property
    def kind(self):
        """"tones", "ofdm", "constant envelope", or None (nothing here says what it is)"""
        if self.tones is not None:
            return "tones"
        if self.ofdm is not None:
            return "ofdm"
        if self.envelope is not None and self.envelope < 1.15:
            return "constant envelope"
        return None

    def __str__(self):
        bits = []
        if self.tones is not None:
            bits.append(str(self.tones))
        if self.ofdm is not None:
            bits.append(str(self.ofdm))
        if self.kind == "constant envelope":
            bits.append(f"constant envelope (E|x|^4/E|x|^2^2 = {self.envelope:.2f}): FM, FSK or CPM, not filtered PSK")
        if self.bursts is not None:
            bits.append(str(self.bursts))
        return "; ".join(bits) if bits else f"nothing recognized (E|x|^4/E|x|^2^2 = {self.envelope:.2f})"

    __repr__ = __str__

def classify(x, Fs):
    """Tones, OFDM, bursts and the envelope of baseband samples x at Fs (see the module): a Waveform"""
    x = np.asarray(x)
    if len(x) < 256:
        return Waveform()
    t = tones(x, Fs)
    b = None if t is not None else bursts(x, Fs) # Tones beat: their envelope swings without switching off
    o = None if t is not None else ofdm(x, Fs)
    # A burst's on and off move the envelope statistic: judge it over the on part only
    on = x
    if b is not None:
        p = np.abs(x)**2
        on = x[p > 0.1*np.percentile(p, 98)]
    return Waveform(t, o, b, envelope(on) if len(on) > 64 else None)
