"""Which feed each carrier is on, from a two-feed capture: its polarization, the cross-polar leakage, frequency reuse

    python projects/reference/polarization.py h.sigmf v.sigmf               # two recordings, one per feed
    python projects/reference/polarization.py h.sigmf                       # one: the feed its receive chain names
    python projects/reference/polarization.py --record cap --freq 1000 --fs 20 --names H V   # two channels of one radio

One feed shows one polarization: what a single capture says about a signal's is only the feed it came through (the
chain's `polarization`, from pysdrlib's receive context). Two feeds at once (a dual-output LNB on a two-channel radio,
the same instants on both) say more, per carrier, over its band:

    C/N on each feed       its power over each feed's own noise floor: unequal chain gains cancel (the two feeds are
                           taken to have similar noise temperatures, as a dual-output LNB's do)
    coherence              how much of the carrier on one feed is the same signal as on the other: the cross-spectrum
                           |S_ab|^2 over what independent noise gives (S_aa S_bb / M over M frames). It is measured bin by
                           bin, so a delay or a phase between the feeds doesn't matter, and it finds a copy far under the
                           other feed's noise

    a / b      on one feed. XPD: how far under it the other feed's copy is (the leakage: a "carrier" a survey of the other
               feed lists there is this one); a lower bound when no copy is seen
    both       the same signal on both, within 10 dB: a slanted linear polarization, or a circular one on linear feeds
    reuse      on both, not the same signal: two carriers at one frequency, one per polarization

Feeds are named from the receive contexts (`chain.polarization`: "H", "V", "RHCP", "LHCP"), else `names`, else a/b.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

ONE_FEED_DB = 10.0  # a carrier more than this over the other feed's copy is on one feed
REUSE_COH = 0.5     # on both feeds, less of it coherent than this: two carriers
Z_SEEN = 6.0        # standard deviations over noise for a copy to count as seen (power or coherent)


@dataclass
class Pol:
    """One carrier on two feeds. sig: the survey's Signal (from either feed). on: "a", "b", "both" or "reuse", named by
    `name`. cn: (C/N on a, on b) in dB, None where nothing is over that feed's noise. ratio: C/N on a over b [dB].
    coherence: the share of the carriers that is the same signal on both (1: one signal; None: not on both). xpd: dB the
    weaker feed's copy is under the stronger (a, b only); xpd_bound: no copy seen, xpd is how far under one would have
    been seen. leak_seen: the other feed's survey listed the copy as a carrier of its own (signals the caller gave: False)"""
    sig: object
    on: str
    names: tuple
    cn: tuple
    ratio: Optional[float]
    coherence: Optional[float]
    xpd: Optional[float] = None
    xpd_bound: bool = False
    leak_seen: bool = False

    @property
    def name(self):
        return {"a": self.names[0], "b": self.names[1]}.get(self.on, self.on)

    def __str__(self):
        db = lambda v: "-" if v is None else f"{v:5.1f}"
        s = (f"{self.sig.freq/1e6:12.6f} MHz {self.sig.width/1e3:9.1f} kHz  C/N {db(self.cn[0])} / {db(self.cn[1])} dB  "
             f"{self.name:>6}")
        if self.xpd is not None:
            s += f"  XPD {'>' if self.xpd_bound else ''}{self.xpd:.1f} dB"
        if self.coherence is not None:
            s += f"  coherence {self.coherence:.2f}"
        if self.leak_seen:
            other = self.names[1] if self.on == "a" else self.names[0]
            s += f"  (on {other}: its leakage)"
        return s


def spectra(xa, xb, Fs, nfft):
    """(S_aa, S_bb, S_ab, floor_a, floor_b, M): averaged auto and cross spectra (power per Hz, fftshifted) over M
    frames that don't overlap (so the noise's cross term has exactly 1/M of S_aa S_bb), and each feed's noise floor
    (rp.analysis.spectrum.find's)"""
    from rpps.analysis.spectrum.find import _hann_win, fit_floor
    M = min(len(xa), len(xb))//nfft
    if M < 8:
        raise ValueError(f"{min(len(xa), len(xb))} samples make {M} frames of {nfft}: need 8 or more")
    win = _hann_win(nfft)
    k = 1/(np.sum(win**2)*Fs)
    X = [np.fft.fftshift(np.fft.fft(np.asarray(x[:M*nfft], np.complex128).reshape(M, nfft)*win, axis=1), axes=1)
         for x in (xa, xb)]
    Pa, Pb = (np.abs(x)**2*k for x in X)
    Sab = np.mean(X[0]*np.conj(X[1]), axis=0)*k
    floors = [fit_floor(P.astype(np.float32), nfft).floor for P in (Pa, Pb)]
    return Pa.mean(axis=0), Pb.mean(axis=0), Sab, floors[0], floors[1], M


INNER = 0.8         # the share of a carrier's band measured: its edges are where a neighbour's skirt is

def _band(sig, Fs, nfft):
    """The bins of the inner part of the signal's band"""
    bin_hz = Fs/nfft
    cut = (1 - INNER)/2*(sig.hi - sig.lo)
    i0 = int(math.ceil((sig.lo + cut)/bin_hz + nfft/2))
    i1 = int(math.floor((sig.hi - cut)/bin_hz + nfft/2)) + 1
    i0, i1 = max(i0, 0), min(i1, nfft)
    if i1 <= i0:
        c = int(round(sig.mid/bin_hz + nfft/2))
        i0, i1 = max(c, 0), min(c + 1, nfft)
    return slice(i0, i1)


def measure(sig, S, Fs, names=("a", "b")):
    """The Pol of one signal, from `spectra`'s S over its band"""
    Saa, Sbb, Sab, Na, Nb, M = S
    band = _band(sig, Fs, len(Saa))
    saa, sbb, sab, na, nb = Saa[band], Sbb[band], Sab[band], Na[band], Nb[band]
    n = len(saa)
    ca, cb = np.sum(saa - na), np.sum(sbb - nb)  # carrier power over each floor (per Hz: the bin width cancels)
    qa, qb = ca/np.sum(na), cb/np.sum(nb)        # C/N on each feed
    q_min = Z_SEEN/math.sqrt(M*n)                # the least C/N a power measurement sees over noise
    D = np.abs(sab)**2 - saa*sbb/M               # the cross-spectrum over what independent noise gives
    z = np.sum(D)/math.sqrt(np.sum((saa*sbb/M)**2))
    coherent = z > Z_SEEN
    seen_a, seen_b = qa > q_min, qb > q_min
    db = lambda q: 10*math.log10(q) if q > 0 else None
    cn = (db(qa) if seen_a else None, db(qb) if seen_b else None)
    ratio = db(qa/qb) if seen_a and seen_b else None
    coh = None
    if seen_a and seen_b:
        coh = float(np.clip(np.sum(D)/np.sum(np.clip(saa - na, 0, None)*np.clip(sbb - nb, 0, None)), 0.0, 1.0))
    if seen_a and seen_b and (not coherent or coh < REUSE_COH):
        return Pol(sig, "reuse", names, cn, ratio, coh)
    if seen_a and seen_b and abs(ratio) < ONE_FEED_DB:
        return Pol(sig, "both", names, cn, ratio, coh)
    if not (seen_a or seen_b):
        return Pol(sig, "a" if qa >= qb else "b", names, cn, ratio, coh)  # (nothing over either floor here)
    strong = "a" if (qa > qb) else "b"
    q_s, s_strong, n_weak = (qa, saa, nb) if strong == "a" else (qb, sbb, na)
    # The weak feed's copy: the part of it the strong feed's signal explains (|S_ab|^2 / S_strong), over its noise
    leak = np.sum(D/s_strong)/np.sum(n_weak)
    if coherent and leak > 0:
        xpd, bound = 10*math.log10(q_s/leak), False
    else:
        xpd, bound = 10*math.log10(q_s/min(q_min, Z_SEEN/(M*math.sqrt(n)))), True
    weak = "b" if strong == "a" else "a"
    return Pol(sig, strong, names, cn, ratio, coh, xpd, bound, leak_seen=weak in getattr(sig, "feeds", ()))


def _merge(sa, sb):
    """The signals of both feeds' surveys, one per carrier: boxes overlapping by half the narrower are one (the one with
    the higher SNR kept). Each gets `feeds`: the feeds whose survey found it"""
    out = []
    for s in sa:
        s.feeds = ("a",)
        out.append(s)
    for s in sb:
        hit = next((i for i, t in enumerate(out) if min(s.hi, t.hi) - max(s.lo, t.lo) >= 0.5*min(s.width, t.width)), None)
        if hit is None:
            s.feeds = ("b",)
            out.append(s)
            continue
        feeds = out[hit].feeds + ("b",)
        if (s.snr or -99) > (out[hit].snr or -99):
            out[hit] = s
        out[hit].feeds = feeds
    return sorted(out, key=lambda s: s.mid)


def _names(a, b, names):
    if names is not None:
        return tuple(names)
    got = tuple(getattr(getattr(getattr(s, "context", None), "chain", None), "polarization", None) for s in (a, b))
    return got if all(got) and got[0] != got[1] else ("a", "b")


def _check(a, b, Fs, cf):
    """Fs and cf of the pair; ValueError when the feeds weren't received alike"""
    fa, fb = getattr(a, "Fs", Fs), getattr(b, "Fs", Fs)
    if fa is None or fb is None:
        raise ValueError("arrays need Fs")
    if abs(fa - fb) > 1e-9*fa:
        raise ValueError(f"the feeds' sample rates differ: {fa:g} and {fb:g} Hz")
    ca, cb = getattr(a, "cf", cf or 0.0), getattr(b, "cf", cf or 0.0)
    if abs(ca - cb) > 1e-3:
        raise ValueError(f"the feeds' centre frequencies differ: {ca:.0f} and {cb:.0f} Hz")
    ra, rb = getattr(a, "context", None), getattr(b, "context", None)
    if ra is not None and rb is not None:
        if ra.freq is not None and rb.freq is not None and abs(ra.freq - rb.freq) > 1e-3:
            raise ValueError(f"the receive contexts' frequencies differ: {ra.freq:.0f} and {rb.freq:.0f} Hz")
    return float(fa), float(ca)


def _nfft(sigs, Fs, N):
    narrow = min((s.width for s in sigs), default=Fs/64)
    want = 2**int(math.ceil(math.log2(max(Fs/max(narrow/32, 1e-9), 256))))
    return int(min(want, 2**15, 2**int(math.log2(max(N//64, 256)))))


def polarization(a, b, signals=None, Fs=None, cf=None, samples=2**20, nfft=None, names=None):
    """[Pol] for every carrier in a two-feed capture

    a, b: Sources (rp.source.SigMF, File, Memory) or arrays (then Fs, and cf), the same instants on both. signals: the
    Signals to measure; None: each feed's survey (rp.analysis.survey), merged, so a carrier on one feed only is found.
    samples: read from each. nfft: the spectra's bins (None: 32 across the narrowest carrier, at least 64 frames).
    names: the feeds' names (None: their receive chains' polarization, else "a", "b")"""
    import rpps as rp
    Fs, cf = _check(a, b, Fs, cf)
    xa = a.peek(samples) if hasattr(a, "peek") else np.asarray(a)[:samples]
    xb = b.peek(samples) if hasattr(b, "peek") else np.asarray(b)[:samples]
    names = _names(a, b, names)
    if signals is None:
        signals = _merge(rp.analysis.survey(xa, Fs=Fs, cf=cf, more=0, baud=False),
                         rp.analysis.survey(xb, Fs=Fs, cf=cf, more=0, baud=False))
    if not signals:
        return []
    S = spectra(xa, xb, Fs, nfft or _nfft(signals, Fs, min(len(xa), len(xb))))
    return [measure(s, S, Fs, names) for s in signals]


def record(prefix, freq, Fs, seconds, gain=0.4, names=None):
    """Record channels 0 and 1 of one radio (radio_io: $RPPS_RADIO, PYSDRLIB_REMOTE) into `prefix`.<name>.sigmf, each
    with its channel's receive context. names: the feed on each channel ("H", "V"), set as its chain's polarization"""
    import rpps as rp
    import radio_io
    sdr = radio_io.open_radio(cf=freq, fs=Fs, gain=gain)
    try:
        if sdr.caps.channels.get("rx", 1) < 2:
            raise RuntimeError(f"{sdr.NAME} has one receive channel: two feeds need two")
        ants = list(sdr.rx_antennas())
        if len(ants) > 1:
            sdr.set_rx_antenna(ants[0], channel=0)
            sdr.set_rx_antenna(ants[1], channel=1)
        for ch in (0, 1):
            radio_io.set_gain(sdr, gain, channel=ch)
        if names:
            _set_feeds(sdr, names)
        sdr.start_rx(channels=(0, 1))
        sdr.flush()
        want, n = int(seconds*Fs), 0
        tags = names or ("a", "b")
        paths = [f"{prefix}.{t}.sigmf" for t in tags]
        sinks = [rp.source.SigMFSink(p, Fs=sdr.get_rx_sample_rate(), cf=sdr.get_rx_freq(), context=sdr.get_rx_context(ch))
                 for ch, p in enumerate(paths)]
        try:
            while n < want:
                x = sdr.read(min(2**17, want - n), timeout=3.0, channel=(0, 1))
                for s, row in zip(sinks, x):
                    s.write(row)
                n += x.shape[1]
        finally:
            for s in sinks:
                s.close()
        sdr.stop_rx()
    finally:
        sdr.close()
    return paths


def _set_feeds(sdr, names):
    """Each channel gets a chain naming its feed (`Chain(channel=)`: a B210's two channels both call their port RX2),
    the radio's own chain for that port kept with its polarization set. A pysdrlib without Chain.polarization: the
    files' contexts don't name the feeds, so pass `names` when reading them"""
    import dataclasses
    from pysdrlib.base import caps
    fields = {f.name for f in dataclasses.fields(caps.Chain)}
    if not {"polarization", "channel"} <= fields:
        print("pysdrlib's Chain has no polarization yet: the recordings won't name their feeds (read them with --names)")
        return
    chains = list(sdr.get_chains() or [])
    out = [c for c in chains if c.direction != "rx" or c.channel not in (0, 1)]
    for ch, pol in enumerate(names):
        port = sdr.get_rx_antenna(ch)
        mine = next((c for c in chains if c.direction == "rx" and c.port in (port, None) and c.channel in (ch, None)), None)
        out.append(dataclasses.replace(mine, port=port, channel=ch, polarization=pol) if mine else
                   caps.Chain(direction="rx", port=port, channel=ch, polarization=pol))
    sdr.set_chains(out)


def main():
    import argparse
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E401,E402,F401
    import rpps as rp
    ap = argparse.ArgumentParser(description="Which feed each carrier of a two-feed capture is on")
    ap.add_argument("files", nargs="*", help="one SigMF recording per feed (one: its feed from its receive chain)")
    ap.add_argument("--names", nargs=2, help="the feeds' names (H V, RHCP LHCP), when the recordings don't say")
    ap.add_argument("--samples", type=int, default=2**20, help="samples read from each feed (default 1048576)")
    ap.add_argument("--record", metavar="PREFIX", help="record two channels of the radio first (PREFIX.<name>.sigmf)")
    ap.add_argument("--freq", type=float, help="centre frequency over the air [MHz] (--record)")
    ap.add_argument("--fs", type=float, default=20.0, help="sample rate [MS/s] (--record, default 20)")
    ap.add_argument("--seconds", type=float, default=0.5, help="seconds to record (--record, default 0.5)")
    ap.add_argument("--gain", type=float, default=0.4, help="receive gain, a fraction of the radio's range (--record)")
    a = ap.parse_args()
    files = list(a.files)
    if a.record:
        if a.freq is None:
            ap.error("--record needs --freq")
        files = record(a.record, a.freq*1e6, a.fs*1e6, a.seconds, a.gain, a.names)
        print("recorded " + ", ".join(files))
    if len(files) == 1:
        src = rp.source.SigMF(files[0])
        ctx = src.context
        pol = getattr(getattr(ctx, "chain", None), "polarization", None)
        print(f"{files[0]}: " + (f"the {pol} feed (one feed: the signals' polarization isn't observable)" if pol else
                                 "its receive chain names no feed" if ctx is not None else "no receive context"))
        return 0
    if len(files) != 2:
        ap.error("give one or two recordings (or --record)")
    srcs = [rp.source.SigMF(f) for f in files]
    pols = polarization(*srcs, samples=a.samples, names=a.names)
    if not pols:
        print("no carriers found on either feed")
        return 1
    names = pols[0].names
    print(f"feeds: a = {names[0]} ({files[0]}), b = {names[1]} ({files[1]})")
    for p in pols:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
