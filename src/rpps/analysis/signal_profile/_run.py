"""Profiling a capture: every signal through the chain, in parallel (rpps.analysis.signal_profile)"""
import numpy as np

from ._receive import receive_steps
from ._chain import _Chain
from ._result import FOUND, GUESS, SignalProfile, Step, UNSOLVED

def profile(source, signals=None, freq=None, blocks=450, block=16384, blind=True, survey=None, symbols=2**18,
            max_blocks=None, progress=None, jobs=1, hint=None):
    """Profiles of signals in a capture: [SignalProfile], in the order chosen

    source: an rp.source Source (File, Memory: a live device is best captured into Memory first, as the chains run slower
        than it does). It is read once, `blocks` blocks of `block` samples, every chosen signal processed along the way.
    symbols, max_blocks: a signal whose code is not decided yet and that has fewer than `symbols` symbols keeps reading
        past `blocks`, up to `max_blocks` (default 8 x blocks): a narrow signal gets symbols enough for the code scans
        (and the layers inside a code) instead of the few thousand a fixed stretch of a wide capture holds
    signals: which ones: None (every one the survey finds), an index or indices into the survey, or Signal objects
    freq: instead, the one nearest this frequency (Hz from the capture's centre)
    survey: the rp.analysis.survey of `source` already made (the indices refer to it; it isn't surveyed again)
    blind: run the blind tools where no definition fits (tens of seconds a signal)
    progress: called as progress(stage, done, total, left) as the work goes: stage "read" (blocks read of the stretch
        planned: `blocks`, then `max_blocks` once some chains read on; `left` the chains still reading), "build" (profiles
        built of the signals), and "again" / "build again" for a second pass at a rate the cyclic spectrum gives
    jobs: profile the signals in this many processes (forked, each with its share of the signals and its own reading of
        the source from where this one starts: a File seeks back, Memory starts over; Linux and macOS). With jobs > 1
        `progress` is called as ("parallel", groups done, jobs, groups left)
    hint: hint(signal) -> None or {"baud": the symbol rate, "codes": the convolutional definitions to try (names in
        config/fec/conv), "blind": False to skip the blind tools}: what an earlier look at the same signal (another
        capture of it) found, so it is not searched for again"""
    from ..spectrum.signal import survey as _survey
    tell = progress or (lambda *a: None)
    sv = survey if survey is not None else None
    if signals is not None and not isinstance(signals, (int, np.integer)) and all(hasattr(s, "channel") for s in signals):
        picked = list(signals)
        sv = sv if sv is not None else picked
    else:
        sv = sv if sv is not None else list(_survey(source))
        if freq is not None:
            picked = [min(sv, key=lambda s: abs(s.mid - freq))] if len(sv) else []
        elif signals is None:
            picked = list(sv)
        else:
            idx = [signals] if isinstance(signals, (int, np.integer)) else list(signals)
            picked = [sv[i] for i in idx]
    index = {id(s): i for i, s in enumerate(sv)}
    if jobs > 1 and len(picked) > 1:
        return _parallel(source, picked, sv, jobs, tell, dict(blocks=blocks, block=block, blind=blind, symbols=symbols,
                                                               max_blocks=max_blocks, hint=hint))
    merged = _merge(source, picked, sv) # (one spread signal over several boxes: their union)
    for m in merged:
        if getattr(m, "merged", None) and id(m) not in index:
            index[id(m)] = min((i for i, b in enumerate(sv) if (b.mid, b.width) in m.merged[0]), default=None)
    picked = merged
    refit = _refit(source, picked) # (each process its own: the survey's boxes are what it is given)
    for old, new in zip(picked, refit):
        if new is not old:
            index[id(new)] = index.get(id(old))
    picked = refit

    cap = max(blocks, max_blocks if max_blocks is not None else 8*blocks)

    def chains_for(sigs, bauds, mods=None):
        """Chains for these signals, their channels from one rp.signal.FFTBank (one FFT of the capture a hop for all
        of them), and which bank channel each reads; mods: the constellations each one's Acquire tries (None: all)"""
        from ...signal import FFTBank
        from ..spectrum.signal import Channel
        live = [s for s in sigs if not getattr(s, "holds", 0)]
        bank = FFTBank(source.Fs, [(s.mid, s.width) for s in live]) if live else None
        slot, out = {}, []
        for s, b, m in zip(sigs, bauds, mods or [None]*len(sigs)):
            ch = None
            if bank is not None and not getattr(s, "holds", 0):
                j = len(slot)
                ch = Channel(None, bank.Fs_out[j], bank.D[j], 0.0, 0)
            h = (hint(s) if hint is not None else None) or {}
            c = _Chain(s, source.Fs, blind and h.get("blind", True), baud=b or h.get("baud"), ch=ch, codes=h.get("codes"),
                       mods=m)
            if ch is not None:
                slot[id(c)] = j
            out.append(c)
        return out, bank, slot

    def run(chains, bank, slot, stage="read"):
        fed = [ch for ch in chains if not ch.dead and id(ch) in slot]
        if bank is None:
            return
        def give(ch, y, last=False): # A chain's channel samples as the bank gives them (its stages don't mind the cut)
            if len(y) or last:
                ch.step(y, last=last)
        for i, blk in enumerate(source(block)):
            if i >= cap:
                break
            if i >= blocks: # Past the common stretch: only the chains still short of a decided code
                for ch in [ch for ch in fed if not ch.wants(symbols)]:
                    give(ch, np.zeros(0, np.complex64), last=True)
                    fed.remove(ch)
                if not fed:
                    return
            outs = bank.step(blk.data, active=[slot[id(ch)] for ch in fed])
            for ch in fed:
                give(ch, outs[slot[id(ch)]])
            tell(stage, i + 1, blocks if i < blocks else cap, len(fed))
        outs = bank.flush(active=[slot[id(ch)] for ch in fed]) # The end of the stream for those still reading
        for ch in fed:
            give(ch, outs[slot[id(ch)]], last=True)

    def build(ch):
        s = ch.sig
        j = index.get(id(s))
        name = f"signal {j if j is not None else '?'}: {s.mid/1e6:+.3f} MHz, {s.width/1e3:.0f} kHz wide"
        try:
            p = ch.profile(name)
        except Exception as e:
            p = SignalProfile(name, s)
            p.add("profile", "building the profile", UNSOLVED, note=f"{type(e).__name__}: {e}")
            return p
        receive_steps(p, source)
        if getattr(s, "merged", None) and p.steps:
            parts, R = s.merged
            why = (f"each shows one rate, {R/1e3:.6g} kHz, far under its width (a spreading code's period)" if R else
                   "OFDM on one subcarrier grid and one symbol clock across them")
            p.steps.insert(1, Step("box", f"{len(parts)} of the survey's boxes as one signal, {s.width/1e3:.0f} kHz wide: {why}",
                                   GUESS, note="boxes " + ", ".join(f"{m/1e6:+.3f} MHz {w/1e3:.0f} kHz" for m, w in parts)))
        if getattr(s, "refit", None) and p.steps:
            m0, w0 = s.refit
            p.steps.insert(1, Step("box", f"fitted to the signal: centre {(s.mid - m0)/1e3:+.1f} kHz, {s.width/1e3:.0f} kHz wide "
                                          f"(the survey's box {w0/1e3:.0f} kHz)", GUESS,
                                   note="a fine spectrum of the capture (its 95th percentile over time), edges 10 dB under the top"))
        return p

    chains, bank, slot = chains_for(picked, [None]*len(picked))
    run(chains, bank, slot)
    out = []
    for k, ch in enumerate(chains):
        out.append(build(ch))
        tell("build", k + 1, len(chains), len(chains) - k - 1)
    # Where the chain found no rate (or a doubtful one) and the cyclic spectrum sees one: a second pass at that rate
    again, parts, extra = {}, {}, []
    notched = set() # Boxes split at notches a fine spectrum shows: carriers, each listed
    for k, (ch, p) in enumerate(zip(chains, out)):
        subs = ch.carriers() if not ch.dead and not getattr(ch.sig, "merged", None) else [] # (a merged box: its dips are its own)
        if subs:
            parts[k] = [(s, None) for s in subs]
            notched.add(k)
            continue
        baud = ch.cyclic_rate(p) if not p.solved else None
        if baud:
            subs = ch.split(baud) # A box holding carriers side by side: each of them on its own (its rate acquired
            if len(subs) > 1 or (subs and subs[0].width < 0.7*ch.sig.width): # alone in its channel: the box's line
                parts[k] = [(s, None) for s in subs] # can be a beat between them), or the one carrier a too-wide box holds
            elif not (ch.stream.det is not None and abs(baud - ch.stream.det.baud) < 0.01*baud): # (not the same again)
                again[k] = (ch.sig, baud)
    todo = list(again.values()) + [sb for v in parts.values() for sb in v]
    if todo:
        redo, bank2, slot2 = chains_for([sb[0] for sb in todo], [sb[1] for sb in todo])
        run(redo, bank2, slot2, stage="again")
        split_chains = redo[len(again):]
        again = dict(zip(again, redo[:len(again)]))
        i = 0
        for k, subs in parts.items():
            mine = split_chains[i:i + len(subs)]
            i += len(subs)
            built = [build(c) for c in mine]
            what = (f"one of {len(subs)} carriers the survey's box held side by side" if len(subs) > 1 else
                    "the one carrier in a survey box wider than it")
            why = ("a fine spectrum of the box shows notches between them" if k in notched else
                   "its cyclic spectrum's rate needs well under the box")
            for q, c in zip(built, mine):
                q.steps.insert(1, Step("split", f"{what} ({why})", GUESS))
            box = out[k].name.split(":")[0] # "signal 16": each carrier in it is "signal 16.1", "signal 16.2"...
            for j, q in enumerate(built):
                q.name = f"{box}.{j + 1}: {q.signal.mid/1e6:+.3f} MHz, {q.signal.width/1e3:.0f} kHz wide"
            best = max(range(len(built)), key=lambda j: _reach(built[j]))
            first = out[k]
            # A box the notches split is not a signal itself, unless it solved as one and no piece did (a dip in one
            # carrier's spectrum)
            cut = k in notched and not (first.solved and not any(q.solved for q in built))
            if cut or _further(built[best], first):
                out[k] = built[best]
            # The rest, where they solved or went further than the box did (a piece that got nowhere is a lump of a
            # carrier's spectrum, or one the box's own profile already said as much of)
            # (carriers a notch separates are each a signal: all listed)
            extra += [q for j, q in enumerate(built) if out[k] is not q and (cut or q.solved or _further(q, first))]
        for j, (k, ch) in enumerate(again.items()):
            q = build(ch)
            tell("build again", j + 1, len(again), len(again) - j - 1)
            if _further(q, out[k]):
                q.steps.insert(1, Step("rate", f"{ch.baud/1e3:.6g} kBd from the cyclic spectrum (rpps.analysis.spectrum.cyclic)",
                                       GUESS, note="the first pass found " + (out[k].steps[1].what if len(out[k].steps) > 1
                                                                              else "nothing")))
                out[k] = q
    # A good lock no code fits: other readings of it (the runner-up constellation, 2x and 1/2 the rate), the one that
    # gets furthest kept (rpps._effort "retries")
    from ... import _effort
    tries = [(k, b, m) for k, ch in enumerate(chains) if k not in again and k not in parts
             for b, m in ch.retries(out[k], _effort.get("retries"))]
    if tries:
        redo, bank3, slot3 = chains_for([chains[k].sig for k, _, _ in tries], [b for _, b, _ in tries],
                                        [m for _, _, m in tries])
        run(redo, bank3, slot3, stage="retry")
        for j, ((k, b, m), c) in enumerate(zip(tries, redo)):
            q = build(c)
            tell("build retry", j + 1, len(tries), len(tries) - j - 1)
            if (q.solved or q.profile_name) and _further(q, out[k]): # (a code or a name: no more steps before the same end)
                what = f"{m[0].upper()} instead" if m else f"{b/1e3:.6g} kBd ({b/chains[k].stream.det.baud:g}x the rate first read)"
                q.steps.insert(1, Step("retry", what, GUESS, note="the first reading locked but no code fit: " +
                                       (out[k].unsolved.what if out[k].unsolved is not None else "")))
                out[k] = q
    out = _named(source, out + extra)
    return out

def _named(source, profs, samples=2**22):
    """Each profile no definition named, tried against the shipped profiles of a fixed rate its box may be
    (_framed.shipped: each on its own channel, named when its own checks verify): a new profile from its channel step
    on, the one whose checks verified named (the linear chain's reading as a note)"""
    from ._framed import shipped
    todo = [k for k, p in enumerate(profs) if not p.profile_name and p.signal is not None
            and not getattr(p.signal, "holds", 0) and getattr(p.signal, "width", None)]
    if not todo or not hasattr(source, "peek"):
        return profs
    try:
        x = np.asarray(source.peek(samples))
    except Exception:
        return profs
    for k in todo:
        p = profs[k]
        try:
            got = shipped(x, float(source.Fs), p.signal.mid, p.signal.width, budget=samples)
        except Exception as e:
            p.add("profile", "the shipped profiles of a fixed rate", UNSOLVED, note=f"{type(e).__name__}: {e}")
            continue
        if not got:
            continue
        name, params, ev = got[0]
        import copy
        sig = copy.copy(p.signal) # (the band the profile decoded, which the box may be a piece of)
        sig.mid, sig.width = float(ev["at"]), float(ev["width"])
        sig.lo, sig.hi = sig.mid - sig.width/2, sig.mid + sig.width/2
        q = SignalProfile(p.name, sig)
        q.symbols, q.det, q.fec = p.symbols, p.det, p.fec
        q.steps = [st for st in p.steps if st.kind in ("channel", "box", "split")][:3] or p.steps[:1]
        checked = ", ".join(f"{s} {ps}/{ps + fs}" for s, (ps, fs) in sorted(ev["checks"].items()))
        others = [t[0] for t in got[1:]]
        q.add("profile", f"{name}: its own receive, from the capture's samples, verifies ({sig.mid/1e6:+.4f} MHz, "
                         f"{sig.width/1e3:.0f} kHz: the band it decoded; the box {p.signal.mid/1e6:+.4f} MHz, "
                         f"{p.signal.width/1e3:.0f} kHz)", FOUND,
              candidates=[(t, 0.0, 0.0) for t in others],
              note=f"checks passed: {checked}" + (f"; the linear chain read: {p.unsolved.kind}: {p.unsolved.what}"
                                                   if p.unsolved is not None else ""))
        q.profile_name, q.profile_params = name, params
        profs[k] = q
    return profs

_JOB = None # What a forked worker of `_parallel` profiles (inherited, not pickled)

def _band(x, Fs, mid, width):
    """x shifted by -mid, filtered to `width` (in frequency) and decimated to about 1.3 x width: (samples, their rate)"""
    y = x*np.exp(-2j*np.pi*mid/Fs*np.arange(len(x)))
    Y = np.fft.fft(y)
    Y[np.abs(np.fft.fftfreq(len(y), 1/Fs)) > width/2] = 0
    D = max(1, int(Fs/(1.3*width)))
    return np.fft.ifft(Y)[::D], Fs/D

def _merge(source, sigs, sv, samples=2**20, gap=0.25, same=0.01, spread=2.0, ofdm_gap=0.5):
    """One spread signal the survey cut into several boxes, profiled as one: the run of adjacent boxes (of the survey `sv`:
    gaps under `gap` of the narrower) round each signal of `sigs` whose cyclic spectra (rpps.analysis.spectrum.cyclic)
    show one rate, the same within `same`, at under 1/`spread` of each box's width. A linear carrier's box is about 1.2
    to 1.35 x its rate; a direct-sequence signal's spectrum has dips (an 802.11b Barker code sums to -1: a dip at its
    centre) the survey splits it at, and every piece shows the spreading code's period (1 MHz, its 11 Mchip/s 4.4 x
    narrower than a piece). Carriers side by side at one rate each fill their own box, and stay apart. Returns `sigs`, a
    signal in such a run as a copy over the union (`merged`: the boxes' (mid, width) and the rate), once. Or OFDM on
    one subcarrier grid and symbol clock (`_one_ofdm`; gaps up to `ofdm_gap` of the wider: subcarriers left empty;
    `merged` then holds no rate)"""
    import copy
    if not sigs or not hasattr(source, "peek") or len(sv) < 2:
        return list(sigs)
    boxes = sorted((s for s in sv if getattr(s, "width", None) and not getattr(s, "holds", 0)), key=lambda s: s.mid)
    def cut(bs, g, of=min):
        out, cur = [], bs[:1]
        for b in bs[1:]:
            a = cur[-1]
            if b.lo - a.hi < g*of(a.width, b.width):
                cur.append(b)
            else:
                out.append(cur)
                cur = [b]
        return out + ([cur] if cur else [])
    keys = {(s.mid, s.width) for s in sigs}
    runs = [r for r in cut(boxes, ofdm_gap, max) if len(r) > 1 and any((q.mid, q.width) in keys for q in r)]
    if not runs:
        return list(sigs)
    try:
        x = np.asarray(source.peek(samples), dtype=np.complex128)
    except Exception:
        return list(sigs)
    if len(x) < 8192:
        return list(sigs)
    Fs = float(source.Fs)
    union = {}
    for r in runs:
        got = _one_ofdm(x, Fs, r)
        if got is not None:
            for part in got:
                m = copy.copy(max(part, key=lambda b: getattr(b, "snr", None) or 0.0))
                m.lo, m.hi = min(b.lo for b in part), max(b.hi for b in part)
                m.mid, m.width = (m.lo + m.hi)/2, m.hi - m.lo
                m.merged = ([(b.mid, b.width) for b in part], None)
                for b in part:
                    union[(b.mid, b.width)] = m
            continue
        for r in cut(r, gap):
            if len(r) < 2:
                continue
            _spread(x, Fs, r, same, spread, union)
    out, seen = [], set()
    for s in sigs:
        m = union.get((s.mid, s.width))
        if m is None:
            out.append(s)
        elif id(m) not in seen:
            seen.add(id(m))
            out.append(m)
    return out

def _spread(x, Fs, r, same, spread, union):
    """_merge's spread-spectrum test on a run of boxes: into `union` when they show one rate far under their widths"""
    import copy
    from ..spectrum.cyclic import features
    rates = []
    for b in r:
        y, F = _band(x, Fs, b.mid, b.width)
        try:
            f = features(y, F)
        except ValueError:
            f = None
        rates.append(None if f is None else f.baud)
    if any(v is None for v in rates):
        return
    R = float(np.median(rates))
    if all(abs(v/R - 1) < same for v in rates) and all(b.width > spread*R for b in r):
        m = copy.copy(max(r, key=lambda b: getattr(b, "snr", None) or 0.0))
        m.lo, m.hi = min(b.lo for b in r), max(b.hi for b in r)
        m.mid, m.width = (m.lo + m.hi)/2, m.hi - m.lo
        m.merged = ([(b.mid, b.width) for b in r], R)
        for b in r:
            union[(b.mid, b.width)] = m

def _one_ofdm(x, Fs, run, grid=0.05, clock=0.05, bin_s=0.5e-6):
    """The runs of `run` (adjacent boxes, in order) that are one OFDM signal: [[boxes]] of two or more, None when no two
    are. Each box's cyclic prefix (rpps.analysis.spectrum.waveform.ofdm) gives its Tu (a system's, when one fits), and
    its correlation at Tu (an exact lag: a phase ramp in frequency) the subcarriers' offset, modulo their spacing, from
    the capture's centre (the box's shift added back), and when its symbols start: that correlation's magnitude pulses
    in each prefix. Boxes of one signal share the Tu, the offset (to `grid` of a spacing) and the pulses (their
    envelopes, in `bin_s` bins, correlate over `clock`). Carriers side by side keep their own grid (a channel raster of
    100 kHz on 15 kHz subcarriers, of 10 MHz LTE carriers 9.9 MHz apart, keeps one grid: the clock tells them) and,
    from different transmitters, their own symbol clock. Not told apart: carriers of one site, time aligned, a gap
    narrower than the gap _merge allows between them. A box between two that agree, too narrow to show a prefix, goes
    with them"""
    from ..spectrum.waveform import ofdm, SYSTEMS
    rows = {name: (tu*1e-6, cp*1e-6) for name, tu, cp in SYSTEMS}
    reads = []
    for b in run:
        y, F = _band(x, Fs, b.mid, b.width)
        try:
            o = ofdm(y, F)
        except Exception:
            o = None
        if o is None or not o.tcp or len(y) < 4*o.tu*F:
            reads.append(None)
            continue
        sys_ = rows.get(o.systems[0][0]) if o.systems else None
        tu, tcp = sys_ if sys_ is not None and abs(sys_[0]/o.tu - 1) < 0.005 else (o.tu, o.tcp)
        f = np.fft.fftfreq(len(y), 1/F)
        later = np.fft.ifft(np.fft.fft(y)*np.exp(2j*np.pi*f*tu)) # y(t + Tu)
        c = (y*np.conj(later))[:len(y) - int(np.ceil(tu*F))]
        off = (-np.angle(np.sum(c))/(2*np.pi*tu) + b.mid) % (1/tu)
        C = max(1, int(round(tcp*F)))
        env = np.abs(np.convolve(c, np.ones(C)/C, mode="same")) # (in a prefix it adds up, elsewhere it doesn't)
        k = (np.arange(len(env))/F/bin_s).astype(np.int64)
        e = np.bincount(k, env)/np.maximum(np.bincount(k), 1)
        reads.append((tu, off, sys_ is not None, e))
    def same(a, b):
        if a is None or b is None or abs(a[0]/b[0] - 1) > 0.01:
            return False
        if a[2] and b[2]: # (the grid, with a system's exact Tu; an estimated one turns the phase across the box)
            df = 1/a[0]
            if abs((a[1] - b[1] + df/2) % df - df/2)*a[0] >= grid:
                return False
        n = min(len(a[3]), len(b[3])) - 2
        if n < 64:
            return False
        u, v = a[3][1:n] - a[3][1:n].mean(), b[3][1:n] - b[3][1:n].mean()
        r = float(np.dot(u, v)/max(np.sqrt(np.dot(u, u)*np.dot(v, v)), 1e-300))
        return r > clock
    out, cur, last = [], [], None
    def close():
        while cur and reads[run.index(cur[-1])] is None:
            cur.pop()
        if len(cur) > 1:
            out.append(list(cur))
    for b, rd in zip(run, reads):
        if rd is None:
            if cur:
                cur.append(b) # (held: kept if a box that agrees follows)
            continue
        if cur and last is not None and same(last, rd):
            cur.append(b)
        else:
            close()
            cur[:] = [b]
        last = rd
    close()
    return out or None

def _refit(source, sigs, samples=2**21, nfft=2**18, drop=10.0, q=95, move=0.08, widths=(0.8, 1.3)):
    """The survey's boxes fitted to their signals in a fine spectrum of the capture: `samples` read once (source.peek),
    the `q`th percentile over time of an `nfft`-bin spectrum (bursts and TDD carriers at their full level), each box's
    edges where it falls `drop` dB under the box's top (at most halfway down to the floor around it, a weak carrier's),
    walking out from its strongest point; a box under 6 dB over that floor keeps the survey's. The survey's own spectrum is
    short and coarse: it places some boxes off their signal (wider, narrower, beside it). A fit that runs out of the
    span looked at (1.5 box widths each side: the signal is part of something wider) or changes the width more than 3x
    keeps the survey's box; so does one that moves the centre under `move` of the width with the width within `widths`
    times (the two measure a carrier differently: occupied power, edges). Returns the boxes, refitted ones as copies
    (`refit` set). On a hand-reviewed pool: 17 of 24 boxes reviewers placed off their one signal moved the way they said
    (2 the other way), 3 of 325 they said were right moved far, and box width over symbol rate went from 0.98-1.35
    (10-90%) to 1.15-1.19: a carrier's -10 dB width, whatever share of its power the survey's box held"""
    import copy
    if not sigs or not hasattr(source, "peek"):
        return list(sigs)
    try:
        x = np.asarray(source.peek(samples))
    except Exception:
        return list(sigs)
    n = 1 << int(np.log2(max(16, min(nfft, len(x)//4))))
    if len(x) < 4*n or n < 4096:
        return list(sigs)
    Fs = float(source.Fs)
    seg = x[:len(x)//n*n].reshape(-1, n)[:16]
    w = np.hanning(n).astype(np.float32)
    P = np.empty((len(seg), n), np.float32)
    for j, v in enumerate(seg):
        P[j] = np.fft.fftshift(np.abs(np.fft.fft(v*w))**2)
    pk = 10*np.log10(np.percentile(P, q, axis=0) + 1e-30)
    del P
    f = np.fft.fftshift(np.fft.fftfreq(n, 1/Fs))
    out = []
    for s in sigs:
        if getattr(s, "holds", 0) or not getattr(s, "width", None) or getattr(s, "merged", None):
            out.append(s)
            continue
        span = np.flatnonzero(np.abs(f - s.mid) <= 1.5*s.width)
        inb = np.flatnonzero(np.abs(f - s.mid) <= s.width/2)
        if len(inb) < 8:
            out.append(s)
            continue
        k = max(1, len(span)//200)
        p = np.convolve(pk[span], np.ones(k)/k, mode="same")
        fs = f[span]
        box = (np.abs(fs - s.mid) <= s.width/2)
        top_i = int(np.flatnonzero(box)[np.argmax(p[box])])
        # Down `drop` dB, but not into the floor (the span's lowest): a weak carrier's edges are halfway down to it
        hi90, floor = np.percentile(p[box], 90), np.percentile(p, 10)
        if hi90 - floor < 6.0: # (too weak over the floor to place its edges)
            out.append(s)
            continue
        thr = max(hi90 - drop, floor + 0.5*(hi90 - floor))
        a = top_i
        while a > 0 and p[a - 1] > thr:
            a -= 1
        b = top_i
        while b < len(p) - 1 and p[b + 1] > thr:
            b += 1
        mid, width = 0.5*(fs[a] + fs[b]), fs[b] - fs[a]
        rest = box.copy()
        rest[a:b + 1] = False
        # More in the box than this signal (carriers side by side: the notch split's to cut, not this to drop)
        others = np.count_nonzero(p[rest] > thr) > 0.1*np.count_nonzero(box)
        wide = width > 2*s.width and abs(mid - s.mid) > 0.2*s.width # (widened far and moved: into a neighbour)
        if others or wide or a == 0 or b == len(p) - 1 or not (s.width/3 <= width <= 3*s.width) or \
           (abs(mid - s.mid) < move*s.width and widths[0] <= width/s.width <= widths[1]):
            out.append(s)
            continue
        r = copy.copy(s)
        r.mid, r.width = float(mid), float(width)
        r.lo, r.hi = r.mid - r.width/2, r.mid + r.width/2
        r.refit = (s.mid, s.width)
        out.append(r)
    return out

def _parallel(source, picked, sv, jobs, tell, kw):
    """profile() of `picked` in `jobs` forked processes: the signals dealt round robin, results in their order"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    global _JOB
    jobs = min(jobs, len(picked))
    groups = [picked[k::jobs] for k in range(jobs)]
    _JOB = (source, getattr(source, "cur_samp", None), groups, sv, kw)
    out = {}
    try:
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
            futs = {ex.submit(_parallel_part, k): k for k in range(jobs)}
            for done, f in enumerate(as_completed(futs), 1):
                for s, p in zip(groups[futs[f]], f.result()):
                    out[id(s)] = p
                tell("parallel", done, jobs, jobs - done)
    finally:
        _JOB = None
    return [out[id(s)] for s in picked]

def _parallel_part(k):
    source, start, groups, sv, kw = _JOB
    if start is not None and hasattr(source, "seek"):
        source.seek(start)
    return profile(source, signals=groups[k], survey=sv, **kw)

def _reach(x):
    """How far a profile got: solved, a profile, steps found before the first unsolved one"""
    n = next((i for i, st in enumerate(x.steps) if st.status == UNSOLVED), len(x.steps))
    return (x.solved, bool(x.profile_name), n)

def _further(q, p):
    """Whether profile q got further than p (`_reach`)"""
    return _reach(q) > _reach(p)
