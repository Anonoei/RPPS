"""Profiling a capture: every signal through the chain, in parallel (rpps.analysis.signal_profile)"""
import numpy as np

from ._chain import _Chain
from ._result import GUESS, SignalProfile, Step, UNSOLVED

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
    refit = _refit(source, picked) # (each process its own: the survey's boxes are what it is given)
    for old, new in zip(picked, refit):
        if new is not old:
            index[id(new)] = index.get(id(old))
    picked = refit

    cap = max(blocks, max_blocks if max_blocks is not None else 8*blocks)

    def chains_for(sigs, bauds):
        """Chains for these signals, their channels from one rp.signal.FFTBank (one FFT of the capture a hop for all
        of them), and which bank channel each reads"""
        from ...signal import FFTBank
        from ..spectrum.signal import Channel
        live = [s for s in sigs if not getattr(s, "holds", 0)]
        bank = FFTBank(source.Fs, [(s.mid, s.width) for s in live]) if live else None
        slot, out = {}, []
        for s, b in zip(sigs, bauds):
            ch = None
            if bank is not None and not getattr(s, "holds", 0):
                j = len(slot)
                ch = Channel(None, bank.Fs_out[j], bank.D[j], 0.0, 0)
            h = (hint(s) if hint is not None else None) or {}
            c = _Chain(s, source.Fs, blind and h.get("blind", True), baud=b or h.get("baud"), ch=ch, codes=h.get("codes"))
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
        subs = ch.carriers() if not ch.dead else []
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
    return out + extra

_JOB = None # What a forked worker of `_parallel` profiles (inherited, not pickled)

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
        if getattr(s, "holds", 0) or not getattr(s, "width", None):
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
