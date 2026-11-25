"""A burst time plan and the bursts detected, on the time-frequency plane: whether the plan explains them"""
import numpy as np

from ..base import axes, styled, subplots

# Tx types: categorical hues, in order (none of the status hues); a burst's place: status hues, each with its own hatch
TX = ["#3987e5", "#199e70", "#d55181", "#9085e9", "#5598e7", "#3fc1c9", "#e87ba4", "#8b8b8b"]
STATUS = {"slot": ("#0ca30c", "", "in a slot of its kind"), "kind": ("#fab219", "xx", "in another kind's slot"),
          "long": ("#ec835a", "//", "overruns its slot"), "none": ("#d03b3b", "..", "in no slot")}

def bursts(ax, check, fold=False, x=None, Fs=None, nfft=256, span=None):
    """A PlanCheck (rp.analysis.burst.check_plan) drawn: the plan's slots (by tx type) and the bursts (by their
    place: in a slot of their kind, another kind's, overrunning it, in none).

    fold=False: the time-frequency plane, capture time along [ms], frequency up [MHz, the plan's], every slot in the
    bursts' span (shaded; edged white where a burst used it); x, Fs: the capture under it as a spectrogram (nfft bins,
    dB over the floor), the slots then outlines and each burst's place a strip under its slot's row; span: (t0, t1) [s]
    to draw only that.
    fold=True: a raster of superframes (of the sequence most bursts are in), one a row, time down the page, time in the
    superframe along [ms], one block of rows per carrier; the slots are columns through every row and each burst a bar
    in its superframe's row. A right plan keeps every bar inside a column all the way down; a wrong frame period slants
    them across the rows, a wrong BTU or slot layout puts them across the columns' edges. Over each column, the share
    of the superframes a burst used it in. (A hopping sequence's slots are drawn where its first superframe has them.)"""
    with styled(): # (the legend and colour bar made now: in rpps' style, as the axes were)
        if fold:
            return _raster(ax, check)
        return _draw(ax, check, x, Fs, nfft, span)

def _draw(ax, check, x, Fs, nfft, span):
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import to_rgba
    ax = axes(ax, layout="constrained")
    c, plan = check, check.plan
    if not np.isfinite(c.origin):
        ax.set_title("no burst is on a carrier of the plan")
        return ax
    k = plan.clock*(1 + c.rate)
    types = sorted(plan.tx_types)
    colour = {t: TX[i % len(TX)] for i, t in enumerate(types)}
    bws = [w.bw for w in c.windows] or [ft.btu_bw for ft in plan.frame_types.values()]
    h_min = min(bws)
    t0, t1 = span if span is not None else (None, None)
    slots = []
    for w, n in zip(c.windows, c.filled):
        a = float(c.time(w.start))
        if span is None or t0 <= a <= t1:
            slots.append((a*1e3, w.length*1e3/k, w.freq, w.bw, w.tx_type, int(n)))
    starts = [r["start"] for r in c.bursts]
    lo = min(starts) if span is None else t0
    hi = max(r["start"] + r["length"] for r in c.bursts) if span is None else t1
    xlim = (lo*1e3, hi*1e3)
    over = x is not None
    fs = [s_[2] for s_ in slots] + [r["freq"] + c.cf for r in c.bursts]
    pad = max(bws)
    ylim = ((min(fs) - pad)/1e6, (max(fs) + pad)/1e6)
    im = _spectrogram(ax, np.asarray(x), Fs, nfft, c.cf, xlim, ylim) if over else None
    rects, faces, edges, widths, styles = [], [], [], [], []
    for a, n, f, bw, tx, used in slots:
        rects.append(Rectangle((a, (f - bw/2)/1e6), n, bw/1e6))
        col = colour.get(tx, TX[-1])
        if over: # (outlines only: the spectrogram shows through; a slot a burst used drawn heavier)
            faces.append("none")
            edges.append(to_rgba(col, 1.0 if used else 0.8))
            widths.append(1.6 if used else 0.8)
            styles.append("solid" if used else (0, (3, 2)))
        else: # (shaded, each slot edged apart from the next; white where a burst used it)
            faces.append(to_rgba(col, 0.4))
            edges.append("white" if used else "black")
            widths.append(0.8)
            styles.append("solid")
    ax.add_collection(PatchCollection(rects, facecolors=faces, edgecolors=edges, linewidths=widths, linestyles=styles))
    for n, (r, st, w) in enumerate(zip(c.bursts, c.status, c.window)):
        bw = w.bw if w is not None else h_min
        f = r["freq"] + c.cf
        col, hatch, _ = STATUS[st]
        if span is not None and not t0 <= r["start"] <= t1:
            continue
        a = r["start"]*1e3
        if over: # (its verdict as a strip under its slot's row: the burst's own energy is the spectrogram's)
            fc = w.freq if w is not None else f
            ax.add_patch(Rectangle((a, (fc - bw/2 - 0.12*bw)/1e6), r["length"]*1e3, 0.08*bw/1e6, facecolor=col,
                                   edgecolor=col, hatch=hatch, linewidth=0))
            continue
        y, h = f, 0.6*bw
        ax.add_patch(Rectangle((a, (y - h/2)/1e6), r["length"]*1e3, h/1e6, facecolor="none", edgecolor=col,
                               hatch=hatch, linewidth=1.2))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if im is not None:
        ax.figure.colorbar(im, ax=ax, pad=0.01, fraction=0.03, label="dB over the floor")
    ax.set_xlabel("capture time [ms]")
    ax.set_ylabel("frequency [MHz]")
    _legend(ax, c, colour, types, filled=not over, solid=over)
    return ax

def _legend(ax, c, colour, types, filled, solid):
    """Tx types (shaded or outlined) and the bursts' places (solid or outlined), and the counts in the title"""
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    cnt, plan = c.counts, c.plan
    handles = [Patch(facecolor=to_rgba(colour[t], 0.4) if filled else "none", edgecolor="black" if filled else colour[t],
                     label=f"tx type {t}" + (f" ({_kind(plan, t)})" if _kind(plan, t) else "")) for t in types]
    handles += [Patch(facecolor=STATUS[s][0] if solid else "none", edgecolor=STATUS[s][0], hatch=STATUS[s][1],
                      label=f"{STATUS[s][2]}: {cnt[s]}") for s in STATUS if cnt[s] or s == "slot"]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=min(len(handles), 4),
              fontsize="small", frameon=False)
    ax.set_title(f"{plan.name}: {cnt['slot']} of {len(c.status)} bursts in place, start error {1e6*c.rms:.1f} us rms, "
                 f"clock {1e6*c.rate:+.1f} ppm", fontsize="medium")

def _raster(ax, c):
    """fold=True: one row per superframe, one block of rows per carrier (see `bursts`)"""
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import to_rgba
    ax = axes(ax, layout="constrained")
    plan = c.plan
    if not np.isfinite(c.origin):
        ax.set_title("no burst is on a carrier of the plan")
        return ax
    types = sorted(plan.tx_types)
    colour = {t: TX[i % len(TX)] for i, t in enumerate(types)}
    seqs = [w.sequence for w in c.window if w is not None] or list(plan.sequences)
    sid = max(set(seqs), key=seqs.count)
    s = plan.sequences[sid]
    P, k = s.duration, plan.clock*(1 + c.rate)
    wins = plan.windows(s.start, s.start + P, sequences=[sid])
    carriers = sorted({w.freq for w in wins})
    bw = {f: max(w.bw for w in wins if w.freq == f) for f in carriers}
    rows = [] # (carrier or None: off the plan's, superframe, time in it [ms], burst)
    for r, st in zip(c.bursts, c.status):
        tick = float(c.tick(r["start"])) - s.start
        sf = int(np.floor(tick/P))
        f = r["freq"] + c.cf
        near = [g for g in carriers if abs(g - f) <= 0.5*bw[g]]
        rows.append((near[0] if near else None, sf, (tick - sf*P)*1e3/k, r, st))
    sf0, sf1 = min(q[1] for q in rows), max(q[1] for q in rows)
    K = sf1 - sf0 + 1
    blocks = [f for f in carriers if any(q[0] == f for q in rows)] + ([None] if any(q[0] is None for q in rows) else [])
    gap = max(2.0, 0.08*K) # (rows between blocks: room for the slots' use over each)
    top = {f: i*(K + gap) for i, f in enumerate(blocks)}
    rects, faces = [], []
    for f in blocks:
        if f is None:
            continue
        for w in wins:
            if w.freq == f:
                rects.append(Rectangle(((w.start - s.start)*1e3/k, top[f] - 0.5), w.length*1e3/k, K))
                faces.append(to_rgba(colour.get(w.tx_type, TX[-1]), 0.3))
    ax.add_collection(PatchCollection(rects, facecolors=faces, edgecolors="black", linewidths=1.0))
    used = {} # (carrier, frame, slot): the superframes a burst was in it
    for (f, sf, *_), w in zip(rows, c.window):
        if w is not None and f is not None:
            used.setdefault((f, w.frame, w.slot), set()).add(sf)
    for f in blocks: # each slot's share of the superframes used, over its column
        for w in (w for w in wins if w.freq == f and w.length*1e3/k >= 0.025*P*1e3/k):
            n = len(used.get((f, w.frame, w.slot), ()))
            ax.text((w.start - s.start + w.length/2)*1e3/k, top[f] - 0.5 - 0.1*gap, f"{100*n/K:.0f}%", ha="center", va="bottom",
                    fontsize=7, color="white" if n else "0.5")
    bars = {st: [] for st in STATUS}
    for f, sf, a, r, st in rows:
        bars[st].append(Rectangle((a, top[f] + sf - sf0 - 0.4), r["length"]*1e3, 0.8))
    for st, rs in bars.items():
        if rs:
            col, hatch, _ = STATUS[st]
            ax.add_collection(PatchCollection(rs, facecolors=col, edgecolors=col, hatch=hatch or None, linewidths=0.3))
    ax.set_xlim(0, P*1e3/k)
    ax.set_ylim(top[blocks[-1]] + K - 0.5, -0.5 - gap) # (time runs down the page)
    ax.set_yticks([top[f] + (K - 1)/2 for f in blocks],
                  [f"{f/1e6:.3f} MHz" if f is not None else "off the plan's carriers" for f in blocks])
    ax.set_xlabel(f"time in the superframe [ms] (sequence {sid}: {P*1e3/plan.clock:g} ms)")
    ax.set_ylabel(f"superframes {sf0}..{sf1}, one a row, time down")
    _legend(ax, c, colour, types, filled=True, solid=True)
    return ax

def _kind(plan, t):
    f = plan.tx_types[t].fields
    return f.get("word") or f.get("kind")

def _spectrogram(ax, x, Fs, nfft, cf, xlim, ylim, cols=1600):
    """The capture under the plan: |STFT|^2 (Hann, half overlapped) in dB over the floor (the median within ylim [MHz]:
    the plan's carriers, not a filtered band's empty edges), at most `cols`
    time columns (the frames' power averaged), time along [ms]; returns the image"""
    if Fs is None:
        raise ValueError("viz.bursts: the capture's Fs with x")
    a, b = max(int(xlim[0]*1e-3*Fs), 0), min(int(np.ceil(xlim[1]*1e-3*Fs)), len(x))
    hop = nfft//2
    n = (b - a - nfft)//hop + 1
    if n < 2:
        return None
    win = np.hanning(nfft)
    p = np.empty((n, nfft))
    for i in range(0, n, 4096): # (in pieces: a long capture)
        j = np.arange(i, min(i + 4096, n))
        seg = x[a + j[:, None]*hop + np.arange(nfft)]*win
        p[j] = np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1))**2
    m = max(1, n//cols)
    p = p[:(n//m)*m].reshape(-1, m, nfft).mean(axis=1)
    z = 10*np.log10(p + 1e-30)
    f = (np.fft.fftshift(np.fft.fftfreq(nfft, 1/Fs)) + cf)/1e6
    inside = (f >= ylim[0]) & (f <= ylim[1])
    z -= np.median(z[:, inside] if inside.any() else z)
    df = (f[1] - f[0])/2
    t_end = (a + (len(z)*m - 1)*hop + nfft)/Fs*1e3
    return ax.imshow(z.T, aspect="auto", origin="lower", cmap="magma", vmin=-3, vmax=max(np.percentile(z, 99.9), 6),
                     interpolation="nearest", extent=(a/Fs*1e3, t_end, f[0] - df, f[-1] + df))

# ---- the bursts' timing, layout, transmitters, and the plan's evidence

def _groups(keys, top=len(TX) - 1):
    """Categorical colours for keys in order of first appearance; past `top` of them, one gray "other" """
    seen = list(dict.fromkeys(keys))
    col = {k: TX[i] for i, k in enumerate(seen[:top])}
    return lambda k: col.get(k, TX[-1]), seen[:top], len(seen) > top

def burst_timing(ax, check, by="carrier"):
    """Each burst's start error (PlanCheck.error: its start minus its slot's, at the fitted origin and clock) against
    capture time, coloured by its carrier, slot or tx type (`by`), the tolerance shaded. A line leaning is a clock the fit
    did not follow, a step the plan's time jumping, a slot or carrier off the rest a transmitter late (its ranging),
    a wide scatter a start detector smeared by noise. Bursts in no slot have no error: counted in the title"""
    with styled():
        ax = axes(ax, layout="constrained")
        c = check
        if by not in ("carrier", "slot", "kind"):
            raise ValueError(f"burst_timing: by carrier, slot or kind, not {by!r}")
        key = {"carrier": lambda w: w.freq, "slot": lambda w: (w.frame, w.slot), "kind": lambda w: w.tx_type}[by]
        pts = [(r["start"], 1e6*e, key(w)) for r, e, w in zip(c.bursts, c.error, c.window) if w is not None]
        ax.axhspan(-1e6*c.tol, 1e6*c.tol, color="0.5", alpha=0.15, lw=0)
        ax.axhline(0, color="0.6", lw=0.8)
        if pts:
            colour, shown, other = _groups([p[2] for p in pts])
            for g in shown + (["other"] if other else []):
                sel = [p for p in pts if (p[2] == g) or (g == "other" and p[2] not in shown)]
                e = np.asarray([p[1] for p in sel])
                name = (f"{g/1e6:.3f} MHz" if by == "carrier" else f"frame {g[0]} slot {g[1]}" if by == "slot" else
                        f"tx type {g}") if g != "other" else "the rest"
                ax.plot([p[0] for p in sel], e, "o", ms=3.5, color=colour(g) if g != "other" else TX[-1],
                        label=f"{name}: {len(sel)}, {np.sqrt(np.mean(e**2)):.1f} us rms")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize="small", frameon=False)
            ax.set_ylim(*(1.4*np.max(np.abs([p[1] for p in pts]))*np.array([-1, 1]))) # (the errors' own scale: the
            # tolerance shaded where it reaches)
        ax.set_xlabel("capture time [s]")
        ax.set_ylabel("start error [us]")
        ax.set_title(f"{c.plan.name}: start error of {len(pts)} bursts in a slot ({c.counts['none']} in none), "
                     f"{1e6*c.rms:.1f} us rms; tolerance shaded", fontsize="medium")
        return ax

def burst_stack(ax, result, kind=0, what="phase"):
    """The bursts of one kind (rp.analysis.burst.demodulate's: its `kind` index) lined up on their known
    symbols, one a row in time order, symbol index along (0: the known run's first); `what` "phase" (degrees, a cyclic
    map) or "magnitude" (over the median). Above it, how far the bursts agree at each symbol, |mean of the unit
    symbols| (1: the same symbol in every burst, about 1/sqrt(n) for data). A burst's anatomy shows: its preamble or
    unique word, words repeating inside it, pilots, a postamble, its ramps and where the data runs"""
    with styled():
        ax = axes(ax, layout="constrained")
        rows = [b for b in result.bursts if b.ok and b.kind == kind and b.symbols is not None]
        if not rows:
            ax.set_title(f"no burst of kind {kind} demodulated")
            return ax
        rows.sort(key=lambda b: b.start)
        first = [b.known[0] if b.known else 0 for b in rows]
        before, after = max(first), max(len(b.symbols) - f for b, f in zip(rows, first))
        Y = np.full((len(rows), before + after), np.nan + 0j)
        for i, (b, f) in enumerate(zip(rows, first)):
            Y[i, before - f:before - f + len(b.symbols)] = b.symbols
        idx = np.arange(-before, after)
        ext = (idx[0] - 0.5, idx[-1] + 0.5, len(rows) - 0.5, -0.5)
        if what == "phase":
            im = ax.imshow(np.degrees(np.angle(Y)), aspect="auto", cmap="twilight", vmin=-180, vmax=180,
                           interpolation="nearest", extent=ext)
            label = "phase [deg]"
        elif what == "magnitude":
            m = np.abs(Y)
            im = ax.imshow(m/np.nanmedian(m), aspect="auto", cmap="magma", vmin=0, vmax=2, interpolation="nearest",
                           extent=ext)
            label = "magnitude (over the median)"
        else:
            raise ValueError(f"burst_stack: what is phase or magnitude, not {what!r}")
        ax.figure.colorbar(im, ax=ax, pad=0.01, fraction=0.03, label=label)
        n = rows[0].known[1] if rows[0].known else 0
        for v in (-0.5, n - 0.5):
            ax.axvline(v, color="white", lw=0.8, ls=":")
        ax.set_xlabel("symbol (0: the known run's first)")
        ax.set_ylabel(f"burst ({len(rows)}, in time order)")
        top = ax.inset_axes([0, 1.02, 1, 0.22], sharex=ax)
        u = Y/np.where(np.abs(Y) > 0, np.abs(Y), 1)
        cnt = np.sum(np.isfinite(u), axis=0)
        agree = np.abs(np.nansum(np.where(np.isfinite(u), u, 0), axis=0))/np.maximum(cnt, 1)
        agree[cnt < 2] = np.nan
        top.plot(idx, agree, color=TX[0], lw=1.0)
        top.axhline(1/np.sqrt(len(rows)), color="0.6", lw=0.8, ls="--")
        top.axvspan(-0.5, n - 0.5, color="0.5", alpha=0.2, lw=0)
        top.set_ylim(0, 1.05)
        top.set_ylabel("agree", fontsize="small")
        top.tick_params(labelbottom=False, labelsize="small")
        top.set_title(f"kind {kind}: {len(rows)} bursts, {n} known symbols (shaded, dotted below); dashed: chance",
                      fontsize="medium")
        return ax

TERMINAL = {"cfo": (lambda b: b.cfo/1e3, "carrier offset [kHz]"), "gain": (lambda b: 20*np.log10(max(b.gain, 1e-12)), "gain [dB]"),
            "clock": (lambda b: 1e6*(b.clock or 0.0), "clock offset [ppm]"), "evm": (lambda b: 100*b.evm, "EVM [%]")}

def burst_terminals(ax, result, x="cfo", y="gain", by="slot"):
    """Each demodulated burst (rp.analysis.burst.demodulate's) as a point of two of its measures (`x`, `y`:
    cfo, gain, clock, evm), coloured by its slot, its kind or nothing (`by`: "slot", "kind", None). A transmitter keeps
    its oscillator's offset and its link's power from burst to burst, so each one is a cluster: how many there are, and
    which slots each has, without decoding them"""
    with styled():
        ax = axes(ax, layout="constrained")
        for k in (x, y):
            if k not in TERMINAL:
                raise ValueError(f"burst_terminals: a measure is one of {sorted(TERMINAL)}, not {k!r}")
        ok = [b for b in result.bursts if b.ok]
        fx, lx = TERMINAL[x]
        fy, ly = TERMINAL[y]
        if by is None:
            ax.plot([fx(b) for b in ok], [fy(b) for b in ok], "o", ms=4, color=TX[0], alpha=0.8)
        else:
            if by not in ("slot", "kind"):
                raise ValueError(f"burst_terminals: by slot, kind or None, not {by!r}")
            key = (lambda b: b.slot) if by == "slot" else (lambda b: b.kind)
            colour, shown, other = _groups([key(b) for b in ok])
            for g in shown + (["other"] if other else []):
                sel = [b for b in ok if key(b) == g or (g == "other" and key(b) not in shown)]
                ax.plot([fx(b) for b in sel], [fy(b) for b in sel], "o", ms=4, alpha=0.8,
                        color=colour(g) if g != "other" else TX[-1],
                        label=f"{by} {g}: {len(sel)}" if g != "other" else f"the rest: {len(sel)}")
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small", frameon=False)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.set_title(f"{len(ok)} bursts demodulated: one cluster per transmitter", fontsize="medium")
        return ax

def plan_evidence(axs, guess):
    """What learn_plan chose its plan on (its Guess's `scores`): the BTU score (the mean cosine of 2 pi d / q over the
    bursts' start differences) against q, the threshold and the q taken; and each frame period tried (BTUs from 2 up) by
    how pure its cells are (the share of bursts whose carrier and BTU in the frame hold their kind), why each was passed
    over (impure, no slot used twice, shorter than a burst, slots colliding) and the one taken. guess: learn_plan's
    Guess, or the NoPlan error it raised when no period passed (its scores too). axs: two axes, or None"""
    with styled():
        if axs is None:
            axs = subplots(1, 2, figsize=(14, 4.5), layout="constrained")[1]
        a, b = axs
        sc = guess.scores
        if sc is None:
            raise ValueError("plan_evidence: a learn_plan Guess or NoPlan error (their scores)")
        if sc["quantum"] is None:
            a.set_title(f"BTU given: {1e3*sc['btu']:.4g} ms", fontsize="medium")
        else:
            qs, score = sc["quantum"]
            a.plot(1e3*qs, score, color=TX[0], lw=1.0)
            a.axhline(np.cos(2*np.pi*sc["tol"]), color="0.6", ls="--", lw=0.8, label=f"threshold (tol {sc['tol']:g})")
            j = int(np.argmin(np.abs(qs - sc["btu"])))
            a.axvline(1e3*sc["btu"], color=STATUS["slot"][0], lw=0.8, ls=":")
            a.plot(1e3*qs[j], score[j], "o", ms=7, mfc="none", mec=STATUS["slot"][0], mew=1.5,
                   label=f"BTU {1e3*sc['btu']:.5g} ms")
            a.set_xscale("log")
            a.set_ylim(-0.2, 1.05)
            a.legend(fontsize="small", frameon=False, loc="lower left")
            a.set_title("BTU: how well the starts sit on multiples of q (largest q passing)", fontsize="medium")
        a.set_xlabel("q [ms]")
        a.set_ylabel("mean cos(2 pi d / q)")
        why = {"": (STATUS["slot"][0], "taken"), "purity": ("0.6", "impure"), "repeats": (TX[0], "no slot used twice"),
               "short": (TX[3], "shorter than a burst"), "overlap": (STATUS["long"][0], "slots collide")}
        fr = sc["frames"]
        for w, (col, name) in why.items():
            sel = [(T, p) for T, p, ww in fr if ww == w]
            if sel:
                T, p = np.asarray(sel).T
                b.plot(1e3*T, p, "o", ms=3 if w else 6, color=col, label=f"{name}: {len(sel)}")
        b.axhline(sc["purity"], color="0.6", ls="--", lw=0.8)
        b.set_ylim(0, 1.05)
        b.set_xlabel("frame period [ms]")
        b.set_ylabel("purity")
        b.legend(fontsize="small", frameon=False, loc="lower right")
        b.set_title(f"frame: the shortest period passing, {1e3*sc['frame']:.4g} ms ({len(fr)} tried)" if sc["frame"] else
                    f"frame: no period of the {len(fr)} tried passes", fontsize="medium")
        return axs
