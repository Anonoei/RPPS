"""A burst time plan and the bursts detected, on the time-frequency plane: whether the plan explains them"""
import numpy as np

from ..base import axes, styled

# Tx types: categorical hues, in order (none of the status hues); a burst's place: status hues, each with its own hatch
TX = ["#3987e5", "#199e70", "#d55181", "#9085e9", "#5598e7", "#3fc1c9", "#e87ba4", "#8b8b8b"]
STATUS = {"slot": ("#0ca30c", "", "in a slot of its kind"), "kind": ("#fab219", "xx", "in another kind's slot"),
          "long": ("#ec835a", "//", "overruns its slot"), "none": ("#d03b3b", "..", "in no slot")}

def bursts(ax, check, fold=False, x=None, Fs=None, nfft=256, span=None):
    """A PlanCheck (rp.analysis.spectrum.burst.check_plan) drawn: the plan's slots (by tx type) and the bursts (by their
    place: in a slot of their kind, another kind's, overrunning it, in none).

    fold=False: the time-frequency plane, capture time along [ms], frequency up [MHz, the plan's], every slot in the
    bursts' span (shaded; edged white where a burst used it); x, Fs: the capture under it as a spectrogram (nfft bins,
    dB over the floor), the slots then outlines and each burst's place a strip under its slot's row; span: (t0, t1) [s]
    to draw only that.
    fold=True: a raster of superframes (of the sequence most bursts are in), one a row, time down the page, time in the
    superframe along [ms], one block of rows per carrier; the slots are columns through every row and each burst a bar
    in its superframe's row. A right plan keeps every bar inside a column all the way down; a wrong frame period slants
    them across the rows, a wrong BTU or slot layout puts them across the columns' edges. (A hopping sequence's slots
    are drawn where its first superframe has them.)"""
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
    top = {f: i*(K + 2) for i, f in enumerate(blocks)} # (rows of a block: its superframes, a gap of 2 between blocks)
    rects, faces = [], []
    for f in blocks:
        if f is None:
            continue
        for w in wins:
            if w.freq == f:
                rects.append(Rectangle(((w.start - s.start)*1e3/k, top[f] - 0.5), w.length*1e3/k, K))
                faces.append(to_rgba(colour.get(w.tx_type, TX[-1]), 0.3))
    ax.add_collection(PatchCollection(rects, facecolors=faces, edgecolors="black", linewidths=1.0))
    bars = {st: [] for st in STATUS}
    for f, sf, a, r, st in rows:
        bars[st].append(Rectangle((a, top[f] + sf - sf0 - 0.4), r["length"]*1e3, 0.8))
    for st, rs in bars.items():
        if rs:
            col, hatch, _ = STATUS[st]
            ax.add_collection(PatchCollection(rs, facecolors=col, edgecolors=col, hatch=hatch or None, linewidths=0.3))
    ax.set_xlim(0, P*1e3/k)
    ax.set_ylim(top[blocks[-1]] + K - 0.5, -0.5) # (time runs down the page)
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
