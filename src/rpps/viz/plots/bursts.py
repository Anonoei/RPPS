"""A burst time plan and the bursts detected, on the time-frequency plane: whether the plan explains them"""
import numpy as np

from ..base import axes

# Tx types: categorical hues, in order (none of the status hues); a burst's place: status hues, each with its own hatch
TX = ["#3987e5", "#199e70", "#d55181", "#9085e9", "#5598e7", "#3fc1c9", "#e87ba4", "#8b8b8b"]
STATUS = {"slot": ("#0ca30c", "", "in a slot of its kind"), "kind": ("#fab219", "xx", "in another kind's slot"),
          "long": ("#ec835a", "//", "overruns its slot"), "none": ("#d03b3b", "..", "in no slot")}

def bursts(ax, check, fold=False, x=None, Fs=None, nfft=256, span=None):
    """A PlanCheck (rp.analysis.spectrum.burst.check_plan) drawn: the plan's slots (shaded by tx type, edged where a
    burst fills them) and the bursts (bars edged and hatched by their place: in a slot of their kind, another kind's,
    overrunning it, in none), frequency up the side in the plan's frequencies [MHz].

    fold=False: capture time along [ms], every slot in the bursts' span; x, Fs: the capture under it as a spectrogram
    (nfft bins), span: (t0, t1) [s] to draw only that. fold=True: the time folded on the superframe (the sequence most
    bursts are in), its slots once: a right plan stacks every burst onto a slot, a wrong BTU or frame smears them across
    it, a drifting clock leans them"""
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.collections import PatchCollection
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
    if fold:
        seqs = [w.sequence for w in c.window if w is not None] or list(plan.sequences)
        sid = max(set(seqs), key=seqs.count)
        s = plan.sequences[sid]
        P = s.duration
        phase = lambda tick: ((np.asarray(tick) - s.start) % P)*1e3/k
        wins = plan.windows(s.start, s.start + P, sequences=[sid])
        fill = {}
        for w in c.window:
            if w is not None and w.sequence == sid:
                key = (w.frame, w.slot)
                fill[key] = fill.get(key, 0) + 1
        slots = [(phase(w.start), w.length*1e3/k, w.freq, w.bw, w.tx_type, fill.get((w.frame, w.slot), 0)) for w in wins]
        xlim = (0, P*1e3/k)
        xlabel = f"time in the superframe [ms] (sequence {sid}, {P*1e3/plan.clock:g} ms)"
    else:
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
        xlabel = "capture time [ms]"
        if x is not None:
            _spectrogram(ax, np.asarray(x), Fs, nfft, c.cf, xlim)
    rects, faces, edges = [], [], []
    for a, n, f, bw, tx, used in slots:
        rects.append(Rectangle((a, (f - bw/2)/1e6), n, bw/1e6))
        faces.append(colour.get(tx, TX[-1]))
        edges.append("white" if used else "black")
    ax.add_collection(PatchCollection(rects, facecolors=faces, edgecolors=edges, alpha=0.2 if x is not None else 0.3,
                                      linewidths=0.6))
    for n, (r, st, w) in enumerate(zip(c.bursts, c.status, c.window)):
        bw = w.bw if w is not None else h_min
        f = r["freq"] + c.cf
        if fold:
            a = float(phase(c.tick(r["start"])))
            y = f + (((n*0.618034) % 1) - 0.5)*0.7*bw # (spread up the slot: each burst seen)
            h = 0.08*bw
        else:
            a, y, h = r["start"]*1e3, f, 0.6*bw
            if span is not None and not t0 <= r["start"] <= t1:
                continue
        col, hatch, _ = STATUS[st]
        ax.add_patch(Rectangle((a, (y - h/2)/1e6), r["length"]*1e3, h/1e6, facecolor="none", edgecolor=col,
                               hatch=hatch, linewidth=1.2))
    fs = [s[2] for s in slots] + [r["freq"] + c.cf for r in c.bursts]
    pad = max(bws)
    ax.set_xlim(*xlim)
    ax.set_ylim((min(fs) - pad)/1e6, (max(fs) + pad)/1e6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency [MHz]")
    cnt = c.counts
    handles = [Patch(facecolor=colour[t], alpha=0.3, label=f"tx type {t}" + (f" ({_kind(plan, t)})" if _kind(plan, t) else ""))
               for t in types]
    handles += [Patch(facecolor="none", edgecolor=STATUS[s][0], hatch=STATUS[s][1], label=f"{STATUS[s][2]}: {cnt[s]}")
                for s in STATUS if cnt[s] or s == "slot"]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small", framealpha=0.6)
    ax.set_title(f"{plan.name}: {cnt['slot']} of {len(c.status)} bursts in place, start error {1e6*c.rms:.1f} us rms, "
                 f"clock {1e6*c.rate:+.1f} ppm", fontsize="medium")
    return ax

def _kind(plan, t):
    f = plan.tx_types[t].fields
    return f.get("word") or f.get("kind")

def _spectrogram(ax, x, Fs, nfft, cf, xlim):
    """The capture under the plan: |STFT|^2 in dB, gray, time along [ms]"""
    if Fs is None:
        raise ValueError("viz.bursts: the capture's Fs with x")
    a, b = max(int(xlim[0]*1e-3*Fs), 0), min(int(np.ceil(xlim[1]*1e-3*Fs)), len(x))
    n = (b - a)//nfft
    if n < 2:
        return
    seg = x[a:a + n*nfft].reshape(n, nfft)*np.hanning(nfft)
    z = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1))**2 + 1e-20)
    f = (np.fft.fftshift(np.fft.fftfreq(nfft, 1/Fs)) + cf)/1e6
    ax.imshow(z.T, aspect="auto", origin="lower", cmap="gray", vmin=np.median(z), vmax=np.percentile(z, 99.9),
              interpolation="nearest",
              extent=(a/Fs*1e3, (a + n*nfft)/Fs*1e3, f[0], f[-1]))
