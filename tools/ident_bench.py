"""Blind identification measured end to end: how far rpps.analysis.profile gets on each shipped profile's signal

    python tools/ident_bench.py                                    # every profile, Es/N0 15 and 8 dB, 3 seeds
    python tools/ident_bench.py --only dvbs/qpsk_3_4 ccsds/tm_rs --esn0 15 10 6 --seeds 1
    python tools/ident_bench.py --variants plain inverted pad neighbour drift
    python tools/ident_bench.py --jobs 6 --update-baseline          # keep it in .benchmarks/ (--check compares)

Each profile transmits its own signal, as tools/profile_bench.py makes it (`p.tx()`; LTE and NR from rpps.grid.<x>.tx;
P25 and DMR from tests/support/lmr.py, as their tests build them), with a carrier offset, a phase and a timing offset.
The signal is then placed in a capture `--wide` times its sample rate, `--place` of that capture off its centre, with
white noise over the whole capture at `--esn0` dB a symbol (the SNR a sample at the signal's own rate for the profiles
with no symbol rate: LTE, NR). That is what a recording looks like: noise on either side of the signal, which the
survey needs to find its edges. `analysis.profile(blind=True)` then profiles the signal nearest where it was placed,
with no hint.

Variants (`--variants`, each a case set of its own):
  plain      as above
  inverted   the spectrum turned over (the samples conjugated)
  pad        4000 samples (at the signal's rate) of noise alone before and after: a signal that starts after the
             recording does
  neighbour  a second signal of the same kind (other data), 3 dB stronger, its centre 1.3 occupied widths away
  drift      a clock 20 ppm off and the carrier ramping 2e-3 of the sample rate over the capture (a Doppler rate)

Each case records the stages it got through, each 1 or 0:
  box     the survey found a signal there
  rate    the symbol rate within 0.1% (only the linear single-carrier profiles, whose rate is Fs/sps)
  code    the code found (rp.sync.AcquireFEC), or a framed profile of the right family named (a wrong name is not)
  named   the profile named is the one that sent it (`family`: one of the same family, as dvbs2/ts for dvbs2/normal)
along with the Es/N0 error of the acquisition (dB, linear profiles), the signals the survey found, the seconds it took,
and where the profile stopped (its first unsolved step, tallied over the cases). Each case runs over `--seeds` draws:
its stages are the share of the draws that passed. The cases without a tool yet stay in the bench, so the work that
adds one shows up here.
"""
import argparse
import multiprocessing as mp
import os
import struct
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): # (one BLAS thread a worker: --jobs share the box)
    os.environ.setdefault(_v, "1")

import numpy as np

import rpps as rp
from rpps.signal import channel as ch

import _bench
import profile_bench as pb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests")) # (support.lmr: the P25 and DMR builders)

PAD = 4000 # the `pad` variant's samples of noise alone either side, at the signal's own rate
STAGES = ("box", "rate", "code", "named", "family")
VARIANTS = ("plain", "inverted", "pad", "neighbour", "drift")

# ---- the signals profile_bench.py doesn't make

def sig_mis(name, rng, seconds):
    """Two input streams (ISI 0 and 1) of a DVB-S2 multistream, sent by dvbs2/normal"""
    p = rp.profile.load("dvbs2/normal")
    t = rp.profile.table("dvbs2/modcod")
    frames, air = [], 0.0
    while air < seconds or len(frames) < 4:
        m = (4, 12, 6)[len(frames) % 3]
        frames.append({"modcod": m, "pilots": 1, "sis": 0, "isi": len(frames) % 2})
        air += t.row({"modcod": m, "short": 0})["slots"]*90/1e6
    bits = rng.integers(0, 2, sum(t.row({**f, "short": 0})["kbch"] - 80 for f in frames)).astype(bool)
    o = p.tx()(rp.Data(bits, Fs=1e6, sps=1, flush=True), fields=frames)
    return rp.profile.load(name), np.asarray(o.data), o.Fs, o.meta.sps, len(bits)

def sig_tm_packets(name, rng, seconds):
    """CCSDS space packets in TM transfer frames (one virtual channel, an idle packet filling the last), RS depth 1.
    Each frame's first header pointer says where its first packet starts (2047: none starts in it)"""
    dfl = 223 - 6
    n = max(4, int(seconds*1e6/(8*(32 + 8*255))))
    pkts = [struct.pack(">HHH", 100, (3 << 14) | i, k - 1) + rng.bytes(k)
            for i, k in enumerate(rng.integers(20, 300, 2*n))]
    starts = np.cumsum([0] + [len(q) for q in pkts])[:-1]
    stream = b"".join(pkts)[:n*dfl]
    frames = []
    for k in range(n):
        chunk = stream[k*dfl:(k + 1)*dfl]
        if len(chunk) < dfl: # (an idle packet to the end, its start a packet start too)
            starts = np.append(starts, k*dfl + len(chunk))
            chunk += struct.pack(">HHH", 2047, 3 << 14, max(0, dfl - len(chunk) - 7)) + bytes(max(0, dfl - len(chunk) - 6))
        first = starts[(starts >= k*dfl) & (starts < (k + 1)*dfl)]
        fhp = int(first[0] - k*dfl) if len(first) else 2047
        frames.append(struct.pack(">HBBH", (42 << 4) | (1 << 1), k, k, (3 << 11) | fhp) + chunk[:dfl])
    bits = np.unpackbits(np.frombuffer(b"".join(frames), np.uint8)).astype(bool)
    _, x, Fs, sps, n = pb._stream("ccsds/tm_rs", bits, depth=1) # (its packet step only receives: the same on air)
    return rp.profile.load(name, depth=1), x, Fs, sps, n

def _lmr(name, rng, seconds):
    from support.lmr import dmr_bursts, fm4
    if name == "dmr/burst":
        levels, _ = dmr_bursts(rng, n=max(8, int(seconds/0.03)))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"/"unit"/"standards"))
        from test_lmr import p25_tsdus
        levels, _ = p25_tsdus(rng, counts=tuple(int(c) for c in rng.integers(1, 4, max(5, int(seconds/0.04)))))
    x = fm4(levels, 0.27 if name == "dmr/burst" else 0.25, rng, noise=0.0)
    return rp.profile.load(name), np.asarray(x, np.complex64), 48000.0, 10, None

SIGNALS = {**pb.SIGNALS, "dvbs2/mis": sig_mis, "ccsds/tm_packets": sig_tm_packets, "p25/tsdu": _lmr, "dmr/burst": _lmr}

# The profiles whose symbol rate is the sample rate over sps (linear, single carrier, one symbol rate): their rate and
# Es/N0 are scored. The others (FSK, PPM, DSSS chips, OFDM) are scored on the box, the code and the name
LINEAR = ("dvbs/qpsk", "dvbs/qpsk_3_4", "dvbs2/normal", "dvbs2/ts", "dvbs2/mis", "ccsds/tm_rs", "ccsds/tm_concat",
          "ccsds/tm_packets")

def place(x, Fs, sps, esn0, wide, at, seed, variant="plain", other=None):
    """(capture, capture Fs, offset Hz): x with a carrier offset, phase and timing offset, and the variant's impairment,
    interpolated to `wide` x Fs, moved `at` x the capture's rate off centre, with white noise over the capture at
    `esn0` dB a symbol (a sample at Fs when sps is None). other: the neighbour's samples (variant neighbour)"""
    on = np.abs(x) > 0
    P = float(np.mean(np.abs(x[on])**2)) if on.any() else 1.0
    pad = np.zeros(PAD if variant == "pad" else 0, np.complex64)
    d = rp.Data(np.concatenate((pad, x, pad)), Fs=Fs, sps=sps or 1, flush=True) + ch.CFO(1e-4*Fs) + ch.PhaseOffset(0.7)
    if sps and sps >= 2:
        d = d + ch.TimingOffset(0.37)
    if variant == "drift":
        d = d + ch.ClockDrift(20.0)
    y = np.asarray(d.data, np.complex128)
    if variant == "drift": # The carrier ramping 2e-3 Fs over the signal: phase pi rate t^2
        t = np.arange(len(y))/Fs
        y = y*np.exp(1j*np.pi*(2e-3*Fs/t[-1])*t**2)
    if variant == "inverted":
        y = np.conj(y)
    n, m = len(y), wide*len(y)
    def up(v):
        V = np.fft.fft(v, n)
        Z = np.zeros(m, np.complex128)
        h = n//2
        Z[:h], Z[m - (n - h):] = V[:h], V[h:]
        return np.fft.ifft(Z)*wide # (the same power a sample)
    Fc = wide*Fs
    off = at*Fc
    z = up(y)*np.exp(2j*np.pi*off/Fc*np.arange(m))
    if variant == "neighbour" and other is not None and sps:
        span = 1.35*Fs/sps # the occupied width
        o = np.asarray(other, np.complex128)[:n]
        o = np.concatenate((o, np.zeros(n - len(o)))) if len(o) < n else o
        Po = float(np.mean(np.abs(o[np.abs(o) > 0])**2))
        z += up(o)*np.sqrt(2*P/Po)*np.exp(2j*np.pi*(off - 1.3*span)/Fc*np.arange(m))
    var = wide*P*(sps or 1)/10**(esn0/10) # Es/N0 = P sps / sigma^2 at Fs; the same density over wide x the band
    rng = np.random.default_rng(seed)
    z += np.sqrt(var/2)*(rng.standard_normal(m) + 1j*rng.standard_normal(m))
    return z.astype(np.complex64), Fc, off

def case(name, esn0, variant, seconds, wide, at, seed):
    """One profile's signal through the survey and the profile: a dict of what it got"""
    rng = np.random.default_rng(seed)
    p, x, Fs, sps, _ = SIGNALS[name](name, rng, seconds)
    other = SIGNALS[name](name, np.random.default_rng(seed + 1000), seconds)[1] if variant == "neighbour" else None
    z, Fc, off = place(x, Fs, sps, esn0, wide, at, seed, variant, other)
    t = time.perf_counter()
    sv = list(rp.analysis.survey(rp.source.Memory(z, Fs=Fc)))
    out = {"signals": len(sv), "box": 0, "rate": None, "code": 0, "named": 0, "family": 0, "esn0_err": None,
           "named_as": None, "stop": "", "stop_kind": "box"}
    near = [s for s in sv if abs(s.mid - off) <= max(s.width/2, 0.05*Fs)]
    if near:
        out["box"], out["stop_kind"] = 1, ""
        q = rp.analysis.profile(rp.source.Memory(z, Fs=Fc), survey=sv,
                                signals=[min(near, key=lambda s: abs(s.mid - off))], blind=True)[0]
        det = q.det
        if name in LINEAR:
            true = Fs/sps
            out["rate"] = int(det is not None and det.baud is not None and abs(det.baud/true - 1) < 1e-3)
            if out["rate"] and det.esn0 is not None:
                out["esn0_err"] = float(det.esn0 - esn0)
        named = q.profile_name
        family = bool(named) and named.split("/")[0] == name.split("/")[0]
        out["code"] = int(family or (not named and q.fec is not None and bool(q.fec.found)))
        out["named"] = int(named == name)
        out["family"] = int(family)
        out["named_as"] = named
        u = q.unsolved
        if u is not None:
            out["stop"] = f"{u.kind}: {u.what}" + (f" ({u.note})" if u.note else "")
            out["stop_kind"] = u.kind
        elif not out["named"]:
            out["stop_kind"] = "misnamed" if named else "unnamed"
    out["seconds"] = time.perf_counter() - t
    return out

def _worker(args, conn):
    try:
        os.nice(10)
    except OSError:
        pass
    try:
        conn.send(case(*args))
    except Exception as e:
        conn.send({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=6)})
    conn.close()

def run_all(cases, jobs, timeout):
    """(case, result) as each finishes, each case in a process of its own (forked), `jobs` at once, stopped after
    `timeout` s"""
    ctx = mp.get_context("fork")
    todo, live = list(cases), {}
    while todo or live:
        while todo and len(live) < jobs:
            c = todo.pop(0)
            a, b = ctx.Pipe(False)
            pr = ctx.Process(target=_worker, args=(c, b), daemon=True)
            pr.start()
            b.close()
            live[c] = (pr, a, time.monotonic())
        for c, (pr, a, t0) in list(live.items()):
            if a.poll(0.2):
                try:
                    r = a.recv()
                except EOFError:
                    r = {"error": "worker died"}
            elif not pr.is_alive():
                r = {"error": f"worker exited ({pr.exitcode})"}
            elif time.monotonic() - t0 > timeout:
                pr.kill()
                r = {"error": f"timed out after {timeout:g} s"}
            else:
                continue
            pr.join(1)
            del live[c]
            yield c, r

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", help="these profiles (default: every one with a signal here)")
    ap.add_argument("--esn0", nargs="*", type=float, default=[15.0, 8.0], help="Es/N0 [dB] a symbol")
    ap.add_argument("--variants", nargs="*", default=["plain"], choices=VARIANTS, help="impairment case sets")
    ap.add_argument("--seeds", type=int, default=3, help="draws a case")
    ap.add_argument("--seconds", type=float, default=0.2, help="signal per profile, about")
    ap.add_argument("--wide", type=int, default=2, help="the capture's rate over the signal's sample rate")
    ap.add_argument("--place", type=float, default=0.125, help="the signal's centre off the capture's, x its rate")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)*9//16), help="cases at once")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds a case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true", help="where each draw stopped, in full")
    _bench.add_args(ap)
    a = ap.parse_args(argv)
    names = a.only or list(SIGNALS)
    unknown = [n for n in names if n not in SIGNALS]
    if unknown:
        ap.error(f"no signal for {unknown}: one of {sorted(SIGNALS)}")
    groups = [(n, e, v) for n in names for e in a.esn0 for v in a.variants]
    cases = [(n, e, v, a.seconds, a.wide, a.place, a.seed + 101*k + s)
             for k, n in enumerate(names) for e in a.esn0 for v in a.variants for s in range(a.seeds)]
    print(f"native={rp._accel.available()}  {len(groups)} cases x {a.seeds} seeds  capture {a.wide}x, signal "
          f"{a.place:g} off  {a.jobs} at once\n")
    rows = {}
    for c, r in run_all(cases, a.jobs, a.timeout):
        rows.setdefault(c[:3], []).append(r)
        if a.verbose:
            n, e, v = c[:3]
            tag = f"{n} {v} @ {e:g} dB seed {c[-1]}"
            print(f"  {tag:44s} " + (f"ERROR {r['error']}\n{r.get('trace', '')}" if "error" in r else
                  " ".join(f"{k}{'+' if r[k] else '-'}" for k in STAGES[:4] if r[k] is not None) +
                  f"  {r['seconds']:.1f} s" + (f"  as {r['named_as']}" if r["named_as"] and not r["named"] else "") +
                  (f"\n      stopped at {r['stop']}" if r["stop"] else "")))
    if a.verbose:
        print()
    rec = _bench.Record("ident")
    totals, stops, errs, nerr = {k: [] for k in STAGES}, Counter(), [], 0
    for g in groups:
        rs = rows.get(g, [])
        ok = [r for r in rs if "error" not in r]
        nerr += len(rs) - len(ok)
        n, e, v = g
        key = f"{n}{'' if v == 'plain' else ' ' + v} @ {e:g} dB"
        share = {k: (np.mean([r[k] for r in ok]) if ok and ok[0][k] is not None else None) for k in STAGES}
        for k in STAGES:
            if share[k] is not None:
                totals[k] += [r[k] for r in ok]
        marks = " ".join(f"{k} {share[k]:.2f}" for k in STAGES[:4] if share[k] is not None)
        secs = np.median([r["seconds"] for r in ok]) if ok else None
        es = [r["esn0_err"] for r in ok if r["esn0_err"] is not None]
        errs += es
        as_ = Counter(r["named_as"] for r in ok if r["named_as"] and not r["named"])
        stop = Counter(r["stop_kind"] for r in ok if r["stop_kind"])
        stops.update(stop)
        print(f"{key:40s} {marks:42s}" + (f" {secs:6.1f} s" if secs is not None else "") +
              (f"  Es/N0 {np.median(es):+.1f} dB" if es else "") +
              (f"  as {', '.join(f'{k} x{c}' for k, c in as_.items())}" if as_ else "") +
              (f"  stops {', '.join(f'{k} x{c}' for k, c in stop.items())}" if stop else "") +
              (f"  {len(rs) - len(ok)} errors" if len(rs) > len(ok) else ""))
        depth = sum(share[k] or 0 for k in STAGES) if ok else None
        rec.add(f"{key} stages", depth, "stages", "higher", tol=0.5)
        if secs is not None:
            rec.add(f"{key} seconds", secs, "s", "lower")
        if es:
            rec.add(f"{key} |Es/N0 error|", float(np.median(np.abs(es))), "dB", "lower", tol=0.5)
    print()
    for k in STAGES:
        if totals[k]:
            print(f"  {k:7s} {sum(totals[k]):4d} / {len(totals[k])}")
            rec.add(f"total {k}", sum(totals[k])/len(totals[k]), "share", "higher", tol=0.02)
    if errs:
        print(f"  Es/N0 error: median {np.median(errs):+.2f} dB, worst {max(errs, key=abs):+.2f} dB")
    print(f"  stops: {', '.join(f'{k} x{c}' for k, c in stops.most_common())}")
    print(f"  errors / timeouts: {nerr}")
    return _bench.finish(a, rec)

if __name__ == "__main__":
    sys.exit(main())
