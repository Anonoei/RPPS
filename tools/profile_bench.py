"""Where a profile's receive time goes: each shipped profile on a synthetic signal, its steps' own time over air time

    python tools/profile_bench.py                     # every profile below, about 0.2 s of signal each
    python tools/profile_bench.py --only dvbs2/normal lte/downlink --seconds 0.5
    python tools/profile_bench.py --update-baseline   # keep this machine's numbers in .benchmarks/ (--check compares)

Each profile transmits its own signal (`p.tx()`; LTE and NR from rpps.grid.<x>.tx), with the inputs its tests use (TS
packets for DVB-S, AX.25 frames at 1200 baud, Mode S squitters...). The signal goes through a carrier offset, a timing
offset and AWGN (`--esn0` dB a symbol), and is received in `--block` sample blocks from rp.source.Memory, once to warm
up (tables, kernels, caches) and once timed. A step's load is its own busy time (rpps.base.meta.timed: what ran inside
it is charged to that) over the signal's air time; the real-time factor is the receive's wall time over air time.

The summary groups steps by kind (`sync:timing`, `code:block`...) over the profiles: each kind's share of a profile's
receive time, its median and mean across the profiles it is in, and the single heaviest step anywhere. The profiles
that only receive through branches with no transmit side here (P25, DMR, DVB-S2 MIS, the packet DVB-S2 ones) are left out.
"""
import argparse
import statistics
import sys
import time

import numpy as np

import rpps as rp
from rpps.signal import channel as ch

import _bench

# ---- the signals: (profile, samples, Fs, sps, data bits sent) for each profile

def _frames(p, frames, Fs, fields=None):
    """Transmit `frames` one call each on one stream (frame-at-a-time profiles: Mode S, AX.25)"""
    tx, s, waves = p.tx(), None, []
    for i, f in enumerate(frames):
        d = rp.Data(f, Fs=Fs, sps=1, stream=s, flush=i == len(frames) - 1)
        s = d.stream
        o = tx(d, fields=fields) if fields else tx(d)
        waves += [np.asarray(o.data), np.zeros(len(o.data)//4, np.complex64)]
    return np.concatenate(waves), o

def _ts(rng, n):
    pk = rng.integers(0, 256, (n, 188)).astype(np.uint8)
    pk[:, 0] = 0x47
    return np.unpackbits(pk.reshape(-1)).astype(bool)

def _stream(name, bits, Fs=1e6, **params):
    p = rp.profile.load(name, **params)
    o = p.tx()(rp.Data(bits, Fs=Fs, sps=1, flush=True))
    return p, np.asarray(o.data), o.Fs, o.meta.sps, len(bits)

def sig_dvbs(name, rng, seconds):
    n = max(300, int(seconds*1e6*2/1504*(0.5 if name.endswith("1_2") or name == "dvbs/qpsk" else 0.75)))
    return _stream(name, _ts(rng, n))

def sig_dvbs2(name, rng, seconds):
    p = rp.profile.load(name)
    t = rp.profile.table("dvbs2/modcod")
    modcods = (4, 13, 19, 6, 14) # QPSK 1/2, 8PSK 2/3, 16APSK 3/4, QPSK 2/3, 8PSK 3/4: a VCM mix
    frames, air = [], 0.0
    while air < seconds or len(frames) < 4:
        m = modcods[len(frames) % len(modcods)]
        frames.append({"modcod": m, "pilots": 1})
        air += t.row({"modcod": m, "short": 0})["slots"]*90/1e6
    if name == "dvbs2/ts":
        bits = _ts(rng, sum(t.row({**f, "short": 0})["kbch"] - 80 for f in frames)//1504)
    else:
        bits = rng.integers(0, 2, sum(t.row({**f, "short": 0})["kbch"] - 80 for f in frames)).astype(bool)
    o = p.tx()(rp.Data(bits, Fs=1e6, sps=1, flush=True), fields=frames)
    return p, np.asarray(o.data), o.Fs, o.meta.sps, len(bits)

def sig_bits(rate):
    def make(name, rng, seconds):
        return _stream(name, rng.integers(0, 2, max(8*400, int(seconds*rate)//8*8)).astype(bool))
    return make

def sig_ccsds(name, rng, seconds):
    per = 8*223*(2 if name == "ccsds/tm_rs" else 1)
    return _stream(name, rng.integers(0, 2, max(3, int(seconds*1e6/2/per))*per).astype(bool), depth=2) \
        if name == "ccsds/tm_rs" else _stream(name, rng.integers(0, 2, max(3, int(seconds*1e6/4/per))*per).astype(bool))

def sig_adsb(name, rng, seconds):
    p = rp.profile.load(name)
    msgs = [rng.integers(0, 2, 83).astype(bool) for _ in range(max(10, int(seconds/150e-6)))]
    x, o = _frames(p, msgs, 2e6, {"df": 17})
    return p, x, o.Fs, o.meta.sps, 83*len(msgs)

def sig_ax25(name, rng, seconds):
    p = rp.profile.load(name, centre=0, sps=10)
    pk = [bytes(rng.integers(0, 256, 40).astype(np.uint8)) for _ in range(max(10, int(seconds*1200/(8*50))))]
    x, o = _frames(p, pk, 1200.0)
    return p, x, o.Fs, o.meta.sps, 8*40*len(pk)

def sig_ble(name, rng, seconds):
    p = rp.profile.load(name)
    pdus = [rng.integers(0, 2, 8*30).astype(bool) for _ in range(max(5, int(seconds/1.3e-3)))]
    o = p.tx()([rp.Data(d, Fs=1e6, sps=1) for d in pdus], fields={"type": 2, "txadd": 1}, flush=True)
    return p, np.asarray(o.data), o.Fs, o.meta.sps, 8*30*len(pdus)

def sig_lte(name, rng, seconds):
    from rpps.grid import lte
    sib = np.random.default_rng(1).integers(0, 2, 224).astype(np.uint8)
    y = lte.tx.channel(lte.tx.downlink(pci=77, ports=2, n_rb=25, frames=max(4, int(seconds/0.01)), sib=sib), fs=7.68e6)
    return rp.profile.load(name, n_rb=25), np.asarray(y, np.complex64), 7.68e6, None, None

def sig_nr(name, rng, seconds):
    from rpps.grid import nr
    size = nr.tx.sib_size(24, 1, 2, 2)
    sib = np.random.default_rng(2).integers(0, 2, size).astype(np.uint8)
    y = nr.tx.downlink(pci=17, nfft=512, frames=max(4, int(seconds/0.01)), sib=sib, ssb=(0, 1), mcs=2, k_ssb=2)
    return rp.profile.load(name), np.asarray(y, np.complex64), 7.68e6, None, None

SIGNALS = {
    "adsb/mode_s": sig_adsb,
    "ax25/afsk1200": sig_ax25,
    "ble/adv": sig_ble,
    "ccsds/tm_rs": sig_ccsds,
    "ccsds/tm_concat": sig_ccsds,
    "dvbs/qpsk": sig_dvbs,
    "dvbs/qpsk_3_4": sig_dvbs,
    "dvbs2/normal": sig_dvbs2,
    "dvbs2/ts": sig_dvbs2,
    "ieee80211b/1m": sig_bits(1e6),
    "ieee80211b/long": sig_bits(1e6),
    "ieee802154/2450": sig_bits(250e3),
    "lte/downlink": sig_lte,
    "nr/downlink": sig_nr,
}

ESN0_MIN = {"ble/adv": 22.0} # GFSK at 0.5 modulation index with a two-bit preamble: its packets want more

def impair(x, Fs, sps, esn0, seed):
    """A carrier offset (1e-4 of the rate), a timing offset and AWGN at `esn0` dB a symbol, with a gap either side"""
    pad = np.zeros(2000, np.complex64)
    x = np.concatenate((pad, x, pad))
    p = float(np.mean(np.abs(x[np.abs(x) > 0])**2))
    d = rp.Data(x, Fs=Fs, sps=sps or 1, flush=True) + ch.CFO(1e-4*Fs) + ch.PhaseOffset(0.7)
    if sps and sps >= 2: # (at a sample a chip, as Mode S is, a fractional delay smears the pulses: not a channel's)
        d = d + ch.TimingOffset(0.37)
    d = d + ch.AWGN(esn0_db=esn0, power=p, seed=seed) # (sps 1 for LTE and NR: the SNR a sample)
    return np.asarray(d.data, np.complex64)

# ---- receiving, and whose time is whose

def _labels(rx, out, depth=0, seen=None):
    """{id(operator): "category:kind (step id)"} of the slots of a runner and the runners inside it"""
    seen = set() if seen is None else seen
    if id(rx) in seen or depth > 4:
        return out
    seen.add(id(rx))
    for s in getattr(rx, "slots", None) or []:
        st = getattr(s, "step", None)
        if st is None:
            continue
        out[id(s.op)] = f"{st.category}:{st.kind}" + (f" ({st.id})" if st.id and st.id != st.kind else "")
    for v in vars(rx).values() if hasattr(rx, "__dict__") else ():
        for w in (v.values() if isinstance(v, dict) else v if isinstance(v, (list, tuple)) else (v,)):
            if hasattr(w, "slots") or hasattr(w, "chains") or hasattr(w, "trunk") or hasattr(w, "pre"):
                _labels(w, out, depth + 1, seen)
    return out

def _label(op, labels, by_class):
    lab = labels.get(id(op))
    if lab is None: # Built per burst or per branch frame: name it as an operator of its class was
        lab = by_class.get(type(op)) or f"{type(op).__module__.removeprefix('rpps.')}.{type(op).__name__}"
    return lab

def _streams(src, outs):
    """Every stream that ran operators: the source's, and a burst's or a branch's own"""
    found = {id(src.stream): src.stream}
    def add(o):
        if isinstance(o, dict):
            for v in o.values():
                add(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                add(v)
        elif getattr(o, "stream", None) is not None:
            found.setdefault(id(o.stream), o.stream)
    add(outs)
    return list(found.values())

def _bits_out(outs):
    n = 0
    def add(o):
        nonlocal n
        if isinstance(o, dict):
            for v in o.values():
                add(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                add(v)
        elif hasattr(o, "data"):
            n += len(o.data)
    add(outs)
    return n

def run(p, x, Fs, sps, block):
    """(wall seconds, air seconds, {step label: busy seconds}, bits out) of one receive"""
    rx = p.rx()
    src = rp.source.Memory(x, Fs=Fs, sps=sps or 0.0)
    t = time.perf_counter()
    outs = list(src(block) + rx)
    dt = time.perf_counter() - t
    labels = _labels(rx, {})
    by_class = {}
    for s in getattr(rx, "slots", None) or []:
        if id(getattr(s, "op", None)) in labels:
            by_class.setdefault(type(s.op), labels[id(s.op)])
    busy = {}
    for st in _streams(src, outs):
        for op, v in st.busy.items():
            k = _label(op, labels, by_class)
            busy[k] = busy.get(k, 0.0) + v
    return dt, len(x)/Fs, busy, _bits_out(outs)

def kind_of(label):
    return label.split(" (")[0]

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", help="these profiles (default: every one in SIGNALS)")
    ap.add_argument("--seconds", type=float, default=0.2, help="signal per profile, about")
    ap.add_argument("--block", type=int, default=65536, help="samples a block")
    ap.add_argument("--esn0", type=float, default=15.0, help="Es/N0 [dB] of the AWGN (SNR a sample for LTE/NR)")
    ap.add_argument("--top", type=int, default=4, help="steps listed per profile")
    _bench.add_args(ap)
    a = ap.parse_args(argv)
    rec = _bench.Record("profile")
    names = a.only or list(SIGNALS)
    print(f"native={rp._accel.available()}  block={a.block}  Es/N0={a.esn0:g} dB  ~{a.seconds:g} s of signal each\n")
    shares, worst = {}, []
    for k, name in enumerate(names):
        rng = np.random.default_rng(k)
        try:
            p, x, Fs, sps, sent = SIGNALS[name](name, rng, a.seconds)
            x = impair(x, Fs, sps, max(a.esn0, ESN0_MIN.get(name, a.esn0)), k)
            run(p, x, Fs, sps, a.block) # (warm up)
            dt, air, busy, got = run(p, x, Fs, sps, a.block)
        except Exception as e:
            print(f"{name:18s} skipped: {type(e).__name__}: {str(e)[:100]}")
            continue
        total = sum(busy.values()) or 1e-12
        rt = dt/air
        rec.add(f"{name} real time", rt, "x", "lower")
        got_s = f"{got}/{sent} bits" if sent else f"{got} bits"
        print(f"{name:18s} {rt:7.3f}x real time  ({air*1e3:6.1f} ms of air, {got_s}; steps {100*total/dt:3.0f}% of the wall time)")
        for lab, v in sorted(busy.items(), key=lambda kv: -kv[1])[:a.top]:
            print(f"      {v/air:7.3f}x  {100*v/total:5.1f}%  {lab}")
            worst.append((v/air, name, lab))
        per = {}
        for lab, v in busy.items():
            per[kind_of(lab)] = per.get(kind_of(lab), 0.0) + v/total
        for kd, v in per.items():
            shares.setdefault(kd, []).append((v, name))
    if not shares:
        return 1
    print("\nStep kinds over the profiles: share of a profile's step time (median, mean, max) and where it is the most")
    rows = sorted(shares.items(), key=lambda kv: -statistics.median([v for v, _ in kv[1]]))
    for kd, vs in rows[:15]:
        vals = [v for v, _ in vs]
        top = max(vs)
        print(f"  {100*statistics.median(vals):5.1f}%  {100*statistics.mean(vals):5.1f}%  {100*top[0]:5.1f}%  in {len(vs):2d}  "
              f"{kd:34s} (most in {top[1]})")
    heaviest = sorted(worst, reverse=True)[:5]
    print("\nHeaviest steps (load: busy over air time):")
    for v, name, lab in heaviest:
        print(f"  {v:8.3f}x  {name:18s} {lab}")
    return _bench.finish(a, rec)

if __name__ == "__main__":
    sys.exit(main())
