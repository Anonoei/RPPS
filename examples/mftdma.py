"""An MF-TDMA network whose burst time plan comes over the air (DVB-RCS2's way), and the same bursts without it

    python examples/mftdma.py

The hub's forward link (DVB-S2) carries the plan as tables in GSE packets (SCT: superframes and their frames, FCT2: each
frame's carriers and timeslots, BCT: each burst type, TBTP2: which terminal has which slot) and the network clock (NCR
packets, each the time of an earlier PL frame's start). The terminals' bursts (profile/demo/mftdma: three carriers, two
symbol rates, three burst types each with its unique word, LDPC units, CRC) go where the plan says.

The receiver reads the forward link (profile/dvbrcs2/forward), builds the plan and the clock from its messages
(rp.plan.builder("dvb-rcs2")), and gives both to the return link's burst step: every burst is cut on its carrier, decoded,
and named by its superframe, frame, timeslot, burst type and terminal. The return path's delay is not told: the bursts'
own unique words pull the clock onto it. Without the forward link, a plan learned from the bursts themselves
(rp.analysis.spectrum.burst.learn_plan) places them as well, under its own names.
"""
import numpy as np

import rpps as rp
from rpps.link import gse
from rpps.plan import rcs2

N0, DELAY = 10**9, 1e-4 # The NCR when both captures start; the return path's delay, unknown to the receiver [s]

def network(rng, superframes=3, per=5, frames=6):
    """(forward link samples at 4 MS/s, return link samples at 8 MS/s, what was sent)"""
    plan = rp.plan.load("demo/mftdma")
    s = plan.sequences[0]
    s.start, s.count = N0 + 27000, 100
    sent = []
    for k in range(superframes): # A few bursts each superframe, each slot assigned to its terminal in that superframe's TBTP2
        ws = plan.windows(s.start + k*s.duration, s.start + (k + 1)*s.duration)
        slots = {}
        for i in sorted(rng.choice(len(ws), per, replace=False)):
            w, term = ws[i], int(rng.integers(1, 250))
            sent.append((w, {"superframe": w.superframe, "frame": w.frame, "slot": w.slot, "terminal": term, "seq": len(sent)},
                         rng.integers(0, 2, plan.tx_types[w.tx_type].fields["bits"] - 32).astype(bool)))
            slots.setdefault(w.frame, {})[w.slot] = term
        for fr, sl in slots.items():
            lo, hi = min(sl), max(sl)
            plan.assign(0, ws[0].superframe, fr, [(sl.get(j, 0), 0) for j in range(lo, hi + 1)], offset=lo)
    # The forward link: the tables in every frame, frame k's NCR packet the NCR of frame k-2's start
    dfl = rp.profile.table("dvbs2/modcod").row({"modcod": 4, "short": 0, "pilots": 1})["kbch"] - 80
    tables = rcs2.tables(plan, network_id=7, superframes=[(0, s.count + k) for k in range(superframes)])
    fwd = rp.profile.load("dvbs2/normal")
    def send(sof):
        fields = []
        for k in range(frames):
            pdus = ([(rcs2.PROTO_NCR, rcs2.ncr_packet(N0 + 27e6*sof[k - 2]/4e6))] if k >= 2 else []) + \
                   [(rcs2.PROTO_L2S, t) for t in tables]
            fields.append(np.unpackbits(np.frombuffer(gse.pack(pdus, dfl//8)[0], np.uint8)).astype(bool))
        return fwd.tx()(rp.Data(np.concatenate(fields), Fs=1e6, sps=1, flush=True), fields=[{"modcod": 4, "pilots": 1}]*frames)
    o = send([0]*frames)
    sof = [o.meta.delay*4e6 + k*(90 + 360*90 + 22*36)*4 for k in range(frames)] # (QPSK, pilots: a frame's samples)
    xf = np.asarray(send(sof).data)
    xf = (xf*np.exp(0.7j) + 0.05*(rng.normal(size=len(xf)) + 1j*rng.normal(size=len(xf)))).astype(np.complex64)
    # The return link: each terminal's burst on its slot, the path's delay late
    tx = rp.profile.load("demo/mftdma").tx()
    tx.burst_tx.plan, tx.burst_tx.origin = plan, N0 - DELAY*27e6
    sent.sort(key=lambda t: t[0].burst)
    xr = np.concatenate([np.asarray(tx(rp.Data(p, sps=1), fields=f).data) for _, f, p in sent] + [np.asarray(tx.flush().data)])
    xr = np.concatenate((xr, np.zeros(4000)))
    xr = (xr + 0.02*(rng.normal(size=len(xr)) + 1j*rng.normal(size=len(xr)))).astype(np.complex64)
    return xf, xr, sent

def decode(xr, plan=None, origin=None, **params):
    rx = rp.profile.load("demo/mftdma", **params).rx()
    if plan is not None:
        rx.burst.control(plan=plan, origin=origin)
    got = {}
    for blk in rp.source.Memory(xr, Fs=8e6)(1 << 15):
        blk + rx
        for m in rx.take_messages():
            if m.kind == "hdr" and m.crc_ok:
                got[(m.fields["terminal"], m.fields["seq"])] = m.fields["burst"]
    return got

def main():
    rng = np.random.default_rng(1)
    xf, xr, sent = network(rng)

    fwd, b = rp.profile.load("dvbrcs2/forward").rx(), rp.plan.builder("dvb-rcs2")
    for blk in rp.source.Memory(xf, Fs=4e6, sps=4)(1 << 16):
        blk + fwd
        for m in fwd.take_messages():
            b.feed(m)
    print(f"forward link: {b.plan}; NCR at t=0 {b.clock(0) - N0:+.0f} ticks from the hub's (its filters' delay)")

    got = decode(xr, b.plan, b.clock(0))
    for w, f, _ in sent:
        g = got[(f["terminal"], f["seq"])]
        assert (g["superframe"], g["frame"], g["slot"], g["tx_type"], g["assignment"]) == (w.superframe, w.frame, w.slot, w.tx_type, f["terminal"])
    print(f"return link, plan from the air: {len(got)} of {len(sent)} bursts, each in its slot, named by its terminal")

    C = [{"freq": -2e6, "width": 337.5e3, "rate": 250e3}, {"freq": -1.5e6, "width": 337.5e3, "rate": 250e3},
         {"freq": 1.5e6, "width": 675e3, "rate": 500e3}]
    bursts = rp.Data(xr, Fs=8e6, flush=True) + rp.sync.Bursts(carriers=C, sps=4, factor=4, win=16, min_s=0.5e-3)
    learned = rp.analysis.spectrum.burst.learn_plan(bursts, carriers=C)
    got = decode(xr, learned.plan)
    assert len(got) == len(sent)
    print(f"return link, plan learned from its own bursts: {learned}; {len(got)} of {len(sent)} bursts")

if __name__ == "__main__":
    main()
