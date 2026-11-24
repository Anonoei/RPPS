"""Burst time plans (rpps.plan), bursts on many carriers placed in them (rp.sync.Bursts, PlanTx), and the demo MF-TDMA
profile (profile/demo/mftdma) sent and received by itself"""
import numpy as np
import pytest

import rpps as rp

NCR = 27e6
BTU = 16200

def _plan(**over):
    obj = {"base": "plan", "clock": NCR,
           "tx_types": {"1": {"btus": 1, "offset": 270}, "2": {"btus": 2, "offset": 270}, "3": {"btus": 3, "offset": 270}},
           "frame_types": {"1": {"duration": 25*BTU, "btu": {"duration": BTU, "bw": 600e3, "symbol_rate": 500e3},
                                 "time_units": 25, "grid": [1e6, -1e6, 0], "sections": [[1, 4], [2, 4], [3, 3], [0, 20]]}},
           "sequences": {"0": {"start": 1000, "duration": 25*BTU, "centre": 14e9, "count": 65534,
                               "frames": [{"type": 1, "start": 0, "offset": 0}]}}}
    obj.update(over)
    return rp.plan.init("test", obj)

def test_slots_fill_each_carrier_in_time_then_the_next():
    """EN 301 545-2 7.5.1.4: numbered by ascending carrier, then time; a slot is its tx type's BTUs"""
    p = _plan()
    slots = p.frame_types[1].slots(p.tx_types)
    assert [(s.carrier, s.start, s.btus) for s in slots[:12]] == [
        (0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (0, 5, 2), (0, 7, 2), (0, 9, 2), (0, 11, 2), (0, 13, 2),
        (0, 15, 3), (0, 18, 3)]
    assert slots[13].carrier == 1 and slots[13].start == 0 # (a 3-BTU slot at 24 would pass the 25th BTU)
    assert sorted({s.freq for s in slots}) == [-1e6, 0.0] and slots[0].freq == -1e6 # (the grid, lowest first; 1 MHz unused)

def test_windows_count_wraps_and_assignments_merge_single_btus():
    p = _plan()
    w = p.windows(0, 2*25*BTU)
    assert len(w) == 2*len(p.frame_types[1].slots(p.tx_types))
    assert {x.superframe for x in w} == {65534, 65535} and w[0].start == 1000 and w[0].burst == 1270
    assert w[0].freq == 14e9 - 1e6
    slots = p.frame_types[1].slots(p.tx_types)
    g = next(s.number for s in slots if s.tx_type == 0)
    p.assign(0, 65534, 0, [(7, 0), (8, 2), (9, 3)], offset=g)
    got = [(x.slot - g, x.length//BTU, x.tx_type, x.assignment) for x in p.windows(0, 25*BTU, assigned=True)]
    assert got == [(0, 1, 0, 7), (1, 2, 2, 8), (3, 3, 3, 9)] # (a type on a single-BTU slot takes its BTUs; the next starts after)

def test_a_definition_round_trips():
    p = _plan()
    q = rp.plan.init("again", p.definition())
    assert [(w.slot, w.start, w.freq) for w in q.windows(0, 25*BTU)] == [(w.slot, w.start, w.freq) for w in p.windows(0, 25*BTU)]

def test_sections_that_do_not_fit_are_refused():
    with pytest.raises(rp.errors.ConfigError, match="carriers"):
        _plan(frame_types={"1": {"duration": 25*BTU, "btu": {"duration": BTU, "symbol_rate": 500e3}, "time_units": 25,
                                 "grid": [0], "sections": [[3, 9]]}})

# ---- the demo network: its own transmitter, a channel, its receiver

def _send(rng, superframes=3, per=5):
    p, plan = rp.profile.load("demo/mftdma"), rp.plan.load("demo/mftdma")
    tx, dur = p.tx(), plan.sequences[0].duration
    items = []
    for sf in range(superframes):
        for w in rng.choice(plan.windows(sf*dur, (sf + 1)*dur), per, replace=False):
            bits = plan.tx_types[w.tx_type].fields["bits"]
            items.append((w, {"superframe": w.superframe, "frame": w.frame, "slot": w.slot, "terminal": int(rng.integers(256)),
                              "seq": len(items)}, rng.integers(0, 2, bits - 32).astype(bool)))
    items.sort(key=lambda t: t[0].burst)
    x = np.concatenate([np.asarray(tx(rp.Data(pay, sps=1), fields=f).data) for _, f, pay in items] + [np.asarray(tx.flush().data)])
    return x, items

LEAD = 7777 # samples before the plan's first tick: the true origin is -LEAD/8e6*NCR ticks

def _channel(x, rng, cfo=1500.0, ppm=20.0, snr_db=15.0):
    x = np.concatenate((np.zeros(LEAD), x, np.zeros(4000)))
    n = np.arange(len(x))
    t = n*(1 + ppm*1e-6)
    x = (np.interp(t, n, x.real) + 1j*np.interp(t, n, x.imag))*np.exp(2j*np.pi*cfo*n/8e6)
    p = np.mean(np.abs(x[np.abs(x) > 0])**2)
    return (x + (rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))*np.sqrt(p*10**(-snr_db/10)/2)).astype(np.complex64)

def _receive(y, block=1 << 15, **params):
    rx, msgs, est = rp.profile.load("demo/mftdma", **params).rx(), [], None
    for blk in rp.source.Memory(y, Fs=8e6)(block):
        blk + rx
        msgs += rx.take_messages()
        est = blk.meta.stream.est.get("bursts", est)
    return {(m.fields["terminal"], m.fields["seq"]): m.fields.get("burst", {}) for m in msgs if m.kind == "hdr" and m.crc_ok}, est

def test_plan_tx_puts_each_burst_on_its_slot(rng):
    x, items = _send(rng, 1, 4)
    p = rp.profile.load("demo/mftdma")
    tx = p.tx()
    for w, f, pay in items:
        tx(rp.Data(pay, sps=1), fields=f)
    tx.flush()
    for (w, _, _), (start, n, freq) in zip(items, tx.sent):
        assert freq == pytest.approx(w.freq) # (the plan's centre is the output's)
        first = w.burst/NCR*8e6 # Its first symbol on the slot's burst start; the pulse's first half (12.5 symbols) before it
        assert first - start == pytest.approx(12.5*8e6/w.symbol_rate, abs=1)

@pytest.mark.parametrize("block", [1 << 15, 12345])
def test_demo_round_trip_with_the_plan(rng, block):
    """Sent by the profile, a 20 ppm clock, a 1.5 kHz offset, noise: every burst decoded and placed; the origin acquired
    from the bursts and followed from their preambles"""
    x, items = _send(rng)
    got, est = _receive(_channel(x, rng), block)
    assert set(got) == {(f["terminal"], f["seq"]) for _, f, _ in items}
    assert all((got[(f["terminal"], f["seq"])].get("superframe"), got[(f["terminal"], f["seq"])].get("slot"),
                got[(f["terminal"], f["seq"])].get("tx_type")) == (w.superframe, w.slot, w.tx_type) for w, f, _ in items)
    assert est["locked"] and abs(est["origin"] + LEAD/8e6*NCR) < 2.5*NCR/250e3 # (measured: within 180 ticks, 1.7 symbols)

def test_demo_detection_only(rng):
    """No plan: the carriers given, each burst's type from its unique word"""
    x, items = _send(rng)
    C = [{"freq": -2e6, "width": 337.5e3, "rate": 250e3}, {"freq": -1.5e6, "width": 337.5e3, "rate": 250e3},
         {"freq": 1.5e6, "width": 675e3, "rate": 500e3}]
    got, _ = _receive(_channel(x, rng), plan="", carriers=C)
    assert set(got) == {(f["terminal"], f["seq"]) for _, f, _ in items}
    assert all("superframe" not in b for b in got.values())

def test_bursts_any_cut(rng):
    x, _ = _send(rng, 2, 4)
    y = _channel(x, rng)
    def cut(block, **kw):
        b, out = rp.sync.Bursts(sps=4, factor=4, win=16, min_s=0.5e-3, **kw), []
        for blk in rp.source.Memory(y, Fs=8e6)(block):
            out += blk + b
        return {(z.ctx["burst"]["freq"], z.meta.pos, len(z), z.ctx["burst"].get("slot")): np.asarray(z.data) for z in out}
    for kw in ({"plan": rp.plan.load("demo/mftdma"), "lock": 4},
               {"carriers": [{"freq": -2e6, "width": 337.5e3, "rate": 250e3}, {"freq": 1.5e6, "width": 675e3, "rate": 500e3}]}):
        a, b = cut(len(y) + 1, **kw), cut(4321, **kw)
        assert a.keys() == b.keys() and len(a) >= 2
        assert all(np.allclose(a[k], b[k], atol=1e-6) for k in a)

def test_plan_gate_cuts_what_energy_misses(rng):
    """The origin given (as a forward link's clock gives it): at -6 dB over the capture the energy gate loses bursts that
    the plan's windows still cut and decode (measured over 10 seeds: energy 0-2 of 15, plan 9-14, at least 7 more)"""
    import json
    x, items = _send(rng)
    y = _channel(x, rng, cfo=500.0, ppm=0.0, snr_db=-6.0)
    obj = json.loads(json.dumps(rp.profile.load("demo/mftdma").obj))
    res = {}
    for gate in ("energy", "plan"):
        for st in obj["steps"]:
            if st.get("sync") == "burst":
                st["gate"], st["origin"] = gate, -LEAD/8e6*NCR
        rx, n = rp._config.load_obj("profile", "test/gate", obj).rx(), 0
        for blk in rp.source.Memory(y, Fs=8e6)(1 << 15):
            blk + rx
            n += sum(1 for m in rx.take_messages() if m.kind == "hdr" and m.crc_ok)
        res[gate] = n
    assert res["plan"] >= 9 and res["plan"] - res["energy"] >= 5

# ---- hopping and trains

def _hopping_plan():
    """The demo's 500 kBd frame on one carrier that hops over four channels, superframe by superframe"""
    d = rp.plan.load("demo/mftdma").definition()
    d["frame_types"] = {"2": d["frame_types"]["2"]}
    d["tx_types"] = {"130": d["tx_types"]["130"]}
    s = d["sequences"]["0"]
    s["frames"] = [{"type": 2, "start": 0, "offset": 0.0}]
    s["hop"] = {"channels": [-3e6, -1e6, 1e6, 3e6], "carrier": "(k*step + f) % 4", "params": {"step": 3}}
    return rp.plan.init("hop", d)

def test_hop_moves_each_superframe():
    p = _hopping_plan()
    dur = p.sequences[0].duration
    freqs = [p.windows(k*dur, (k + 1)*dur)[0].freq for k in range(6)]
    assert freqs == [-3e6, 3e6, 1e6, -1e6, -3e6, 3e6] # (3k mod 4)
    assert sorted(p.carriers()) == [-3e6, -1e6, 1e6, 3e6]

def test_ble_csa1():
    from rpps.plan import HOPS
    assert [HOPS["ble_csa1"](k, 0, 37, hop=5) for k in range(4)] == [5, 10, 15, 20]
    assert HOPS["ble_csa1"](1, 0, 37, hop=5, used=[0, 1, 2, 3]) == 10 % 4 # (an unused channel, remapped)

def test_a_hopping_carrier_round_trip(rng):
    plan = _hopping_plan()
    dur, items = plan.sequences[0].duration, []
    tx = rp.profile.load("demo/mftdma").tx()
    tx.burst_tx.plan = plan
    for k in range(6):
        for w in rng.choice(plan.windows(k*dur, (k + 1)*dur), 2, replace=False):
            items.append((w, {"superframe": w.superframe, "frame": w.frame, "slot": w.slot, "terminal": int(rng.integers(256)),
                              "seq": len(items)}, rng.integers(0, 2, 972 - 32).astype(bool)))
    items.sort(key=lambda t: t[0].burst)
    x = np.concatenate([np.asarray(tx(rp.Data(pay, sps=1), fields=f).data) for _, f, pay in items] + [np.asarray(tx.flush().data)])
    y = _channel(x, rng, ppm=0.0)
    rx = rp.profile.load("demo/mftdma").rx()
    rx.burst.control(plan=plan)
    msgs = []
    for blk in rp.source.Memory(y, Fs=8e6)(1 << 15):
        blk + rx
        msgs += rx.take_messages()
    got = {(m.fields["terminal"], m.fields["seq"]): m.fields["burst"] for m in msgs if m.kind == "hdr" and m.crc_ok}
    assert {(f["terminal"], f["seq"]) for _, f, _ in items} == set(got)
    assert all(got[(f["terminal"], f["seq"])].get("superframe") == w.superframe and got[(f["terminal"], f["seq"])]["freq"] == w.freq
               for w, f, _ in items)

def test_a_train_is_a_superframe_of_its_bursts():
    """A burst every 100 ms for a second, every minute (a beacon train): ten frames in a 60 s superframe"""
    p = rp.plan.init("train", {"base": "plan", "clock": 1000.0,
                               "tx_types": {"1": {"btus": 1}},
                               "frame_types": {"1": {"duration": 10, "btu": {"duration": 7, "symbol_rate": 40e3}, "time_units": 1,
                                                     "sections": [[1, 0]]}},
                               "sequences": {"0": {"start": 0, "duration": 60000, "frames": [{"type": 1, "start": 0, "repeat": 10, "every": 100}]}}})
    starts = [w.start for w in p.windows(0, 2*60000)]
    assert starts == [100*i for i in range(10)] + [60000 + 100*i for i in range(10)]

def test_a_plan_learned_from_detected_bursts(rng, tmp_path):
    """No forward link: the plan learned from the bursts detection finds (its BTU, frame, slots), saved and loaded as a
    definition, then used to place them: the same slot of every frame gets the same name"""
    x, items = _send(rng, superframes=8, per=6)
    y = _channel(x, rng)
    C = [{"freq": -2e6, "width": 337.5e3, "rate": 250e3}, {"freq": -1.5e6, "width": 337.5e3, "rate": 250e3},
         {"freq": 1.5e6, "width": 675e3, "rate": 500e3}]
    bursts = rp.Data(y, Fs=8e6, flush=True) + rp.sync.Bursts(carriers=C, sps=4, factor=4, win=16, min_s=0.5e-3)
    g = rp.analysis.spectrum.burst.learn_plan(bursts, carriers=C)
    assert g.evidence["btu"] == pytest.approx(BTU/NCR, rel=1e-3) and g.evidence["frame"] == pytest.approx(25*BTU/NCR, rel=1e-3)
    g.save("learned_test", tmp_path/"plan")
    rp.add_config_path(tmp_path)
    try:
        plan = rp.plan.load("guess/learned_test")
    finally:
        rp._config._EXTRA.remove(tmp_path) if tmp_path in rp._config._EXTRA else None
    rx = rp.profile.load("demo/mftdma").rx()
    rx.burst.control(plan=plan)
    msgs = []
    for blk in rp.source.Memory(y, Fs=8e6)(1 << 15):
        blk + rx
        msgs += rx.take_messages()
    got = {(m.fields["terminal"], m.fields["seq"]): m.fields["burst"] for m in msgs if m.kind == "hdr" and m.crc_ok}
    assert len(got) == len(items)
    names = {}
    for w, f, _ in items:
        b = got[(f["terminal"], f["seq"])]
        names.setdefault((w.frame, w.slot), set()).add((b["frame"], b["slot"]))
    assert all(len(v) == 1 for v in names.values()) and len(set().union(*names.values())) == len(names)

def test_a_capture_at_any_rate(rng):
    """7.8125 MS/s (100 MS/s / 12.8) is no whole multiple of 4 x 250 kBd: each carrier is decimated, then resampled"""
    from rpps.signal.sample.fractional import FractionalPolyphase
    x, items = _send(rng)
    y = np.asarray((rp.Data(_channel(x, rng), Fs=8e6, flush=True) + FractionalPolyphase(7.8125/8)).data)
    rx, got = rp.profile.load("demo/mftdma").rx(), set()
    for blk in rp.source.Memory(y, Fs=7.8125e6)(1 << 15):
        blk + rx
        got |= {(m.fields["terminal"], m.fields["seq"]) for m in rx.take_messages() if m.kind == "hdr" and m.crc_ok}
    assert got == {(f["terminal"], f["seq"]) for _, f, _ in items}

# ---- checking a plan against the bursts detected

C3 = [{"freq": -2e6, "width": 337.5e3, "rate": 250e3}, {"freq": -1.5e6, "width": 337.5e3, "rate": 250e3},
      {"freq": 1.5e6, "width": 675e3, "rate": 500e3}]

def _detected(rng):
    x, items = _send(rng, superframes=6, per=6)
    y = _channel(x, rng)
    return rp.Data(y, Fs=8e6, flush=True) + rp.sync.Bursts(carriers=C3, sps=4, factor=4, win=16, min_s=0.5e-3), items

def _demo(edit=None):
    d = rp.plan.load("demo/mftdma").definition()
    if edit:
        edit(d)
    return rp.plan.init("edited", d)

def test_the_right_plan_explains_every_burst_and_finds_its_origin(rng):
    bursts, items = _detected(rng)
    check = rp.analysis.spectrum.burst.check_plan(rp.plan.load("demo/mftdma"), bursts)
    assert len(bursts) == len(items) and check.explained == 1.0 and check.rms < 20e-6
    assert abs(check.origin - (-LEAD/8e6*NCR)) < check.tol*NCR # (where the plan's tick 0 is in the capture: its first superframe)
    assert int(np.sum(check.filled)) == len(items) and np.all(check.filled <= 1)
    learned = rp.analysis.spectrum.burst.learn_plan(bursts, carriers=C3)
    assert rp.analysis.spectrum.burst.check_plan(learned.plan, bursts).explained == 1.0

def test_a_wrong_btu_or_slot_layout_leaves_bursts_in_no_slot(rng):
    bursts, _ = _detected(rng)
    def btu(d):
        for ft in d["frame_types"].values():
            ft["btu"]["duration"], ft["duration"] = int(ft["btu"]["duration"]*1.03), int(ft["duration"]*1.03)
        d["sequences"]["0"]["duration"] = int(d["sequences"]["0"]["duration"]*1.03)
    def layout(d):
        d["frame_types"]["1"]["sections"] = [[129, 2], [128, 2], [128, 1], [129, 2], [128, 0]]
    check = rp.analysis.spectrum.burst.check_plan
    assert check(_demo(btu), bursts).explained < 0.6
    c = check(_demo(layout), bursts)
    assert c.counts["none"] >= 3 and c.explained < 0.95

def _ideal(plan, origin, rate, seconds, every=7):
    """A burst every `every`th window over `seconds` of capture, its kind its tx type's word, at exact starts"""
    ws = plan.windows(origin, origin + seconds*plan.clock*(1 + rate))[::every]
    return [{"freq": w.freq, "start": (w.burst - origin)/(plan.clock*(1 + rate)), "length": 0.8*w.length/plan.clock,
             "kind": plan.tx_types[w.tx_type].fields["word"]} for w in ws]

def test_another_kinds_slot_and_the_clock_offset():
    plan = rp.plan.load("demo/mftdma")
    recs = _ideal(plan, 123456.0, 150e-6, 4.0) # (600 us over the capture: twice the tolerance)
    c = rp.analysis.spectrum.burst.check_plan(plan, recs)
    assert c.explained == 1.0 and c.rate == pytest.approx(150e-6, abs=1e-6) and abs(rp.sync.track.wrap(c.origin - 123456.0, plan.sequences[0].duration)) < 50
    stale = rp.analysis.spectrum.burst.check_plan(plan, recs, fit=False, origin=123456.0) # (the clock not followed)
    assert stale.counts["none"] > 0
    def swap(d):
        a, b = d["tx_types"]["128"]["fields"], d["tx_types"]["129"]["fields"]
        a["word"], b["word"] = b["word"], a["word"]
    c = rp.analysis.spectrum.burst.check_plan(_demo(swap), recs, origin=123456.0, rate=150e-6, fit=False)
    assert c.counts["kind"] > 0 and c.counts["kind"] + c.counts["slot"] == len(recs)

def test_the_btu_is_made_exact_over_the_whole_capture():
    """learn_plan's coarse BTU is right to 1e-4 of itself: over 16000 BTUs that is more than half of one, so each
    start's count of BTUs would slip; the starts' line makes it exact"""
    rng = np.random.default_rng(1)
    q = 0.6e-3
    k = np.sort(rng.choice(16000, 500, replace=False))
    starts = 0.01 + k*q + rng.normal(0, 16e-6, len(k))
    got = rp.analysis.spectrum.burst._stretch(starts, q*(1 + 1e-4), 0.1)
    assert abs(got/q - 1) < 1e-6

def _sparse(seed, rate=30e-6, sec=30.0, use=0.02, origin=5e5):
    """Bursts in `use` of the demo plan's windows, at exact starts plus 5 us of jitter, their kinds known"""
    plan, rng = rp.plan.load("demo/mftdma"), np.random.default_rng(seed)
    ws = [w for w in plan.windows(origin, origin + sec*plan.clock*(1 + rate)) if rng.random() < use]
    return [{"freq": w.freq, "start": (w.burst - origin)/(plan.clock*(1 + rate)) + rng.normal(0, 5e-6),
             "length": 0.8*w.length/plan.clock, "kind": plan.tx_types[w.tx_type].fields["word"]} for w in ws]

def test_a_burst_of_a_known_kind_is_placed_in_its_kinds_slot_not_the_filler_beside_it():
    """A learned plan fills the BTUs no burst used with untyped single-BTU slots: a burst of a known kind is placed in its
    own kind's slot, not one BTU over in the filler (where it would overrun)"""
    B = rp.analysis.spectrum.burst
    recs = _sparse(0)
    g = B.learn_plan(recs)
    assert B.check_plan(g.plan, recs).explained == 1.0
