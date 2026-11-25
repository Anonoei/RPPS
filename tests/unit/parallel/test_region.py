"""A profile's parallel region: where it starts, its plan and its worker"""
import threading
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.errors import ConfigError
from rpps.parallel import Frame, Scheduler, ThreadExecutor
from support.parallel import frames, halve, receive
from support.dvbs2 import KNOWN

# ---- the profile's region

def profile_with(name, parallel, **params):
    obj = _config.read(_config.find("profile", name))
    obj = dict(obj)
    obj["parallel"] = parallel
    values = {**_config.declared(obj), **params}
    return rp.profile.Profile(name, _config.resolve(obj, params, name), params=values)

def test_dvbs2_region_starts_after_the_pls_header():
    p = rp.profile.load("dvbs2/normal")
    assert [s.id for s in p.region.steps] == ["plscram", "phase", "pilots", "map", "bitil", "ldpc", "bch", "bbscram", "bbheader"]
    assert p.region.driven["bbscram"] == ["length"]

def test_profiles_without_a_marker_have_no_region():
    assert rp.profile.load("dvbs/qpsk_3_4").region is None

def test_a_step_with_state_across_frames_is_refused():
    steps = rp.profile.load("dvbs/qpsk_3_4").steps
    for st in steps:
        if st.id in ("align", "viterbi") or st.category == "sync":
            with pytest.raises(ConfigError, match="not frame scoped"):
                profile_with("dvbs/qpsk_3_4", st.id)
            return
    pytest.fail("no stream step found")

def test_a_header_inside_the_region_is_refused():
    for marker in ("pls", "frame"):
        with pytest.raises(ConfigError, match="publishes"):
            profile_with("dvbs2/normal", marker)

def test_unknown_marker_is_refused():
    with pytest.raises(ConfigError, match="no such step"):
        profile_with("dvbs2/normal", "nope")

def test_an_operator_that_needs_a_control_is_refused_without_it():
    from rpps.profile import parallel
    from rpps.profile.steps import Step
    sc = rp.code.scram.load("additive/dvbs2_bb")
    st = Step("code", "scramble", [("/", sc)], [("*", sc)], rp.Rep.DATA, rp.Rep.DATA)
    assert parallel.scope_of(st, ())[0] == "stream"
    assert parallel.scope_of(st, ("length",))[0] == "frame"

# ---- the parallel runner on DVB-S2: the same output as the serial runner

# ---- kernels, plans, the worker

def region_worker(profile="dvbs2/normal"):
    p = rp.profile.load(profile, sync=KNOWN)
    return p, rp.parallel.PlanWorker(lambda: p._build()[p.region.first:])

def test_plan_describes_itself_and_is_cached_by_key(s2):
    import json
    p, x, o, data, dfl = s2
    w = rp.parallel.PlanWorker(lambda: p._build()[p.region.first:])
    f = Frame(0, np.zeros(33192, np.complex64), {"plscram": {"length": 100}}, tags=(rp.Data, rp.Category.UNK, rp.dtype.SYMBOLS,
                                                                                    rp.Rep.SYMBOLS, 1e6, 1, 0.0, 0.0))
    plan = w.plan(f)
    assert w.plan(f) is plan and len(w.plans) == 1
    d = plan.describe(33192)
    assert d[0]["kernel"] == "rotate:plscram" and d[0]["in"] == 33192 and d[0]["out"] == 33192
    assert [k["kernel"] for k in d][-1] == "loop:bbheader" and d[-1]["out"] is None # (a step with no kernel: its length is known only by running)
    json.dumps(d) # a backend gets it as data

def test_unknown_controlled_parameter_is_refused():
    p = rp.profile.load("dvbs2/normal", sync=KNOWN)
    steps = p._build()[p.region.first:]
    with pytest.raises(ConfigError, match="cannot be controlled"):
        rp.parallel.compile(steps, {"ldpc": {"nope": 1}})

def test_a_batch_holds_one_key():
    p, w = region_worker()
    tags = (rp.Data, rp.Category.UNK, rp.dtype.SYMBOLS, rp.Rep.SYMBOLS, 1e6, 1, 0.0, 0.0)
    a = Frame(0, np.zeros(10, np.complex64), {"x": 1}, tags=tags)
    b = Frame(1, np.zeros(10, np.complex64), {"x": 2}, tags=tags)
    with pytest.raises(ValueError, match="one key"):
        w([a, b])

def test_a_failing_row_is_dropped_and_the_rest_go_on(s2):
    """Frames of one batch share a plan; one that cannot be processed is a gap, the others come out as they would alone"""
    from rpps.base.kernel import Batch, Kernel, Form
    class Picky(Kernel):
        name = "picky"
        def run(self, xp, b):
            b.fail(1, "bad row")
            return b
    plan = rp.parallel.BatchPlan([Picky()])
    b = Batch(np.zeros((3, 4), np.float32), Form(rp.Data, rp.Category.UNK, rp.dtype.LLR, rp.Rep.CHANNEL, 1, 1, 0, 0), [0, 4, 8], 4)
    b = plan.run(np, b)
    assert list(b.bad) == [False, True, False] and b.why == {1: "bad row"}
    assert plan.spent["picky"] >= 0 and plan.batches == 1

def test_a_batch_is_split_across_threads_and_comes_back_in_order():
    sizes, lock = [], threading.Lock()
    def worker(batch):
        with lock:
            sizes.append(len(batch))
        return halve(batch)
    with Scheduler(ThreadExecutor(4), worker, max_batch=64, eager=False) as sch:
        out = []
        for f in frames(10):
            out += sch.push(f)
        out += sch.kick() + sch.flush()
    assert [r.seq for r in out] == list(range(10))
    assert sum(sizes) == 10 and len(sizes) > 1 and max(sizes) <= 3

def test_kick_runs_what_is_waiting():
    with Scheduler("inline", halve, max_batch=64, eager=False) as sch:
        for f in frames(3):
            assert sch.push(f) == []
        assert [r.seq for r in sch.kick()] == [0, 1, 2]

def test_sequences_are_shared_and_safe_across_threads():
    from concurrent.futures import ThreadPoolExecutor
    a, b = rp.sequence.load("dvbs2_pl"), rp.sequence.load("dvbs2_pl")
    assert a.components[0] is b.components[0] # one generation per definition
    want = rp.sequence.load("dvbs2_pl").bits(5000, 100).copy()
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda n: rp.sequence.load("dvbs2_pl").bits(n, 100)[:4900], [5000 + 7*i for i in range(16)]))
    assert all(np.array_equal(g, want[:4900]) for g in got)

def test_the_region_counts_its_workers_time(s2):
    """The region's kernels run on workers (threads, processes): their time comes back with the results, into the stream's
    load as one lane per worker, and by kernel in est["parallel"]["spent"]"""
    p, x, o, data, dfl = s2
    rx = p.rx(parallel=rp.parallel.Parallel("thread", workers=3))
    got, st = receive(rx, x, o)
    rx.close()
    key = next(k for k in st.busy if isinstance(k, str) and k.startswith("parallel:"))
    spent = st.est["parallel"]["spent"]
    assert st.busy[key] == pytest.approx(sum(spent.values())) and st.busy[key] > 0
    assert st.thread_of[key] == "parallel" and st.lanes["parallel"] == 3
    assert any(k.startswith("rotate:") for k in spent)
    st.add_air(len(x)/o.Fs)
    assert st.realtime() > 0
