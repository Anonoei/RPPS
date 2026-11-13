"""Parallel processing of frames: work units, executors, scheduler and the profile's parallel region (README.dev.md,
Parallelism)"""
import threading
import time

import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.errors import ConfigError
from rpps.parallel import Frame, Result, Scheduler, InlineExecutor, ThreadExecutor, DROPPED

def frames(n, modcods=(1,)):
    return [Frame(i, np.full(8, i, dtype=np.float32), {"ldpc": {"name": modcods[i % len(modcods)]}}, pos=8*i, pos_out=4*i, n_out=4)
            for i in range(n)]

def halve(batch):
    """A worker: the first half of each frame (the 'decode')"""
    return [Result(f.seq, f.data[:4] * 2, f.pos_out, f.n_out) for f in batch]

def drain(sch, fs):
    out = []
    for f in fs:
        out += sch.push(f)
    return out + sch.flush()

def test_key_groups_by_params_and_shape():
    a, b, c = frames(3, modcods=(1, 2))
    assert a.key == c.key and a.key != b.key
    assert Frame(0, np.zeros(9, np.float32), a.params).key != a.key

@pytest.mark.parametrize("ex", ["inline", ThreadExecutor(4)])
def test_results_equal_the_worker_in_order(ex):
    fs = frames(50, modcods=(1, 2, 3))
    with Scheduler(ex, halve, max_batch=4) as sch:
        out = drain(sch, fs)
    assert [r.seq for r in out] == list(range(50))
    for r, f in zip(out, fs):
        np.testing.assert_array_equal(r.data, f.data[:4]*2)
        assert (r.pos_out, r.n_out) == (f.pos_out, f.n_out)

def test_late_early_batches_still_come_out_in_order():
    """Earlier frames finish last: nothing is released before the frames ahead of it"""
    def slow_first(batch):
        time.sleep(0.15 if batch[0].seq == 0 else 0.0)
        return halve(batch)
    with Scheduler(ThreadExecutor(4), slow_first, max_batch=1) as sch:
        fs = frames(8)
        got = []
        for f in fs:
            got += [r.seq for r in sch.push(f)]
        assert got == [] or got == list(range(len(got)))
        got += [r.seq for r in sch.flush()]
    assert got == list(range(8))

def test_in_flight_is_bounded():
    live, peak, lock = 0, 0, threading.Lock()
    def worker(batch):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.01)
        with lock:
            live -= 1
        return halve(batch)
    ex = ThreadExecutor(8, capacity=3)
    with Scheduler(ex, worker, max_batch=1) as sch:
        drain(sch, frames(40))
        assert sch.max_flying <= 3
    assert peak <= 3

def test_batches_hold_one_key_and_at_most_max_batch():
    seen = []
    def worker(batch):
        seen.append([f.key for f in batch])
        return halve(batch)
    with Scheduler("inline", worker, max_batch=3, eager=False) as sch:
        drain(sch, frames(20, modcods=(1, 2)))
    assert all(len(set(b)) == 1 and len(b) <= 3 for b in seen)
    assert sum(len(b) for b in seen) == 20

def test_flush_runs_a_partial_batch_and_a_trickle_is_not_held_back():
    with Scheduler("inline", halve, max_batch=64) as sch:
        assert [r.seq for r in sch.push(frames(1)[0])] == [0] # eager: idle capacity runs it now
    with Scheduler("inline", halve, max_batch=64, eager=False) as sch:
        assert sch.push(frames(1)[0]) == []
        assert [r.seq for r in sch.flush()] == [0]

def test_dropped_frame_keeps_its_place_and_position():
    fs = frames(5)
    def worker(batch):
        return [Result.dropped(f, "timing slip") if f.seq == 2 else r for f, r in zip(batch, halve(batch))]
    with Scheduler("inline", worker) as sch:
        out = drain(sch, fs)
    assert [r.status for r in out] == ["ok", "ok", DROPPED, "ok", "ok"]
    assert (out[2].pos_out, out[2].n_out) == (fs[2].pos_out, 4) and len(out[2].data) == 0
    assert out[2].reports[0]["frame"] == 2

def test_frames_must_come_in_order():
    with Scheduler("inline", halve) as sch:
        with pytest.raises(ValueError):
            sch.push(frames(2)[1])

def test_a_worker_bug_comes_out_and_a_wrong_count_is_refused():
    def boom(batch):
        raise KeyError("bug")
    with Scheduler("inline", boom) as sch:
        with pytest.raises(KeyError):
            sch.push(frames(1)[0])
    with Scheduler("inline", lambda b: []) as sch:
        with pytest.raises(RuntimeError):
            sch.push(frames(1)[0])

def test_per_thread_workers():
    made = []
    class W:
        def __call__(self, batch):
            return halve(batch)
        def per_thread(self):
            made.append(threading.get_ident())
            return halve
    with Scheduler(ThreadExecutor(3), W(), max_batch=1) as sch:
        drain(sch, frames(30))
    assert len(made) == 3 # one per thread of the pool, made up front

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

KNOWN = {"timing": {"method": "known"}}
FRAMES = [{"modcod": 4, "pilots": 1}, {"modcod": 12, "pilots": 1}, {"modcod": 1, "pilots": 0}, {"modcod": 4, "pilots": 1}]

@pytest.fixture(scope="module")
def s2():
    p = rp.profile.load("dvbs2/normal", sync=KNOWN)
    t = rp.profile.table("dvbs2/modcod")
    dfl = [t.row({**f, "short": 0})["kbch"] - 80 for f in FRAMES]
    data = np.random.default_rng(4).integers(0, 2, sum(dfl)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=FRAMES)
    return p, np.concatenate((o.data, np.zeros(200, np.complex64))), o, data, dfl

def receive(rx, x, o, cuts=None, noise=0.0):
    rng = np.random.default_rng(5)
    x = (x*np.exp(0.7j) + noise*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)
    st, outs, i = None, [], 0
    cuts = cuts or [len(x)]
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], Fs=o.Fs, sps=4, delay=o.delay, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        outs.append(np.asarray(rx(d).data).astype(bool))
    return np.concatenate(outs), st

@pytest.mark.parametrize("ex,kw", [("inline", {}), ("thread", {"workers": 4}), ("thread", {"workers": 3, "max_batch": 1}),
                                   ("process", {"workers": 2}), ("process", {"workers": 3, "max_batch": 2})])
def test_same_output_as_the_serial_runner(s2, ex, kw):
    p, x, o, data, dfl = s2
    want, _ = receive(p.rx(), x, o)
    rx = p.rx(parallel=rp.parallel.Parallel(ex, **kw))
    got, st = receive(rx, x, o)
    rx.close()
    assert np.array_equal(got, want) and np.array_equal(got, data)
    assert st.est["parallel"]["frames"] == len(FRAMES)
    assert st.est["dvbs2_bb"]["crc_failed"] == 0 if "dvbs2_bb" in st.est else True

def test_any_block_cut(s2):
    p, x, o, data, dfl = s2
    rx = p.rx(parallel=rp.parallel.Parallel("thread", workers=2))
    got, _ = receive(rx, x, o, cuts=[30011]*(len(x)//30011) + [len(x) % 30011])
    rx.close()
    assert np.array_equal(got, data)

def test_noise_and_failed_frames_match_the_serial_runner(s2):
    """At low SNR some frames fail to decode: the same ones, in the same place, as in the serial runner"""
    p, x, o, data, dfl = s2
    want, _ = receive(p.rx(), x, o, noise=0.5)
    rx = p.rx(parallel=rp.parallel.Parallel("thread", workers=2))
    got, _ = receive(rx, x, o, noise=0.5)
    rx.close()
    assert np.array_equal(got, want)

def test_frame_cut_short_is_reported_and_the_next_frames_decode(s2):
    p, x, o, data, dfl = s2
    cut = 4*5000
    y = np.concatenate((x[:cut], x[cut + 8:]))
    want, _ = receive(p.rx(), y, o)
    rx = p.rx(parallel=rp.parallel.Parallel("inline"))
    got, st = receive(rx, y, o)
    assert np.array_equal(got, want) and np.array_equal(got, data[dfl[0]:])
    assert any(r["what"] == "cut short" for r in st.control.reports)

def test_a_profile_without_a_region_refuses_parallel():
    with pytest.raises(ConfigError, match="no parallel region"):
        rp.profile.load("dvbs/qpsk_3_4").rx(parallel=rp.parallel.Parallel())

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
    from rpps.parallel.kernel import Batch, Kernel, LoopKernel
    class Picky(Kernel):
        name = "picky"
        def run(self, xp, b):
            b.fail(1, "bad row")
            return b
    plan = rp.parallel.BatchPlan([Picky()])
    b = Batch(np.zeros((3, 4), np.float32), rp.parallel.Form(rp.Data, rp.Category.UNK, rp.dtype.LLR, rp.Rep.CHANNEL, 1, 1, 0, 0), [0, 4, 8], 4)
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

# ---- the front: pieces that were made cheaper keep their results

def test_overlap_save_convolution_equals_np_convolve():
    from rpps.signal.filters.convolve import convolve
    rng = np.random.default_rng(0)
    for nx in (4096, 5000, 100003):
        for nh in (16, 26, 101, 513):
            for cplx in (False, True):
                x = rng.normal(size=nx) + (1j*rng.normal(size=nx) if cplx else 0)
                h = rng.normal(size=nh)
                for mode in ("full", "same", "valid"):
                    a, b = np.convolve(x, h, mode), convolve(x, h, mode)
                    assert a.shape == b.shape and np.allclose(a, b, atol=1e-9), (nx, nh, cplx, mode)

def test_phase_line_phasor_equals_the_exponential():
    from rpps.sync import track
    for length in (1, 7, 1000, 33333):
        for rate in (0.0, 1e-4, -3e-3):
            want = np.exp(-1j*(0.3 + rate*np.arange(length)))
            np.testing.assert_allclose(track.line_phasor(0.3, rate, length), want, atol=1e-12)
    assert len(track.line_phasor(0.0, 0.1, 0)) == 0

def test_power_by_squaring():
    from rpps.sync.phase.mpower import _power
    rng = np.random.default_rng(1)
    u = np.exp(1j*rng.uniform(-np.pi, np.pi, 500))
    for M in (1, 2, 3, 4, 8, 12):
        np.testing.assert_allclose(_power(u, M), u**M, atol=1e-12)

def test_the_matched_filter_gives_symbols_back_at_their_amplitude():
    """The profile's pulse shaping then its matched filter: unit symbols are about unit at their sampling instants, so a
    demapper sees the constellation's scale (they came back a quarter of it, at 4 samples per symbol)"""
    p = rp.profile.load("dvbs2/normal", sync={"timing": {"method": "known", "fuse": False}})
    syms = np.exp(1j*(np.pi/4 + np.pi/2*np.random.default_rng(2).integers(0, 4, 600))).astype(np.complex64)
    matched = p.rx().ops[0][1]
    pulse = p.tx().ops[-1][1]
    y = np.asarray((rp.Data(syms, Fs=1e6, sps=1, flush=True) + rp.signal.sample.up.Zeros(4) + pulse + matched).data)
    medians = [np.median(np.abs(y[k + 400:-400:4])) for k in range(4)]
    assert 0.9 < max(medians) < 1.1

def test_native_word_metric_equals_the_numpy_one(monkeypatch):
    """FrameSync's header-word metric in C++ and in numpy: the same frames found, metrics equal to single precision"""
    from rpps import _accel
    if _accel.native is None:
        pytest.skip("rpps._native is not built")
    p = rp.profile.load("dvbs2/normal", sync=KNOWN)
    fs = next(op for _, op in p.rx().ops if type(op).__name__ == "FrameSync")
    rng = np.random.default_rng(3)
    y = (rng.normal(size=20000) + 1j*rng.normal(size=20000)).astype(np.complex128)
    y[5000:5000 + fs.L] = fs.preamble*0.8
    c = np.convolve(y, np.conj(fs.preamble[::-1]), "valid")
    cs = np.concatenate(([0.0], np.cumsum(y.real**2 + y.imag**2)))
    m = np.abs(c)/np.sqrt(np.maximum(cs[fs.L:fs.L + len(c)] - cs[:len(c)], 1e-30)*fs._ep)
    a = fs._with_words(y, c, m, cs)
    monkeypatch.setattr(_accel, "native", None)
    b = fs._with_words(y, c, m, cs)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    assert (a > 0).sum() > 100

def test_process_executor_runs_a_picklable_worker_and_keeps_order():
    with Scheduler("process", halve, max_batch=3, eager=False, workers=3) as sch:
        out = drain(sch, frames(20, modcods=(1, 2)))
    assert [r.seq for r in out] == list(range(20))
    for r, f in zip(out, frames(20, modcods=(1, 2))):
        np.testing.assert_array_equal(r.data, f.data[:4]*2)

def test_process_executor_brings_a_workers_bug_back():
    with Scheduler("process", _boom, workers=2) as sch:
        with pytest.raises(KeyError):
            sch.push(frames(1)[0])
            sch.flush()

def _boom(batch):
    raise KeyError("bug in a worker process")

def test_fused_matched_filter_gives_the_same_symbols_as_filtering_every_sample():
    """Known timing with the matched filter evaluated only at its instants, against the filter then the pick, for any block cut"""
    from rpps import _accel
    rng = np.random.default_rng(4)
    for name, kw in (("dvbs2/normal", {}), ("ieee802154/2450", {"sps": 4})):
        fused = rp.profile.load(name, sync=KNOWN, **kw)
        apart = rp.profile.load(name, sync={"timing": {"method": "known", "fuse": False}}, **kw)
        x = (rng.normal(size=40000) + 1j*rng.normal(size=40000)).astype(np.complex64)
        def front(p, cuts):
            ops = [op for _, op in p.rx().ops]
            stop = next(i for i, o in enumerate(ops) if "Timing" in str(o)) + 1
            st, outs, i = None, [], 0
            for k, n in enumerate(cuts):
                d = rp.Data(x[i:i + n], Fs=1e6, sps=4, stream=st, flush=k == len(cuts) - 1)
                st, i = d.stream, i + n
                for o in ops[:stop]:
                    d = d + o
                outs.append(np.asarray(d.data))
            return np.concatenate(outs)
        want = front(apart, [40000])
        for cuts in ([40000], [10000]*4, [1, 7, 12345, 40000 - 12353]):
            got = front(fused, cuts)
            assert got.shape == want.shape, (name, cuts)
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)

def test_native_carrier_helpers_equal_the_numpy_ones(monkeypatch):
    from rpps import _accel
    from rpps.sync.phase import mpower
    if _accel.native is None:
        pytest.skip("rpps._native is not built")
    rng = np.random.default_rng(6)
    seg = (rng.normal(size=1500) + 1j*rng.normal(size=1500))
    seg[::37] = 0.0 # zeros stay zero
    for M, phi, rate in ((8, 0.4, 3e-4), (4, -2.0, -1e-3), (2, 0.0, 0.0), (1, 1.0, 5e-3)):
        a = mpower._measure(seg, phi, rate, M)
        r = mpower._turn_back(seg, phi, rate)
        monkeypatch.setattr(_accel, "native", None)
        b = mpower._measure(seg, phi, rate, M)
        q = mpower._turn_back(seg, phi, rate)
        monkeypatch.undo()
        np.testing.assert_allclose(a[0], b[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(a[1], b[1], rtol=1e-9)
        np.testing.assert_allclose(r, q, rtol=1e-5, atol=1e-6)

def test_native_preamble_metric_equals_the_numpy_one():
    from rpps import _accel
    if _accel.native is None:
        pytest.skip("rpps._native is not built")
    rng = np.random.default_rng(7)
    for L in (1, 7, 26, 255):
        y = rng.normal(size=20000) + 1j*rng.normal(size=20000)
        y[3000:3000 + L] *= 4
        pre = np.exp(2j*np.pi*rng.random(L))
        c, m = _accel.native.preamble_metric(y, pre, float(np.sum(np.abs(pre)**2)))
        want_c = np.correlate(y, pre, "valid")
        cs = np.concatenate(([0.0], np.cumsum(np.abs(y)**2)))
        e = cs[L:] - cs[:-L]
        want_m = np.abs(want_c)/np.sqrt(np.maximum(e, 1e-30)*np.sum(np.abs(pre)**2))
        np.testing.assert_allclose(c, want_c, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(m, want_m, rtol=1e-8, atol=1e-9)
