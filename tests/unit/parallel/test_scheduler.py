"""The scheduler and executors"""
import threading
import time
import numpy as np
import pytest

import rpps as rp
from rpps.parallel import Frame, Result, Scheduler, ThreadExecutor, DROPPED
from support.parallel import frames, halve

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

def test_device_is_numpy_when_asked_or_without_a_gpu(monkeypatch):
    from rpps import _accel
    monkeypatch.setenv("RPPS_DEVICE", "none")
    assert _accel.device() == (np, None)
    monkeypatch.setenv("RPPS_DEVICE", "nope")
    with pytest.raises(RuntimeError, match="none, cuda or hip"):
        _accel.device()
    monkeypatch.delenv("RPPS_DEVICE")
    xp, backend = _accel.device()
    assert (xp is np) == (backend is None) and backend in (None, "cuda", "hip")

def test_a_device_does_not_cross_a_fork():
    from support import fakedevice
    with pytest.raises(ValueError, match="inline or thread"):
        rp.parallel.Parallel("process", xp=fakedevice)
