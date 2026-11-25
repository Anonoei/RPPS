"""The parallel runner on DVB-S2 gives the serial runner's output"""
import numpy as np
import pytest

import rpps as rp
from rpps.errors import ConfigError
from rpps.parallel import Frame
from support.parallel import FRAMES, receive

# ---- the parallel runner on DVB-S2: the same output as the serial runner

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

def _devices():
    from support import fakedevice
    from rpps import _accel
    xp, backend = _accel.device()
    return [fakedevice] + ([xp] if backend else [])

@pytest.mark.parametrize("xp", _devices(), ids=lambda m: m.__name__)
def test_on_a_device_the_output_is_the_serial_runners(s2, xp):
    """The region on a device: the pretend one (tests/unit/fakedevice.py, which refuses implicit host/device moves as CuPy
    does), and a real GPU when there is one. Noisy frames, some failing, come out as the serial runner gives them"""
    p, x, o, data, dfl = s2
    want, _ = receive(p.rx(), x, o, noise=0.5)
    rx = p.rx(parallel=rp.parallel.Parallel("thread", workers=2, xp=xp))
    got, _ = receive(rx, x, o, noise=0.5)
    rx.close()
    assert np.array_equal(got, want)

def test_a_device_plan_moves_the_batch_once_each_way_and_around_host_kernels(s2):
    from support import fakedevice
    p, x, o, data, dfl = s2
    w = rp.parallel.PlanWorker(lambda: p._build()[p.region.first:], xp=fakedevice)
    f = Frame(0, np.zeros(33192, np.complex64), {"plscram": {"length": 100}}, tags=(rp.Data, rp.Category.UNK, rp.dtype.SYMBOLS,
                                                                                    rp.Rep.SYMBOLS, 1e6, 1, 0.0, 0.0))
    names = [k["kernel"] for k in w.plan(f).describe(33192)]
    assert names[0] == "to:device" and names[-1] != "to:device"
    for i, n in enumerate(names): # a host kernel is reached only through a download, and the device again through an upload
        k = w.plan(f).kernels[i]
        if k.where == "host":
            prev = [m for m in names[:i] if m.startswith("to:")]
            assert prev and prev[-1] == "to:host", names
    assert names.count("to:host") >= 1 and all(a != b for a, b in zip(names, names[1:]) if a.startswith("to:"))
    assert rp.parallel.plan.placed(w.plan(f).kernels, np) == w.plan(f).kernels # (numpy: nothing to move)

def test_the_pretend_device_catches_an_implicit_move():
    """A kernel that mixes a numpy array into device work fails on the pretend device (so CI sees it without a GPU)"""
    from support import fakedevice
    from rpps.base.kernel import Batch, Kernel, Form
    class Leaky(Kernel):
        name = "leaky"
        def run(self, xp, b):
            b.x = b.x*np.ones(b.x.shape[1], np.float32)[None, :]
            return b
    b = Batch(np.zeros((2, 4), np.float32), Form(rp.Data, rp.Category.UNK, rp.dtype.LLR, rp.Rep.CHANNEL, 1, 1, 0, 0), [0, 4], 4)
    plan = rp.parallel.BatchPlan(rp.parallel.plan.placed([Leaky()], fakedevice))
    with pytest.raises(fakedevice.DeviceError):
        plan.run(fakedevice, b)

def test_a_profile_without_a_region_refuses_parallel():
    with pytest.raises(ConfigError, match="no parallel region"):
        rp.profile.load("dvbs/qpsk_3_4").rx(parallel=rp.parallel.Parallel())
