"""Oerder & Meyr and MPower on a GPU (`"device": true`): each window's measurement and its symbols or correction on the
device, the tracking on the host as ever; the host's output to float32 rounding, and a profile decodes the same"""
import numpy as np
import pytest

import rpps as rp
from rpps import _accel
from rpps.sync.phase import MPower
from rpps.sync.time import OerderMeyr

XP, BACKEND = _accel.device()
GPU = pytest.mark.skipif(BACKEND is None, reason="no GPU (rpps._accel.device())")

def _shaped(sps, n=60000, ppm=20.0, seed=0):
    """QPSK through an RRC 0.35 at `sps`, with a clock `ppm` fast, a turning carrier and noise"""
    rng = np.random.default_rng(seed)
    s = ((rng.choice([1, -1], n) + 1j*rng.choice([1, -1], n))/np.sqrt(2)).astype(np.complex64)
    x = np.asarray((rp.Data(s, Fs=1e6/sps, sps=1) + rp.wave.Shape(sps, rp.signal.filters.load("pulse/rrc/0p35"))).data)
    t = np.arange(len(x))*(1 + ppm*1e-6) + 0.37
    x = np.interp(t, np.arange(len(x)), x.real) + 1j*np.interp(t, np.arange(len(x)), x.imag)
    x = x*np.exp(1j*(0.7 + 2*np.pi*2e-5*np.arange(len(x))))
    return (x + (rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))*0.05).astype(np.complex64)

def _run(make, x, sps, block):
    st, stream, out = make(), None, []
    for i in range(0, len(x), block):
        d = rp.Data(x[i:i + block], Fs=1e6, sps=sps, stream=stream, flush=i + block >= len(x))
        stream = d.stream
        out.append(np.asarray((d + st).data))
    return np.concatenate(out), stream.est

def test_without_a_gpu_device_is_the_host(monkeypatch):
    """device=True with no GPU (numpy) runs the host's path: the same output exactly"""
    monkeypatch.setenv("RPPS_DEVICE", "none")
    x = _shaped(4)
    for make in (lambda dev: OerderMeyr(4, device=dev), lambda dev: OerderMeyr(4, device=dev) + MPower(4, np.pi, device=dev)):
        a, _ = _run(lambda: make(False), x, 4, 7777)
        b, _ = _run(lambda: make(True), x, 4, 7777)
        assert np.array_equal(a, b)

@GPU
@pytest.mark.parametrize("sps", [4, 2]) # 2: the doubled window (made on the host) and the 8-tap sinc read
@pytest.mark.parametrize("block", [65536, 7777])
def test_timing_on_the_device_is_the_hosts(sps, block):
    x = _shaped(sps)
    host, eh = _run(lambda: OerderMeyr(sps), x, sps, block)
    dev, ed = _run(lambda: OerderMeyr(sps, device=True), x, sps, block)
    assert host.shape == dev.shape and np.max(np.abs(host - dev)) < 1e-5*np.sqrt(np.mean(np.abs(host)**2))
    assert ed["timing"]["ppm"] == pytest.approx(eh["timing"]["ppm"], abs=1e-3)

@GPU
@pytest.mark.parametrize("block", [65536, 7777])
def test_carrier_on_the_device_is_the_hosts(block):
    syms, _ = _run(lambda: OerderMeyr(4), _shaped(4), 4, 65536)
    host, eh = _run(lambda: MPower(4, np.pi), syms, 1, block)
    dev, ed = _run(lambda: MPower(4, np.pi, device=True), syms, 1, block)
    assert host.shape == dev.shape and np.max(np.abs(host - dev)) < 1e-5*np.sqrt(np.mean(np.abs(host)**2))
    assert ed["carrier"]["frequency"] == pytest.approx(eh["carrier"]["frequency"], abs=1e-9)

@GPU
def test_dvbs2_with_its_front_on_the_device():
    """dvbs2/normal with timing and carrier on the device decodes what the host does"""
    from test_matrack import FRAMES, receive
    from rpps.math.interp import interpolate
    p = rp.profile.load("dvbs2/normal")
    t = rp.profile.table("dvbs2/modcod")
    data = np.random.default_rng(1).integers(0, 2, sum(t.row({**f, "short": 0})["kbch"] - 80 for f in FRAMES)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=FRAMES)
    x = np.concatenate((np.zeros(3000, np.complex64), o.data, np.zeros(400, np.complex64)))
    x = interpolate(x, np.arange(int(len(x)/(1 + 20e-6)) - 20)*(1 + 20e-6))
    rng = np.random.default_rng(2)
    x = (x*np.exp(0.7j) + 0.05*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)
    sync = {"timing": {"method": "oerder_meyr", "device": True}, "carrier": {"method": "mpower", "device": True}}
    got, st = receive(x, o, sync)
    assert np.array_equal(got, data) and abs(st.est["timing"]["ppm"] - 20) < 1
