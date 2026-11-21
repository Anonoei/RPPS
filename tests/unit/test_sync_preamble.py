"""rp.sync.Preamble: a burst's preamble found, by a known waveform or by its repetition, and its carrier offset"""
import numpy as np
import pytest

import rpps as rp

FS = 20e6

def ofdm_symbol(rng):
    X = np.exp(1j*np.pi/2*rng.integers(0, 4, 64))
    X[0] = 0
    return np.fft.ifft(X)*8

def frame(rng):
    """802.11a-shaped: an STF (ten 16-sample periods), a 32-sample guard, two copies of a 64-sample LTF symbol, data.
    Returns (samples, the LTF symbol, where the first LTF copy starts)"""
    period = ofdm_symbol(rng)[:16]
    stf = np.tile(period, 10)
    ltf = ofdm_symbol(rng)
    data = np.concatenate([ofdm_symbol(rng) for _ in range(6)])
    x = np.concatenate((stf, ltf[-32:], ltf, ltf, data))
    return x, np.concatenate((ltf, ltf)), 160 + 32 # (the template: both LTF copies, so the first is the one found)

def place(rng, x, at, n, cfo, noise=0.05):
    y = noise*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
    y[at:at + len(x)] += x
    return (y*np.exp(2j*np.pi*cfo/FS*np.arange(n))).astype(np.complex64)

@pytest.mark.parametrize("block", [300, 4096, 100_000])
def test_known_template_finds_the_ltf_and_the_offset(rng, block):
    x, ltf, k = frame(rng)
    y = place(rng, x, 1234, 6000, cfo=40e3)
    pre = rp.sync.Preamble(ltf, repeat=64, window=64, offset=-k, correct=True)
    out = list(rp.source.Memory(y, Fs=FS)(block) + pre)
    est = out[-1].stream.est["preamble"]
    assert est["found"] and est["pos"] == 1234 + k and est["metric"] > 0.85 # (40 kHz turns the template 1.6 rad: sinc loss)
    assert est["cfo"] == pytest.approx(40e3, abs=1.5e3) # (of a 312.5 kHz subcarrier spacing)
    got = [b for b in out if len(b)]
    assert got[0].meta.pos == 1234 # The output starts at the STF, where the frame starts
    z = np.concatenate([np.asarray(b.data) for b in got])
    assert len(z) == 6000 - 1234
    # The offset taken off: the LTF's two halves at one phase against the template (40 kHz uncorrected turns 0.4 rad
    # between them), and its second copy the first
    seg, t = z[k:k + 64], ltf[:64]
    turn = np.angle(np.vdot(t[32:], seg[32:])*np.conj(np.vdot(t[:32], seg[:32])))
    assert abs(turn) < 0.05 and np.allclose(z[k + 64:k + 128], seg, atol=0.3)

def test_repetition_finds_the_stf_plateau(rng):
    x, _, _ = frame(rng)
    y = place(rng, x, 2000, 6000, cfo=-120e3)
    out = rp.Data(y, Fs=FS, flush=True) + rp.sync.Preamble(repeat=16, window=128, gate=0.8)
    est = out.stream.est["preamble"]
    assert est["found"] and 1990 <= est["pos"] <= 2010 and out.meta.pos == est["pos"]
    assert est["cfo"] == pytest.approx(-120e3, abs=2e3)

def test_noise_has_no_preamble(rng):
    _, ltf, _ = frame(rng)
    y = place(rng, np.zeros(0), 0, 5000, 0.0, noise=1.0)
    out = rp.Data(y, Fs=FS, flush=True) + rp.sync.Preamble(ltf, gate=0.6)
    assert len(out) == 0 and out.stream.est["preamble"] == {"found": False}

def test_search_limits_the_wait(rng):
    x, ltf, k = frame(rng)
    y = place(rng, x, 100, 50_000, 0.0)
    pre = rp.sync.Preamble(ltf, search=1000)
    out = list(rp.source.Memory(y, Fs=FS)(2048) + pre)
    assert sum(len(b) for b in out[:2]) > 0 # Decided once 1000 starts (and a template) were in: not at the end
    assert out[-1].stream.est["preamble"]["pos"] == 100 + k

def test_after_burst_each_burst_finds_its_own(rng):
    a, ltf, k = frame(rng)
    y = place(rng, a, 3000, 60_000, 25e3)
    y[30_000:30_000 + len(a)] += (a*np.exp(2j*np.pi*-60e3/FS*np.arange(len(a)))).astype(np.complex64)
    pre = rp.sync.Preamble(ltf, repeat=64, window=64, offset=-k)
    out = list(rp.source.Memory(y, Fs=FS)(8192) + rp.sync.Burst(win=16) + pre)
    found = [b.stream.est["preamble"] for b in out]
    assert [f["pos"] for f in found] == [3000 + k, 30_000 + k]
    assert [round(f["cfo"]/1e3) for f in found] == [25, -60]
    assert [b.meta.pos for b in out] == [3000, 30_000]

def test_needs_something_to_find():
    with pytest.raises(ValueError):
        rp.sync.Preamble()
