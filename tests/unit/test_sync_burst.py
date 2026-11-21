"""rp.sync.Burst: a stream cut into its bursts, each a block of its own stream at its place in the source"""
import numpy as np
import pytest

import rpps as rp

FS = 4e6

def batch_spans(x, fs, factor=6.0, win=100, min_us=0.0, max_us=None):
    """The rule the projects used on a whole capture (ble_adv, wifi_dsss, wifi_ofdm `bursts`), as it was"""
    p = np.abs(x)**2
    e = p[:len(p)//win*win].reshape(-1, win).mean(1)
    on = e > factor*np.percentile(e, 20)
    d = np.diff(np.r_[0, on.astype(int), 0])
    st = np.where(d == 1)[0]*win; en = np.where(d == -1)[0]*win
    return [(s, e_) for s, e_ in zip(st, en) if (e_ - s)/fs*1e6 >= min_us and (max_us is None or (e_ - s)/fs*1e6 <= max_us)]

def capture(rng, spans, n=200_000, amp=1.0, noise=0.05):
    """Noise with tone bursts at the given (start, end) samples"""
    x = noise*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
    for s, e in spans:
        x[s:e] += amp*np.exp(2j*np.pi*0.01*np.arange(e - s))
    return x.astype(np.complex64)

SENT = [(1000, 5000), (20_000, 20_300), (50_050, 90_000), (120_000, 121_000), (199_000, 200_000)]

def spans(bursts):
    return [(b.meta.pos, b.meta.pos + len(b)) for b in bursts]

@pytest.mark.parametrize("win,min_us,max_us", [(100, 0.0, None), (8, 60.0, 3000.0), (100, 150.0, None)])
def test_one_capture_gives_the_batch_spans(rng, win, min_us, max_us):
    x = capture(rng, SENT)
    got = rp.Data(x, Fs=FS, flush=True) + rp.sync.Burst(win=win, min_s=min_us*1e-6, max_s=None if max_us is None else max_us*1e-6)
    assert spans(got) == batch_spans(x, FS, win=win, min_us=min_us, max_us=max_us)

@pytest.mark.parametrize("block", [777, 4096, 65536])
def test_any_block_cut_gives_the_same_bursts(rng, block):
    x = capture(rng, SENT)
    whole = rp.Data(x, Fs=FS, flush=True) + rp.sync.Burst()
    cut = list(rp.source.Memory(x, Fs=FS)(block) + rp.sync.Burst())
    assert spans(cut) == spans(whole)
    for a, b in zip(cut, whole):
        assert np.array_equal(a.data, b.data) and np.array_equal(a.data, x[a.meta.pos:a.meta.pos + len(a)])

def test_limits_and_estimates(rng):
    x = capture(rng, SENT)
    data = rp.Data(x, Fs=FS, flush=True)
    got = data + rp.sync.Burst(min_s=100e-6, max_s=5e-3) # 300 samples = 75 us is short, 39950 = 10 ms long
    assert spans(got) == [(1000, 5000), (120_000, 121_000), (199_000, 200_000)]
    est = data.stream.est["burst"]
    assert (est["bursts"], est["short"], est["long"]) == (3, 1, 1) and est["floor"] > 0

def test_each_burst_is_a_stream_of_its_own(rng):
    """What follows a burst starts over on every burst: its state is the burst's, not the source's"""
    x = capture(rng, SENT)
    shift = rp.signal.filters.ShiftFreq(1e3)
    out = rp.Data(x, Fs=FS, flush=True, t0=100.0) + (rp.sync.Burst() + shift)
    assert isinstance(out, list) and len(out) == len(SENT)
    streams = {id(b.stream) for b in out}
    assert len(streams) == len(out)
    for b, (s, _) in zip(out, batch_spans(x, FS)): # (edges to a window: 100 samples)
        assert b.meta.pos == s and b.meta.time_of() == pytest.approx(100.0 + s/FS)
        assert b.state(shift) is not None

def test_blocks_run_what_follows_on_every_burst(rng):
    x = capture(rng, SENT)
    shift = rp.signal.filters.ShiftFreq(1e3)
    out = list(rp.source.Memory(x, Fs=FS)(8192) + rp.sync.Burst() + shift)
    assert spans(out) == batch_spans(x, FS)

def test_quiet_stream_has_no_bursts(rng):
    x = (0.05*(rng.standard_normal(50_000) + 1j*rng.standard_normal(50_000))).astype(np.complex64)
    assert rp.Data(x, Fs=FS, flush=True) + rp.sync.Burst(factor=6.0) == []

def test_takes_samples_only():
    with pytest.raises(rp.errors.StreamError):
        rp.Data(np.zeros(16, bool)) + rp.sync.Burst()
