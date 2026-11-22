"""Channels in shared memory (rpps.source.shm): a sink publishes blocks at their positions on pysdrlib's ring, a reader in
another thread (or process) gets them back with their rate, centre, positions and gaps"""
import threading
import time

import numpy as np
import pytest

pytest.importorskip("pysdrlib.remote.shm")
import rpps as rp
from rpps.source import ShmSink, Shm, read_index, manifest

@pytest.fixture
def d(tmp_path):
    return str(tmp_path)

def tone(n, f=0.01, start=0):
    return np.exp(2j*np.pi*f*np.arange(start, start + n)).astype(np.complex64)

def blocks(x, cuts, Fs=1e5, cf=1e6):
    st, i = None, 0
    for n in cuts:
        b = rp.Data(x[i:i + n], Fs=Fs, cf=cf, stream=st)
        st, i = b.stream, i + n
        yield b

def read_all(name, d, count=1000, timeout=0.3, backlog=1 << 20):
    out = {"blocks": []}
    def run():
        src = Shm(name, backlog=backlog, timeout=timeout, directory=d)
        out["src"] = src
        for b in src(count):
            out["blocks"].append(b)
    th = threading.Thread(target=run)
    th.start()
    return th, out

def test_round_trip_any_cut(d):
    x = tone(9000)
    with ShmSink("a", directory=d) as sink:
        for b in blocks(x, [1, 999, 3000, 5000]):
            b >> sink
        m = manifest("a", d)
        assert m["Fs"] == 1e5 and m["cf"] == 1e6 and m["gen"] == 0 and m["dtype"] == "cf32" and read_index(d)["a"]["gen"] == 0
        th, out = read_all("a", d, count=777)
        th.join()
    got = np.concatenate([np.asarray(b.data) for b in out["blocks"]])
    assert np.array_equal(got, x)
    assert [b.meta.pos for b in out["blocks"]] == list(np.cumsum([0] + [len(b) for b in out["blocks"]])[:-1])
    assert out["blocks"][0].Fs == 1e5 and out["blocks"][0].cf == 1e6 and out["src"].stream.dropped == 0
    assert manifest("a", d) is None and "a" not in read_index(d) # (closed: gone)

def test_a_gap_upstream_is_a_gap_here(d):
    x = tone(6000)
    with ShmSink("g", directory=d) as sink:
        a = rp.Data(x[:2000], Fs=1e5)
        a >> sink
        b = rp.Data(x[2000:4000], Fs=1e5, stream=a.stream)
        b.meta.pos += 500 # 500 samples lost before it
        b >> sink
        th, out = read_all("g", d)
        th.join()
    bl = out["blocks"]
    pos = [(b.meta.pos, len(b)) for b in bl if len(b)]
    assert pos[0][0] == 0 and any(p == 2500 for p, _ in pos) and sum(n for _, n in pos) == 4000
    assert out["src"].stream.dropped == 500

def test_a_reader_that_falls_a_ring_behind_counts_what_it_missed(d):
    with ShmSink("lap", capacity=1 << 18, directory=d) as sink:
        st = None
        b = rp.Data(tone(1000), Fs=1e5)
        b >> sink
        src = Shm("lap", backlog=1000, timeout=0.2, directory=d)
        it = iter(src(500))
        first = next(it)
        for k in range(3): # 3 x 2^18 more than it has read
            c = rp.Data(tone(1 << 18), Fs=1e5, stream=b.stream)
            c >> sink
        sink.close()
        rest = list(it)
    assert first.meta.pos == 0 and src.stream.dropped > 0
    assert rest[-1].meta.pos + len(rest[-1]) == 1000 + 3*(1 << 18) - (0) # positions stay true to the publisher's

def test_a_retune_starts_the_reader_over(d):
    with ShmSink("r", directory=d) as sink:
        rp.Data(tone(3000), Fs=1e5, cf=1e6) >> sink
        th, out = read_all("r", d, timeout=1.0)
        time.sleep(0.2)
        rp.Data(tone(2000), Fs=1e5, cf=2e6) >> sink # a new centre: a new ring, gen 1
        assert manifest("r", d)["gen"] == 1
        time.sleep(0.3)
    th.join()
    cfs = sorted({b.cf for b in out["blocks"] if len(b)})
    assert cfs == [1e6, 2e6]
    assert sum(len(b) for b in out["blocks"] if b.cf == 2e6) == 2000

def test_channelized_columns_each_a_channel(d):
    """A channelizer's output, one ring per column, each at its own centre; a tone in channel 1 comes out of ring 1"""
    M, Fs, cf = 4, 4e5, 1e6
    x = tone(40000, f=1/M + 0.002)
    names = [f"c{k}" for k in range(M)]
    with ShmSink(names, directory=d, labels={"c1": "the tone"}, bw=8e4) as sink:
        for b in blocks(x, [10000]*4, Fs=Fs, cf=cf):
            (b + rp.signal.filters.Channelizer(M)) >> sink
        ms = [manifest(n, d) for n in names]
        assert [m["cf"] for m in ms] == [cf, cf + 1e5, cf - 2e5, cf - 1e5]
        assert all(m["Fs"] == Fs/M and m["parent"] == {"Fs": Fs, "cf": cf} for m in ms) and ms[1]["label"] == "the tone"
        assert all(m["bw"] == 8e4 for m in ms) and manifest.__module__ == "rpps.source.shm"
        power = []
        for n in names:
            th, out = read_all(n, d, timeout=0.2)
            th.join()
            power.append(np.mean(np.abs(np.concatenate([np.asarray(b.data) for b in out["blocks"]]))**2))
    assert int(np.argmax(power)) == 1 and power[1] > 100*max(power[k] for k in (0, 2, 3))

def test_names_are_checked_and_a_missing_channel_times_out(d):
    with pytest.raises(ValueError):
        ShmSink("../x", directory=d)
    with pytest.raises(FileNotFoundError):
        Shm("nobody", timeout=0.05, directory=d)
