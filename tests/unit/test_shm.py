"""Channels in shared memory (rpps.source.shm): ShmSink publishes blocks at their positions as pysdrlib `shm` streams, and
another thread (or process) receives one with rp.source.Device("shm@rpps.<name>"): rate, centre, positions and gaps"""
import threading
import time

import numpy as np
import pytest

pytest.importorskip("pysdrlib.shm")
import rpps as rp
from rpps.source import ShmSink

@pytest.fixture(autouse=True)
def shm_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSDRLIB_SHM_DIR", str(tmp_path))
    return str(tmp_path)

def tone(n, f=0.01):
    return np.exp(2j*np.pi*f*np.arange(n)).astype(np.complex64)

def receive(name, count=1000):
    """Receive `name` in a thread until its publisher stops; returns (thread, {"blocks", "src"}). Receiving starts with
    what is published after it opens, so give it a moment first (`started`)"""
    out = {"blocks": []}
    def run():
        src = rp.source.Device(f"shm@rpps.{name}")
        out["src"] = src
        for b in src(count):
            out["blocks"].append(b)
    th = threading.Thread(target=run)
    th.start()
    time.sleep(0.3)
    return th, out

def joined(out):
    return np.concatenate([np.asarray(b.data) for b in out["blocks"]])

def test_round_trip_any_cut(count=1000):
    x = tone(9000)
    with ShmSink("a", labels={"a": "a tone"}, bw=8e4) as sink:
        st = rp.Data(x[:0], Fs=1e5, cf=1e6) # (announces the stream: receivers can open it)
        st >> sink
        th, out = receive("a", count)
        i = 0
        for n in (1, 999, 3000, 5000):
            rp.Data(x[i:i + n], Fs=1e5, cf=1e6, stream=st.stream) >> sink
            i += n
        time.sleep(0.2)
        import pysdrlib
        found = [d for d in pysdrlib.find() if getattr(d, "serial", None) == "rpps.a"]
        assert found and found[0].extra["info"]["label"] == "a tone" and found[0].extra["info"]["bw"] == 8e4
    th.join(5)
    assert np.array_equal(joined(out), x)
    bl = [b for b in out["blocks"] if len(b)]
    assert [b.meta.pos for b in bl] == list(np.cumsum([0] + [len(b) for b in bl])[:-1])
    assert bl[0].Fs == 1e5 and bl[0].cf == 1e6 and out["src"].stream.dropped == 0
    assert out["blocks"][-1].flush and out["src"].lost # (the publisher closed: the blocks end)

def test_the_end_of_a_stream_short_of_a_block_comes_out():
    test_round_trip_any_cut(777)

def test_a_gap_upstream_is_a_gap_here():
    x = tone(4000)
    with ShmSink("g") as sink:
        a = rp.Data(x[:0], Fs=1e5)
        a >> sink
        th, out = receive("g")
        rp.Data(x[:2000], Fs=1e5, stream=a.stream) >> sink
        b = rp.Data(x[2000:], Fs=1e5, stream=a.stream)
        b.meta.pos += 500 # 500 samples lost before it
        b >> sink
        time.sleep(0.2)
    th.join(5)
    pos = [(b.meta.pos, len(b)) for b in out["blocks"] if len(b)]
    assert pos[0][0] == 0 and any(p == 2500 for p, _ in pos) and sum(n for _, n in pos) == 4000
    assert out["src"].stream.dropped == 500

def test_a_retune_starts_the_stream_over():
    with ShmSink("r") as sink:
        a = rp.Data(tone(0), Fs=1e5, cf=1e6)
        a >> sink
        th, out = receive("r")
        rp.Data(tone(3000), Fs=1e5, cf=1e6, stream=a.stream) >> sink
        rp.Data(tone(2000), Fs=1e5, cf=2e6, stream=a.stream) >> sink # a new centre: a new generation
        time.sleep(0.3)
    th.join(5)
    by_cf = {}
    for b in out["blocks"]:
        by_cf[b.cf] = by_cf.get(b.cf, 0) + len(b)
    assert by_cf.get(1e6) == 3000 and by_cf.get(2e6) == 2000
    after = [b for b in out["blocks"] if b.cf == 2e6 and len(b)]
    assert after[0].meta.pos == 0 # (the stream started over at the change)

def test_channelized_columns_each_a_channel():
    """A channelizer's output, one stream per column, each at its own centre; a tone in channel 1 comes out of stream 1"""
    import pysdrlib
    M, Fs, cf = 4, 4e5, 1e6
    x = tone(40000, f=1/M + 0.002)
    names = [f"c{k}" for k in range(M)]
    ch = rp.signal.filters.Channelizer(M)
    with ShmSink(names) as sink:
        first = rp.Data(x[:4000], Fs=Fs, cf=cf)
        (first + ch) >> sink
        info = {d.serial: d for d in pysdrlib.find() if str(getattr(d, "serial", "")).startswith("rpps.c")}
        man = {n: pysdrlib.open(f"shm@rpps.{n}") for n in names}
        try:
            assert [man[n].get_rx_freq() for n in names] == [cf, cf + 1e5, cf - 2e5, cf - 1e5]
            assert all(man[n].get_rx_sample_rate() == Fs/M for n in names)
        finally:
            for d in man.values():
                d.close()
        assert info["rpps.c1"].extra["info"]["parent"] == {"Fs": Fs, "cf": cf}
        runs = [receive(n) for n in names]
        for i in range(4000, len(x), 4000):
            (rp.Data(x[i:i + 4000], Fs=Fs, cf=cf, stream=first.stream) + ch) >> sink
        time.sleep(0.3)
    power = []
    for th, out in runs:
        th.join(5)
        power.append(np.mean(np.abs(joined(out))**2))
    assert int(np.argmax(power)) == 1 and power[1] > 100*max(power[k] for k in (0, 2, 3))

def test_a_block_must_match_the_names():
    with ShmSink(["a", "b"]) as sink:
        with pytest.raises(ValueError, match="2 name"):
            rp.Data(tone(10), Fs=1e5) >> sink
