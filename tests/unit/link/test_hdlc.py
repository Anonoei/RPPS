"""HDLC framing: round trips, a damaged check sequence, and streaming across any block cut"""
import numpy as np
import pytest

import rpps as rp
from support.signals import hard, as_data_bits

@pytest.mark.parametrize("name", ["hdlc", "hdlc_nrzi"])
def test_hdlc_streams_frames_across_any_block_cut(name, rng):
    fr = rp.link.load(f"frame/{name}")
    pk = [bytes(rng.integers(0, 256, n).astype(np.uint8)) for n in (10, 33, 100, 12)]
    bits, ctx = [], None
    for p in pk:
        d = rp.Data(p) if ctx is None else rp.Data(p, ctx=ctx)
        o = d * fr
        ctx = o.ctx
        bits.append(np.asarray(o.data))
    stream = np.concatenate([rng.integers(0, 2, 37).astype(bool)] + bits + [rng.integers(0, 2, 51).astype(bool)])
    ref = np.unpackbits(np.frombuffer(b"".join(pk), np.uint8)).astype(bool)
    for cuts in ([len(stream)], [5]*(len(stream)//5) + [len(stream) % 5], [1000, 3, len(stream) - 1003]):
        outs, ctx, i = [], None, 0
        for k, c in enumerate(cuts):
            seg = stream[i:i + c]
            d = as_data_bits(seg, flush=k == len(cuts) - 1) if ctx is None else as_data_bits(seg, ctx=ctx, flush=k == len(cuts) - 1)
            o = d / fr
            ctx, i = o.ctx, i + c
            outs.append(hard(o))
        assert np.array_equal(np.concatenate(outs), ref), cuts[:3]

HDLC = [n for n in rp.link.ls()["frame"] if isinstance(rp.link.load(f"frame/{n}").encoder, rp.link.hdlc)] # (not byte stuffing)

@pytest.mark.parametrize("name", HDLC)
def test_hdlc_round_trip(name, rng):
    h = rp.link.load(f"frame/{name}").encoder
    frames = [bytes(rng.integers(0, 256, n).astype(np.uint8)) for n in (max(10, h.min_bytes), 33, 100)] # (AX.25: 17 at least)
    frames.append(b"\xff"*max(12, h.min_bytes)) # Runs of ones need bit stuffing
    got, aborts = h.deframe(h.encode(frames))
    assert aborts == 0
    assert [p for p, ok in got] == frames
    assert all(ok for p, ok in got)

@pytest.mark.parametrize("name", HDLC)
def test_hdlc_bad_fcs_flagged(name, rng):
    h = rp.link.load(f"frame/{name}").encoder
    frames = [bytes(rng.integers(0, 256, 40).astype(np.uint8)) for _ in range(3)]
    bits = h.encode(frames).copy()
    mid = len(bits)//2
    bits[mid] ^= True
    out, status = h.decode(bits, status=True)
    got, aborts = h.deframe(bits)
    assert list(status) == [0 if ok else -1 for _, ok in got]
    assert sum(ok for _, ok in got) < len(frames) and ((status < 0).sum() + aborts) >= 1
