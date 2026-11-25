"""HDLC frames from a stream of bits"""
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
