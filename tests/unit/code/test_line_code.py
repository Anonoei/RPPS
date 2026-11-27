"""Line codes"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from support.signals import hard, as_data_bits, as_llr

# ---- line codes

@pytest.mark.parametrize("name", rp.code.line.NAMES)
def test_line_code_round_trip_hard_and_soft(name, rng):
    b = rng.integers(0, 2, 300).astype(bool)
    L = rp.code.Line(name)
    y = rp.Data(b) * L
    assert y.REP == Rep.CHANNEL and len(y) == (2 if L.pairs else 1)*len(b)
    assert np.array_equal(hard(y / L), b)
    s = as_llr(y) / L
    assert s.DT == rp.dtype.LLR and s.REP == Rep.DATA and np.array_equal(hard(s), b)

@pytest.mark.parametrize("name", rp.code.line.NAMES)
def test_line_code_any_block_cut(name, rng):
    b = rng.integers(0, 2, 200).astype(bool)
    L = rp.code.Line(name)
    whole = hard(rp.Data(b) * L)
    for cuts in ([7, 1, 192], [3]*66 + [2]):
        outs, ctx, i = [], None, 0
        for c in cuts:
            d = rp.Data(b[i:i + c]) if ctx is None else rp.Data(b[i:i + c], ctx=ctx)
            o = d * L
            ctx, i = o.ctx, i + c
            outs.append(hard(o))
        assert np.array_equal(np.concatenate(outs), whole)
    coded = whole
    outs, ctx, i = [], None, 0
    for c in [5, 1, len(coded) - 6]:
        d = as_data_bits(coded[i:i + c], Rep.CHANNEL) if ctx is None else as_data_bits(coded[i:i + c], Rep.CHANNEL, ctx=ctx)
        o = d / L
        ctx, i = o.ctx, i + c
        outs.append(hard(o))
    assert np.array_equal(np.concatenate(outs), b)

def test_line_conventions():
    L = rp.code.Line("nrzi")
    assert np.array_equal(hard(rp.Data(np.array([1, 1, 0, 1, 0, 0], bool)) * L), [0, 0, 1, 1, 0, 1]) # 0: a transition
    L = rp.code.Line("nrzm")
    assert np.array_equal(hard(rp.Data(np.array([1, 1, 0, 1, 0, 0], bool)) * L), [1, 0, 0, 1, 1, 1]) # 1: a transition
    assert np.array_equal(hard(rp.Data(np.array([0, 1], bool)) * rp.code.Line("manchester")), [1, 0, 0, 1]) # IEEE: 0 -> 10
    assert np.array_equal(hard(rp.Data(np.array([0, 1], bool)) * rp.code.Line("thomas")), [0, 1, 1, 0])
    assert np.array_equal(hard(rp.Data(np.array([0, 1], bool)) * rp.code.Line("invert")), [1, 0])
    with pytest.raises(ValueError):
        rp.code.Line("ami")
    m = rp.code.Line("manchester")
    out = as_data_bits([1, 1, 0, 0, 1, 0], Rep.CHANNEL) / m
    assert list(out.failed) == [True, True, False] # 11 and 00 are not Manchester pairs: those bits failed
    soft = as_llr(as_data_bits([1, 1, 0, 0, 1, 0], Rep.CHANNEL)) / m
    assert list(soft.failed) == [True, True, False]

@pytest.mark.parametrize("name", ["biphase_m", "biphase_s", "miller"])
def test_irig_line_codes(name, rng):
    import rpps as rp
    x = rng.integers(0, 2, 400).astype(bool)
    line = rp.code.Line(name)
    ctx, parts = None, []
    for a, b in ((0, 3), (3, 3), (3, 250), (250, 400)): # any block cut
        o = (rp.Data(x[a:b]) if ctx is None else rp.Data(x[a:b], ctx=ctx)) * line
        ctx = o.ctx
        parts.append(np.asarray(o.data, bool))
    y = np.concatenate(parts)
    if name.startswith("biphase"):
        assert all(y[2*i - 1] != y[2*i] for i in range(1, 400)) # a transition at every bit start
    else: # Miller: never more than two bits without a transition
        runs = np.diff(np.flatnonzero(np.diff(y.astype(int)) != 0))
        assert runs.max() <= 4
    assert np.array_equal(np.asarray((rp.Data(y) / line).data, bool), x)
    assert np.array_equal(np.asarray((rp.Data(~y) / line).data, bool), x) # an inverted line decodes the same

@pytest.mark.parametrize("name", ["biphase_m", "biphase_s", "miller"])
def test_two_level_codes_without_a_loop_are_the_state_machine(name, rng):
    """The level as a running parity of the transitions is the state machine stepped a bit at a time"""
    from rpps.code.line import MID
    def machine(x, level=False, prev=True):
        y = []
        for b in x:
            if name != "miller" or (not b and not prev):
                level = not level
            y.append(level)
            if bool(b) == MID[name]:
                level = not level
            y.append(level)
            prev = bool(b)
        return np.array(y, bool)
    b = rng.integers(0, 2, 3001).astype(bool)
    L = rp.code.Line(name)
    outs, ctx, i = [], None, 0
    for c in [1, 0, 2, 500, 2498]:
        d = rp.Data(b[i:i + c]) if ctx is None else rp.Data(b[i:i + c], ctx=ctx)
        o = d * L
        ctx, i = o.ctx, i + c
        outs.append(hard(o))
    assert np.array_equal(np.concatenate(outs), machine(b))
