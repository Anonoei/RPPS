"""Convolutional code termination: tail-biting and its names"""
import numpy as np
import pytest

import rpps as rp
from support.signals import bits

# --- convolutional termination --------------------------------------------------------------------------------

def _circular_encode(msg, gens, K):
    """36.212 5.1.3.1: the register starts with the last K-1 message bits"""
    out = []
    n = len(msg)
    for k in range(n):
        window = [msg[(k - i) % n] for i in range(K)] # current input first
        out.append([sum(window[i] & ((g >> (K - 1 - i)) & 1) for i in range(K)) & 1 for g in gens])
    return np.array(out, dtype=bool).reshape(-1)

def test_tailbiting_encoder(rng):
    gens = [0o133, 0o171, 0o165]
    code = rp.code.Conv(gens, 7, termination="tailbiting")
    for n in (7, 40, 64):
        msg = bits(rng, n).astype(int)
        assert np.array_equal(code.encoder.encode(msg), _circular_encode(msg, gens, 7))

@pytest.mark.parametrize("decoder", ["viterbi", "bcjr"])
def test_tailbiting_decodes_noisy_frames(decoder, rng):
    code = rp.code.Conv([0o133, 0o171, 0o165], 7, termination="tailbiting", decoder=decoder)
    good = 0
    for _ in range(20):
        msg = bits(rng, 40)
        cw = code.encoder.encode(msg)
        llr = 2*((1 - 2.0*cw) + rng.normal(0, 0.7, len(cw)))/0.49
        good += np.array_equal(code.decoder.decode(llr), msg)
    assert good >= 19

def test_tailbiting_in_a_pipeline(rng):
    code = rp.code.Conv([0o133, 0o171], 7, termination="tailbiting")
    msg = bits(rng, 48)
    cw = (rp.Data(msg, flush=True) * code).data
    assert len(cw) == 96 # Nothing added
    assert np.array_equal((rp.Data(cw, flush=True) / code).data, msg)

def test_termination_names_and_octal_generators(rng):
    obj = {"base": "conv", "constraint": 7, "generator": ["0o133", "0o171"], "termination": "zero"}
    a = rp.code.serial.init("t", obj)
    b = rp.code.load("conv/k7_133_171/1_2")
    assert np.array_equal(a.encoder.generator, b.encoder.generator)
    assert a.encoder.terminated and a.decoder.terminated # One set of keys: the decoder has the encoder's termination
    msg = bits(rng, 50)
    assert np.array_equal(a.decoder.decode(a.encoder.encode(msg)), msg)
    with pytest.raises(ValueError):
        rp.code.Conv([0o133, 0o171], 7, termination="circular")
