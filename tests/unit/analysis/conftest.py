"""Fixtures the analysis tests share"""
import numpy as np
import pytest

import rpps as rp

@pytest.fixture(scope="module")
def hdlc():
    rng = np.random.default_rng(1)
    fr = rp.link.load("frame/hdlc")
    frames = [bytes(rng.integers(0, 256, int(rng.integers(20, 200))).astype(np.uint8)) for _ in range(120)]
    return np.asarray(fr.encoder.encode(frames)).astype(np.uint8)[:100000]
