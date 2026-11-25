"""Fixtures the parallel tests share"""
import numpy as np
import pytest

import rpps as rp
from support.dvbs2 import KNOWN
from support.parallel import FRAMES

@pytest.fixture(scope="module")
def s2():
    p = rp.profile.load("dvbs2/normal", sync=KNOWN)
    t = rp.profile.table("dvbs2/modcod")
    dfl = [t.row({**f, "short": 0})["kbch"] - 80 for f in FRAMES]
    data = np.random.default_rng(4).integers(0, 2, sum(dfl)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=FRAMES)
    return p, np.concatenate((o.data, np.zeros(200, np.complex64))), o, data, dfl
