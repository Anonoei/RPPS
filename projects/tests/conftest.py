import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(1234)
