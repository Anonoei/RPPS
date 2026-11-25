import os
os.environ.setdefault("MPLBACKEND", "Agg")
# Each xdist worker is one process: BLAS and RPPS's own pools stay at one thread, so `-n auto` doesn't oversubscribe
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RPPS_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pytest

import rpps as rp
from rpps import _accel, _config

@pytest.hookimpl(optionalhook=True) # only with pytest-xdist; the wheel tests run without it
def pytest_xdist_auto_num_workers(config):
    """`-n auto`: half the cores, so the machine stays usable while the suite runs; every core on CI (where `CI` is set,
    as GitHub does). RPPS_TEST_WORKERS chooses"""
    n = os.environ.get("RPPS_TEST_WORKERS")
    cores = os.cpu_count() or 2
    return int(n) if n else cores if os.environ.get("CI") else max(1, cores//2)

@pytest.fixture
def rng():
    return np.random.default_rng(1234)

@pytest.fixture(autouse=True)
def _process_state():
    """A test leaves the config layers and the kernel switch as it found them (restored here, and the test fails)"""
    extra, native = list(_config._EXTRA), _accel.native
    yield
    left = [name for name, same in (("rpps._config._EXTRA", _config._EXTRA == extra), ("rpps._accel.native", _accel.native is native))
            if not same]
    _config._EXTRA[:], _accel.native = extra, native
    assert not left, f"the test left {' and '.join(left)} changed"

@pytest.fixture
def config_layer(_process_state):
    """config_layer(path): search `path` before the shipped definitions, for this test only"""
    extra = list(_config._EXTRA)
    yield rp.add_config_path
    _config._EXTRA[:] = extra
