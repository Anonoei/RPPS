"""Everything here compares the C++ kernels with the Python reference: marked `native`, skipped when they are off or not
built"""
from pathlib import Path

import pytest

from rpps import _accel

HERE = Path(__file__).parent

def pytest_collection_modifyitems(items):
    skip = None if _accel.available() else pytest.mark.skip(reason="rpps._native is not built, or RPPS_NO_NATIVE is set")
    for item in items:
        if HERE in item.path.parents:
            item.add_marker(pytest.mark.native)
            if skip:
                item.add_marker(skip)
