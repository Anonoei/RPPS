"""The soft-symbol cache of atsc_decode / atsc_diag finds its metadata file whatever the cache is called (a name without "_soft" used to read the symbols back as the metadata)"""
import pathlib
import sys

import pytest

pytest.importorskip("pysdrlib")
pytest.importorskip("scipy")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401
import atsc_decode  # noqa: E402

@pytest.mark.parametrize("soft,meta", [("a_soft.npy", "a_meta.npy"), ("data/ch35_soft.npy", "data/ch35_meta.npy"), ("ch35.npy", "ch35_meta.npy"), ("ch35", "ch35_meta.npy"),
                                       ("x_soft", "x_meta.npy")])
def test_meta_path(soft, meta):
    m = atsc_decode.meta_path(soft)
    assert m == meta and m != soft
