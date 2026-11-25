"""Constellation: any point set and bit map from config, and the bugs the old two-class design had"""
import numpy as np
import pytest

import rpps as rp
from rpps.symbol import Mapping, mapping as mapping_mod

def test_bits_per_symbol_follows_the_points():
    for M, bps in ((2, 1), (4, 2), (8, 3), (16, 4), (64, 6)):
        c = rp.symbol.Constellation(np.exp(1j*2*np.pi*np.arange(M)/M))
        assert c.bits_per_symbol == bps and c.order == M
    with pytest.raises(rp.errors.ConfigError):
        rp.symbol.Constellation([1, -1, 1j])

def test_instances_do_not_share_state():
    a, b = rp.symbol.Constellation([1, -1]), rp.symbol.Constellation([1, -1])
    assert a.mapping is not b.mapping and a.points is not b.points
    a.mapping.arr[0] = 1
    a.mapping.arr[1] = 0
    assert b.mapping.arr.tolist() == [0, 1]
    assert rp.symbol.ASK.__name__ == "ASK" and rp.symbol.generate("4ASK").points is not rp.symbol.generate("4ASK").points

def test_default_labels_are_the_points_in_order():
    c = rp.symbol.Constellation(np.exp(1j*np.arange(4)))
    assert c.mapping.arr.tolist() == [0, 1, 2, 3]
    bits = np.array([0, 1, 1, 0], bool) # value 1, then 2
    assert np.allclose(c.modulate(bits), c.points[[1, 2]])

def test_first_bit_is_the_most_significant_on_every_host():
    c = rp.symbol.Constellation(np.exp(1j*2*np.pi*np.arange(8)/8))
    assert c.labels()[5].tolist() == [1, 0, 1] and c.labels()[1].tolist() == [0, 0, 1]
    assert np.allclose(c.modulate(np.array([1, 0, 0], bool)), c.points[4])

def test_mapping_of_nothing_is_empty():
    assert len(Mapping(None)) == 0 and len(Mapping([])) == 0

def test_expand_maps_leaves_the_callers_maps_alone():
    pts = np.exp(1j*(np.pi/4 + np.pi/2*np.arange(4)))
    given = [Mapping([0, 1, 3, 2], "Grey")]
    out = mapping_mod.expand_maps(pts, given)
    assert given[0].comment == "Grey Normal" and len(given) == 1 and len(out) == 8
    assert out[0].comment == "R000 Grey Normal" and out[0] is not given[0]

def test_bad_maps_are_refused():
    c = rp.symbol.load("QPSK")
    with pytest.raises(ValueError):
        c.set_mapping([0, 1, 1, 2])
    with pytest.raises(ValueError):
        c.set_mapping([0, 1, 2])

def test_points_are_normalised_unless_told_not_to():
    obj = {"base": "constellation", "points": [{"re": 3, "im": 0}, {"re": -3, "im": 0}]}
    assert np.mean(np.abs(rp.symbol.init("x", obj).points)**2) == pytest.approx(1.0)
    raw = rp.symbol.init("x", {**obj, "normalize": False})
    assert np.allclose(raw.points, [3, -3])

def test_a_config_can_put_points_anywhere_with_any_map():
    """A 4-point constellation of arbitrary shape, points as polar, rectangular and pairs, with a made-up labelling"""
    obj = {"base": "constellation", "comment": "irregular",
           "points": [{"mag": 1, "pha": 0}, {"re": 0.0, "im": 2.0}, [-1.5, 0.0], {"mag": 0.5, "pha": 270}],
           "maps": [{"map": [3, 1, 0, 2], "comment": "odd"}]}
    c = rp.symbol.init("irregular", obj)
    assert c.order == 4 and c.mapping.arr.tolist() == [3, 1, 0, 2]
    pts = c.points*np.sqrt(np.mean(np.abs([1, 2, 1.5, 0.5])**2))
    assert np.allclose(pts, [1, 2j, -1.5, -0.5j])
    msg = np.random.default_rng(0).integers(0, 2, 400).astype(bool)
    llr = c.llr(c.modulate(msg), noise_var=0.01)
    assert np.array_equal(llr < 0, msg)
    # point 0 carries value 3 (bits 11)
    assert np.allclose(c.modulate(np.array([1, 1], bool)), c.points[0])

def test_a_config_without_points_is_an_error():
    with pytest.raises(rp.errors.ConfigError):
        rp.symbol.init("none", {"base": "constellation"})

def test_a_config_file_in_a_user_directory(tmp_path):
    from rpps._config import format_json
    d = tmp_path / "constellation"
    d.mkdir(parents=True)
    (d / "diamond.json").write_text(format_json({"points": [[1, 0], [0, 1], [-1, 0], [0, -1]],
                                                  "maps": [{"map": [0, 1, 3, 2], "comment": "Grey"}]}))
    rp.add_config_path(tmp_path)
    try:
        c = rp.symbol.load("diamond")
        assert c.order == 4 and "diamond" in rp.symbol.ls()["constellation"]
    finally:
        rp._config._EXTRA.remove(tmp_path)

def test_shapes_are_generated_and_definitions_loaded():
    assert rp.symbol.generate("16QAM").order == 16 and rp.symbol.generate("OOK").order == 2 and rp.symbol.generate("8PSK").bits_per_symbol == 3
    assert rp.symbol.load("8PSK").bits_per_symbol == 3 # psk/8psk, a definition
    with pytest.raises(rp.errors.ConfigError, match="generate"):
        rp.symbol.load("16QAM")
    with pytest.raises(rp.errors.ConfigError):
        rp.symbol.load("nonsense")
    assert rp.symbol.load_or_generate("16QAM").order == 16 # Names from a detector, either kind

def test_bpsk_is_on_the_real_axis_however_made():
    for c in (rp.symbol.PSK(2), rp.symbol.generate("BPSK"), rp.symbol.load("psk/bpsk")):
        assert np.allclose(sorted(c.points.real), [-1, 1]) and np.allclose(c.points.imag, 0)

def test_rings_and_rotate_build_points(tmp_path):
    from rpps._config import format_json
    d = tmp_path / "constellation" / "x"
    d.mkdir(parents=True)
    (d / "ring8.json").write_text(format_json({"comment": "8PSK as a ring", "rings": [{"n": 8}]}))
    (d / "ring8_axes.json").write_text(format_json({"extends": "x/ring8", "comment": "turned", "rotate": -22.5}))
    (d / "two.json").write_text(format_json({"comment": "4+12", "params": {"gamma": 2.0},
                                              "rings": [{"n": 4, "pha": 45}, {"n": 12, "mag": {"param": "gamma"}, "pha": 15}],
                                              "normalize": False}))
    rp.add_config_path(tmp_path)
    try:
        assert np.allclose(rp.symbol.load("x/ring8").points, rp.symbol.load("psk/8psk").points)
        assert np.allclose(np.angle(rp.symbol.load("x/ring8_axes").points[:2], deg=True), [0, 45])
        two = rp.symbol.load("x/two", gamma=3.0)
        assert np.allclose(np.abs(two.points), [1]*4 + [3]*12)
        assert np.allclose(np.abs(rp.symbol.load("x/two").points), [1]*4 + [2]*12)
    finally:
        rp._config._EXTRA.remove(tmp_path)
