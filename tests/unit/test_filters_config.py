import json
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.errors import ConfigError

def test_listing_has_fir_shaping_and_window_presets():
    assert set(rp.signal.filters.ls()) >= {"fir/taps", "fir/windowed", "pulse/rrc", "window/kaiser"}
    assert "hann" not in sum(rp.signal.filters.ls().values(), []) # Built-in windows are listed apart
    assert "hann" in rp.signal.filters.Window.s()

def test_explicit_taps_filter_a_stream():
    f = rp.signal.filters.load("fir", "taps/moving_average_4")
    x = np.arange(100, dtype=np.float32)
    out = (rp.Data(x, Fs=1e3) + f).data
    assert np.allclose(out[4:], np.convolve(x, [0.25]*4)[4:100])

@pytest.mark.parametrize("kind", ["lowpass", "highpass"])
def test_windowed_matches_the_code_designs(kind):
    f = rp.signal.filters.load("fir", f"windowed/{kind}_0p1_60db")
    ref = getattr(rp.signal.filters, f"{'low' if kind == 'lowpass' else 'high'}_pass_kaiser")(0.1, 0.02, 60, 1.0).filt
    assert np.allclose(f.taps(), ref)

def test_windowed_meets_its_stopband():
    h = rp.signal.filters.load("fir", "windowed/lowpass_0p1_60db").taps()
    H = np.abs(np.fft.rfft(h, 8192))
    f = np.fft.rfftfreq(8192)
    assert H[0] == pytest.approx(1.0, abs=1e-3)
    assert 20*np.log10(H[f > 0.12].max()) < -58 # transition 0.02 wide around 0.1

def test_hz_units_follow_the_sample_rate(tmp_path):
    (tmp_path / "filter" / "fir" / "windowed").mkdir(parents=True)
    obj = {"base": "fir", "type": "windowed", "pass": "lowpass", "cutoff": 25e3, "taps": 101, "units": "hz"} # No window: np.hamming, as low_pass
    (tmp_path / "filter" / "fir" / "windowed" / "lp_25k.json").write_text(json.dumps(obj))
    rp.add_config_path(tmp_path)
    try:
        f = rp.signal.filters.load("fir", "windowed/lp_25k")
        assert np.allclose(f.taps(Fs=250e3), rp.signal.filters.low_pass(101, 25e3, 250e3).filt)
        with pytest.raises(ConfigError, match="sample rate"):
            f.taps()
    finally:
        _config._EXTRA.remove(tmp_path)

def test_highpass_needs_odd_taps(tmp_path):
    (tmp_path / "filter" / "fir").mkdir(parents=True)
    obj = {"base": "fir", "type": "windowed", "pass": "highpass", "cutoff": 0.2, "taps": 100}
    (tmp_path / "filter" / "fir" / "hp.json").write_text(json.dumps(obj))
    rp.add_config_path(tmp_path)
    try:
        with pytest.raises(ConfigError, match="odd"):
            rp.signal.filters.load("fir", "hp").taps()
    finally:
        _config._EXTRA.remove(tmp_path)

def test_bad_definitions_fail_on_load(tmp_path):
    (tmp_path / "filter" / "fir").mkdir(parents=True)
    for n, obj in {"nopass": {"type": "windowed", "taps": 11, "cutoff": 0.1},
                   "nospec": {"type": "windowed", "pass": "lowpass", "cutoff": 0.1},
                   "badtype": {"type": "iir"},
                   "badunits": {"type": "taps", "taps": [1], "units": "rad"},
                   "empty": {"type": "taps", "taps": []}}.items():
        (tmp_path / "filter" / "fir" / f"{n}.json").write_text(json.dumps({"base": "fir", **obj}))
    rp.add_config_path(tmp_path)
    try:
        for n in ("nopass", "nospec", "badtype", "badunits", "empty"):
            with pytest.raises(ConfigError):
                rp.signal.filters.load("fir", n)
    finally:
        _config._EXTRA.remove(tmp_path)

@pytest.mark.parametrize("beta", ["0p35", "0p25", "0p20"])
def test_rrc_matches_the_class(beta):
    f = rp.signal.filters.load("pulse", f"rrc/{beta}")
    b = float(beta.replace("p", "."))
    assert np.allclose(f.taps(sps=8), rp.signal.filters.ShapingRRC(None, 8, b).pulse)

def test_shaping_takes_sps_from_the_stream():
    f = rp.signal.filters.load("pulse", "rc/0p35")
    out = (rp.Data(np.ones(500, dtype=np.float32), Fs=8e3, sps=8) + f).data
    assert len(out) == 500
    with pytest.raises(ConfigError, match="samples per symbol"):
        f.taps()

def test_window_presets_and_builtins():
    assert np.allclose(rp.signal.filters.Window.get("kaiser/a8")(64), np.kaiser(64, 8))
    assert np.allclose(rp.signal.filters.Window.get("a8")(64), np.kaiser(64, 8)) # a bare name, while unique
    assert np.allclose(rp.signal.filters.Window.get("hann")(16), rp.signal.filters.impl.window.hann(16))
    w = rp.signal.filters.Window("tukey/a0p25", 32)
    assert np.allclose(w.w, rp.signal.filters.impl.window.tukey(32, 0.25))
    with pytest.raises(ConfigError, match="No window"):
        rp.signal.filters.Window.get("nope")
    with pytest.raises(ConfigError):
        rp.signal.filters.Window.get("np") # Not a window, though it is a name in the module

def test_inline_window_in_a_design(tmp_path):
    (tmp_path / "filter" / "fir").mkdir(parents=True)
    obj = {"base": "fir", "type": "windowed", "pass": "lowpass", "cutoff": 0.2, "taps": 51, "window": {"type": "kaiser", "params": {"a": 8}}}
    (tmp_path / "filter" / "fir" / "inline.json").write_text(json.dumps(obj))
    obj["window"] = "kaiser/a8"
    (tmp_path / "filter" / "fir" / "ref.json").write_text(json.dumps(obj))
    rp.add_config_path(tmp_path)
    try:
        a, b = rp.signal.filters.load("fir", "inline").taps(), rp.signal.filters.load("fir", "ref").taps()
        assert np.allclose(a, b) and np.allclose(a, rp.signal.filters.low_pass(51, 0.2, fs=1.0, window=rp.signal.filters.kaiser_window(8)).filt)
    finally:
        _config._EXTRA.remove(tmp_path)

def test_remez_design(tmp_path):
    (tmp_path / "filter" / "fir").mkdir(parents=True)
    obj = {"base": "fir", "type": "remez", "taps": 41, "bands": [0, 0.1, 0.15, 0.5], "desired": [1, 0]}
    (tmp_path / "filter" / "fir" / "rz.json").write_text(json.dumps(obj))
    rp.add_config_path(tmp_path)
    try:
        assert np.allclose(rp.signal.filters.load("fir", "rz").taps(), rp.signal.filters.remez(41, [0, 0.1, 0.15, 0.5], [1, 0]).filt, atol=1e-6)
    finally:
        _config._EXTRA.remove(tmp_path)

def test_named_hamming_is_the_builtin_one(tmp_path):
    (tmp_path / "filter" / "fir").mkdir(parents=True)
    obj = {"base": "fir", "type": "windowed", "pass": "lowpass", "cutoff": 0.1, "taps": 101, "window": "hamming"}
    (tmp_path / "filter" / "fir" / "h.json").write_text(json.dumps(obj))
    rp.add_config_path(tmp_path)
    try:
        h = rp.signal.filters.load("fir", "h").taps()
        assert np.allclose(h, rp.signal.filters.low_pass(101, 0.1, fs=1.0, window=rp.signal.filters.impl.window.hamming).filt)
        assert not np.allclose(h, rp.signal.filters.low_pass(101, 0.1, fs=1.0).filt)
    finally:
        _config._EXTRA.remove(tmp_path)
