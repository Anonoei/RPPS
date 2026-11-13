import json
import warnings
import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.errors import ConfigError
from rpps.signal.filters import iir
from test_iir import lfilter_ref, gain_db

def layer(tmp_path):
    rp.add_config_path(tmp_path)
    return tmp_path

@pytest.fixture
def user(tmp_path):
    rp.add_config_path(tmp_path)
    yield tmp_path
    _config._EXTRA.remove(tmp_path)

def put(user, category, ref, obj):
    f = user / ("pulse" if category == "pulse" else "filter") / category / f"{ref}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(obj))

# ---- IIR definitions

def test_shipped_iir_definitions_load_and_behave():
    assert set(rp.signal.filters.ls()) >= {"iir/sos", "iir/butter", "iir/single_pole", "iir/biquad"}
    dc = rp.signal.filters.load("iir", "sos/dc_blocker_0p995")
    x = np.full(4000, 3.0) + np.sin(0.3*np.arange(4000))
    y = dc.run(x)
    assert abs(y[-500:].mean()) < 0.05 and np.std(y[-500:]) > 0.5 # The mean goes, the tone stays
    assert gain_db(rp.signal.filters.load("iir", "butter/lowpass_4_0p1").sos(), 0.1) == pytest.approx(-3.0103, abs=1e-6)
    assert gain_db(rp.signal.filters.load("iir", "biquad/notch_0p1_q30").sos(), 0.1) < -150
    assert np.allclose(rp.signal.filters.load("iir", "single_pole/alpha_0p05").sos(), iir.single_pole(0.05))

def test_iir_definition_in_a_stream():
    f = rp.signal.filters.load("iir", "butter/lowpass_4_0p1")
    x = np.random.default_rng(0).normal(size=5000).astype(np.float32)
    out = (rp.Data(x, Fs=1e3) + f).data
    assert out.dtype == np.float32 and np.allclose(out, iir.IIR(f.sos()).run(x), atol=1e-5)

def test_iir_types(user):
    put(user, "iir", "ba1", {"base": "iir", "type": "ba", "b": [0.2], "a": [1, -0.8]})
    put(user, "iir", "zpk1", {"base": "iir", "type": "zpk", "z": [-1], "p": [[0.5, 0.3], [0.5, -0.3]], "k": 0.1})
    put(user, "iir", "ch1", {"base": "iir", "type": "cheby1", "order": 3, "pass": "bandpass", "cutoff": [0.1, 0.2], "ripple_db": 1, "gain": 2})
    ba = rp.signal.filters.load("iir", "ba1")
    assert np.allclose(ba.sos(), iir.single_pole(0.2))
    z = rp.signal.filters.load("iir", "zpk1")
    x = np.random.default_rng(3).normal(size=100)
    a = np.poly([0.5 + 0.3j, 0.5 - 0.3j]).real
    assert np.allclose(z.run(x), lfilter_ref([0, 0.1, 0.1], a, x)) # One zero for two poles: a sample of delay
    base = iir.design("cheby1", 3, "bandpass", [0.1, 0.2], ripple_db=1)
    assert gain_db(rp.signal.filters.load("iir", "ch1").sos(), 0.15) == pytest.approx(gain_db(base, 0.15) + 20*np.log10(2), abs=1e-9)

def test_iir_hz_units_follow_the_sample_rate(user):
    put(user, "iir", "lp_hz", {"base": "iir", "type": "butter", "order": 2, "pass": "lowpass", "cutoff": 25, "units": "hz"})
    f = rp.signal.filters.load("iir", "lp_hz")
    with pytest.raises(ConfigError, match="sample rate"):
        f.sos()
    assert gain_db(f.sos(Fs=250), 25/250) == pytest.approx(-3.0103, abs=1e-6)
    assert gain_db(f.sos(Fs=1000), 25/1000) == pytest.approx(-3.0103, abs=1e-6)
    a = (rp.Data(np.ones(500, dtype=np.float32), Fs=250.0) + f).data
    b = (rp.Data(np.ones(500, dtype=np.float32), Fs=1000.0) + f).data
    assert not np.allclose(a, b) # A slower rate means a faster filter, per sample

@pytest.mark.parametrize("name,obj,match", [
    ("notype", {"type": "wobble"}, "unknown iir type"),
    ("nosos", {"type": "sos"}, "needs `sos`"),
    ("noorder", {"type": "butter", "pass": "lowpass", "cutoff": 0.1}, "needs `order`"),
    ("badpass", {"type": "butter", "order": 2, "pass": "left", "cutoff": 0.1}, "pass must be"),
    ("highcut", {"type": "butter", "order": 2, "pass": "lowpass", "cutoff": 0.7}, "cutoff"),
    ("unstable", {"type": "sos", "sos": [[1, 0, 0, 1, -1.5, 0]]}, "unstable"),
    ("badunits", {"type": "single_pole", "alpha": 0.1, "units": "rad"}, "units"),
    ("badpair", {"type": "zpk", "z": [], "p": [[1, 2, 3]], "k": 1}, "pair"),
    ("badalpha", {"type": "single_pole", "alpha": 2}, "alpha"),
])
def test_bad_iir_definitions_fail_on_load(user, name, obj, match):
    put(user, "iir", name, {"base": "iir", **obj})
    with pytest.raises(ConfigError, match=match):
        rp.signal.filters.load("iir", name)

def test_iir_designed_for_another_rate_warns(user):
    put(user, "iir", "rated", {"base": "iir", "type": "single_pole", "alpha": 0.1, "fs": 1000.0})
    f = rp.signal.filters.load("iir", "rated")
    with pytest.warns(UserWarning, match="designed for Fs=1000"):
        rp.Data(np.ones(10, dtype=np.float32), Fs=500.0) + f

# ---- Complex FIR taps

def test_complex_taps(user):
    put(user, "fir", "taps/c", {"base": "fir", "type": "taps", "taps": [1, 0.5], "taps_imag": [0, 0.25]})
    f = rp.signal.filters.load("fir", "taps/c")
    assert np.allclose(f.taps(), [1, 0.5 + 0.25j])
    x = np.arange(8, dtype=np.complex64)
    assert np.allclose((rp.Data(x, Fs=1.0) + f).data, np.convolve(x, [1, 0.5 + 0.25j])[:8])
    put(user, "fir", "taps/bad", {"base": "fir", "type": "taps", "taps": [1, 2], "taps_imag": [1]})
    with pytest.raises(ConfigError, match="taps_imag"):
        rp.signal.filters.load("fir", "taps/bad")

# ---- Readers

def test_text_formats(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("# taps\n0.1, 0.2 % comment\n0.3 // more\n;0.4\n")
    d = rp.signal.filters.import_fir(f)
    assert d["taps"] == [0.1, 0.2, 0.3, 0.4] and "taps_imag" not in d and d["comment"] == "FIR from t.txt"
    f.write_text("1+2j\n3-1i\n")
    d = rp.signal.filters.import_fir(f)
    assert d["taps"] == [1, 3] and d["taps_imag"] == [2, -1]
    f.write_text("1 2 x3\n")
    with pytest.raises(ValueError, match=r"t\.txt:1: 'x3' is not a number"):
        rp.signal.filters.import_fir(f)
    f.write_text("1 nan\n")
    with pytest.raises(ValueError, match="inf or nan"):
        rp.signal.filters.import_fir(f)

def test_numpy_formats(tmp_path):
    np.save(tmp_path / "a.npy", np.array([0.5, 0.25]))
    assert rp.signal.filters.import_fir(tmp_path / "a.npy")["taps"] == [0.5, 0.25]
    np.savez(tmp_path / "one.npz", taps=np.array([1.0, 2.0]))
    assert rp.signal.filters.import_fir(tmp_path / "one.npz")["taps"] == [1, 2]
    np.savez(tmp_path / "two.npz", taps=np.array([1.0]), other=np.array([2.0]))
    with pytest.raises(ValueError, match="say key"):
        rp.signal.filters.import_fir(tmp_path / "two.npz")
    assert rp.signal.filters.import_fir(tmp_path / "two.npz", key="other")["taps"] == [2]
    with pytest.raises(ValueError, match="no array"):
        rp.signal.filters.import_fir(tmp_path / "two.npz", key="nope")
    np.save(tmp_path / "o.npy", np.array([None, 1], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        rp.signal.filters.import_fir(tmp_path / "o.npy") # Pickled data is never loaded

def test_raw_binary(tmp_path):
    np.array([0.1, 0.2, 0.3], dtype=np.float32).tofile(tmp_path / "a.f32")
    d = rp.signal.filters.import_fir(tmp_path / "a.f32")
    assert d["taps"] == [0.1, 0.2, 0.3] # float32 values are written short, not as 0.10000000149011612
    assert "0.10000000149011612" not in json.dumps(d)
    np.array([0.1 + 0.2j, 0.3 - 0.1j], dtype=np.complex64).tofile(tmp_path / "c.bin")
    d = rp.signal.filters.import_fir(tmp_path / "c.bin", dtype="cf32")
    assert d["taps"] == [0.1, 0.3] and d["taps_imag"] == [0.2, -0.1]
    np.array([1, -2], dtype="<i2").tofile(tmp_path / "i.raw")
    assert rp.signal.filters.import_fir(tmp_path / "i.raw", dtype="i16", scale=0.5)["taps"] == [0.5, -1]
    with pytest.raises(ValueError, match="no header"):
        rp.signal.filters.import_fir(tmp_path / "c.bin")
    (tmp_path / "odd.f32").write_bytes(b"\0" * 6)
    with pytest.raises(ValueError, match="whole number"):
        rp.signal.filters.import_fir(tmp_path / "odd.f32")

def test_source_arrays(tmp_path):
    f = tmp_path / "h.h"
    f.write_text("// generated\nstatic const float taps[8] = {0.1f, -0.2f, /* skip, 9 */ 3.0e-2f,\n 4f, 1.5L};\nint n = 5;\n")
    assert rp.signal.filters.import_fir(f)["taps"] == [0.1, -0.2, 0.03, 4, 1.5]
    g = tmp_path / "t.py"
    g.write_text("import numpy as np\ntaps = np.array([0.25, 0.5,  # half\n  0.25])\n")
    assert rp.signal.filters.import_fir(g)["taps"] == [0.25, 0.5, 0.25]
    g.write_text("taps = firwin(11, 0.2)\n")
    with pytest.raises(ValueError, match="no array literal"):
        rp.signal.filters.import_fir(g)
    g.write_text("taps = [0.1, 0.2\n")
    with pytest.raises(ValueError, match="unbalanced"):
        rp.signal.filters.import_fir(g)

def test_xilinx_coe(tmp_path):
    f = tmp_path / "x.coe"
    f.write_text("; comment\nradix = 10;\ncoefdata = 16384, -8192,\n 32767;\n")
    assert rp.signal.filters.import_fir(f, scale=1/32768)["taps"] == [0.5, -0.25, 32767/32768]
    f.write_text("radix = 16;\ncoefdata = 40, 20;\n")
    with pytest.raises(ValueError, match="radix 16"):
        rp.signal.filters.import_fir(f)
    f.write_text("radix = 10;\n")
    with pytest.raises(ValueError, match="no coefdata"):
        rp.signal.filters.import_fir(f)

def test_unknown_files(tmp_path):
    (tmp_path / "w.xyz").write_text("1 2")
    with pytest.raises(ValueError, match="unknown file type"):
        rp.signal.filters.import_fir(tmp_path / "w.xyz")
    assert rp.signal.filters.import_fir(tmp_path / "w.xyz", fmt="text")["taps"] == [1, 2]
    (tmp_path / "m.txt").write_text("1 2 3\n4 5\n")
    assert rp.signal.filters.import_fir(tmp_path / "m.txt")["taps"] == [1, 2, 3, 4, 5] # Wrapped lines are one list
    (tmp_path / "n.h").write_text("a = {{1, 2}, {3, 4}};")
    with pytest.raises(ValueError, match="2 rows"):
        rp.signal.filters.import_fir(tmp_path / "n.h")

# ---- IIR imports

def test_import_sos(tmp_path):
    sos = iir.design("butter", 4, "lowpass", 0.2)
    np.save(tmp_path / "s.npy", sos)
    np.savetxt(tmp_path / "s.txt", sos)
    for f in ("s.npy", "s.txt"):
        d = rp.signal.filters.import_iir(tmp_path / f)
        assert d["type"] == "sos" and np.allclose(d["sos"], sos)
    np.savez(tmp_path / "s.npz", sos=sos)
    assert np.allclose(rp.signal.filters.import_iir(tmp_path / "s.npz")["sos"], sos)
    (tmp_path / "bad.txt").write_text("1 2 3\n")
    with pytest.raises(ValueError):
        rp.signal.filters.import_iir(tmp_path / "bad.txt")
    (tmp_path / "unstable.txt").write_text("1 0 0 1 -1.2 0\n")
    with pytest.raises(ValueError, match="unstable"):
        rp.signal.filters.import_iir(tmp_path / "unstable.txt")

def test_import_ba_and_zpk(tmp_path):
    (tmp_path / "ba.txt").write_text("0.2\n1 -0.8\n")
    d = rp.signal.filters.import_iir(tmp_path / "ba.txt", form="ba")
    assert d["type"] == "ba" and d["b"] == [0.2] and d["a"] == [1, -0.8]
    np.savez(tmp_path / "ba.npz", b=np.array([0.2]), a=np.array([1, -0.8]))
    assert rp.signal.filters.import_iir(tmp_path / "ba.npz", form="ba")["a"] == [1, -0.8]
    (tmp_path / "zpk.txt").write_text("-1\n0.5+0.3j 0.5-0.3j\n0.1\n")
    d = rp.signal.filters.import_iir(tmp_path / "zpk.txt", form="zpk")
    assert d["z"] == [-1] and d["p"] == [[0.5, 0.3], [0.5, -0.3]] and d["k"] == 0.1
    np.savez(tmp_path / "zpk.npz", z=np.array([-1.0]), p=np.array([0.5 + 0.3j, 0.5 - 0.3j]), k=np.array(0.1))
    assert rp.signal.filters.import_iir(tmp_path / "zpk.npz", form="zpk")["k"] == 0.1
    with pytest.raises(ValueError, match="two rows"):
        rp.signal.filters.import_iir(tmp_path / "zpk.txt", form="ba")
    with pytest.raises(ValueError, match="needs"):
        rp.signal.filters.import_iir(tmp_path / "ba.npz", form="zpk")
    with pytest.raises(ValueError, match="form must be"):
        rp.signal.filters.import_iir(tmp_path / "ba.txt", form="tf")

def test_gnuradio_feedback_sign():
    """GNU Radio's iir_filter_ffd: oldstyle (default) y[n] = sum ff x + sum fb[k] y[n-k], fb[0] unused; the wiki's single pole is
    fftaps [alpha], fbtaps [1, 1-alpha]"""
    alpha = 0.3
    old = rp.signal.filters.ba_def([alpha], [1.0, 1 - alpha], convention="gnuradio")
    assert old["a"] == [1, -(1 - alpha)]
    assert np.allclose(iir.tf2sos(old["b"], old["a"]), iir.single_pole(alpha))
    x = np.random.default_rng(2).normal(size=60)
    ff, fb = [0.5, 0.2, 0.1], [9.0, 0.4, -0.3] # fb[0] is ignored
    ref = np.zeros(60)
    for n in range(60):
        ref[n] = sum(ff[k]*x[n-k] for k in range(3) if n-k >= 0) + sum(fb[k]*ref[n-k] for k in (1, 2) if n-k >= 0)
    d = rp.signal.filters.ba_def(ff, fb, convention="gnuradio")
    assert np.allclose(iir.IIR(iir.tf2sos(d["b"], d["a"])).run(x), ref)
    new = rp.signal.filters.ba_def([alpha], [1.0, -(1 - alpha)], convention="gnuradio-new")
    assert new["a"] == [1, -(1 - alpha)] == rp.signal.filters.ba_def([alpha], [1.0, -(1 - alpha)])["a"]
    with pytest.raises(ValueError, match="convention"):
        rp.signal.filters.ba_def([1], [1], convention="octave")

def test_builders_validate():
    with pytest.raises(ValueError):
        rp.signal.filters.fir_def([])
    with pytest.raises(ValueError):
        rp.signal.filters.fir_def([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="inf or nan"):
        rp.signal.filters.fir_def([1, np.inf])
    with pytest.raises(ValueError, match="unstable"):
        rp.signal.filters.zpk_def([], [1.2], 1)
    with pytest.raises(ValueError, match="unstable"):
        rp.signal.filters.ba_def([1], [1, -2])
    assert rp.signal.filters.sos_def(np.array([[1, 0, 0, 1, 0, 0]], dtype=np.float32))["sos"] == [[1, 0, 0, 1, 0, 0]]

# ---- Writing, and loading what was written

def test_write_then_load(tmp_path):
    taps = np.array([0.1, 0.4, 0.4, 0.1], dtype=np.float32)
    taps.tofile(tmp_path / "t.f32")
    out = tmp_path / "mine"
    f = rp.signal.filters.import_fir(tmp_path / "t.f32", name="my_lp", dest=out, comment="mine")
    assert f == out / "filter" / "fir" / "taps" / "my_lp.json"
    text = f.read_text()
    assert text.index('"base"') < text.index('"comment"') < text.index('"taps": [') and "0.10000000149" not in text
    sos = iir.design("cheby2", 4, "highpass", 0.2, atten_db=50)
    np.save(tmp_path / "s.npy", sos)
    g = rp.signal.filters.import_iir(tmp_path / "s.npy", name="my_hp", dest=out)
    assert g == out / "filter" / "iir" / "sos" / "my_hp.json"
    rp.add_config_path(out)
    try:
        assert np.allclose(rp.signal.filters.load("fir", "taps/my_lp").taps(), taps)
        assert np.allclose(rp.signal.filters.load("iir", "sos/my_hp").sos(), sos)
        assert "taps/my_lp" in [f"{k.split('/')[1]}/{n}" for k, v in rp.signal.filters.ls().items() if k.startswith("fir/") for n in v]
    finally:
        _config._EXTRA.remove(out)

def test_save_never_replaces_or_escapes(tmp_path):
    (tmp_path / "t.txt").write_text("1 2 3")
    out = tmp_path / "cfg"
    first = rp.signal.filters.import_fir(tmp_path / "t.txt", name="x", dest=out)
    with pytest.raises(FileExistsError):
        rp.signal.filters.import_fir(tmp_path / "t.txt", name="x", dest=out)
    (tmp_path / "t.txt").write_text("4 5 6")
    rp.signal.filters.import_fir(tmp_path / "t.txt", name="x", dest=out, overwrite=True)
    assert json.loads(first.read_text())["taps"] == [4, 5, 6]
    for bad in ("../x", "/etc/x", "a//b", ".hidden", "UP per", ""):
        with pytest.raises(ValueError, match="not a definition name"):
            rp.signal.filters.import_fir(tmp_path / "t.txt", name=bad, dest=out)
    assert sorted(p.name for p in out.rglob("*.json")) == ["x.json"]

def test_a_failed_import_writes_nothing(tmp_path):
    (tmp_path / "bad.txt").write_text("1 two 3")
    out = tmp_path / "cfg"
    with pytest.raises(ValueError):
        rp.signal.filters.import_fir(tmp_path / "bad.txt", name="x", dest=out)
    assert not out.exists()

def test_default_destination_is_the_first_added_layer(user, tmp_path):
    (tmp_path / "t.txt").write_text("1 2 3")
    f = rp.signal.filters.import_fir(tmp_path / "t.txt", name="dflt")
    assert f == user / "filter" / "fir" / "taps" / "dflt.json"

def test_alist_import_keeps_files(tmp_path):
    from test_coding import HAMMING_ALIST
    (tmp_path / "h.alist").write_text(HAMMING_ALIST)
    rp.code.import_alist(tmp_path / "h.alist", "h", dest=tmp_path / "cfg")
    with pytest.raises(FileExistsError):
        rp.code.import_alist(tmp_path / "h.alist", "h", dest=tmp_path / "cfg")
