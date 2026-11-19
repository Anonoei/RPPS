"""The command line: listing, showing a definition, receiving a capture"""
import numpy as np
import pytest

import rpps as rp
from rpps import cli


def test_ls_and_show(capsys):
    assert cli.main(["ls", "crc"]) == 0
    assert "crc32_mpeg2" in capsys.readouterr().out
    assert cli.main(["show", "fec/block/rs/204_188"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# ") and '"base": "block"' in out
    assert cli.main(["show", "nope/x"]) == 1

@pytest.mark.slow
def test_rx_decodes_a_capture(tmp_path, capsys):
    p = rp.profile.load("dvbs2/normal")
    t = rp.profile.table("dvbs2/modcod")
    frames = [{"modcod": 4, "pilots": 1}]*3
    data = np.random.default_rng(1).integers(0, 2, sum(t.row({**f, "short": 0})["kbch"] - 80 for f in frames)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=frames)
    pad = np.zeros(3000, np.complex64)
    (np.concatenate((pad, o.data, pad))*np.exp(0.7j)).astype(np.complex64).tofile(tmp_path/"cap.cf64")
    assert cli.main(["rx", "dvbs2/normal", str(tmp_path/"cap.cf64"), "--fs", str(o.Fs), "--out", str(tmp_path/"out.bin")]) == 0
    got = np.unpackbits(np.fromfile(tmp_path/"out.bin", np.uint8)).astype(bool)
    assert np.array_equal(got[:len(data)], data)

def test_solve_names_a_prbs_in_a_bit_file(tmp_path, capsys):
    from rpps.cli import main
    rng = np.random.default_rng(0)
    s = list(rng.integers(0, 2, 23))
    while len(s) < 30000:
        s.append(s[-23] ^ s[-18])
    path = tmp_path / "bits.npy"
    np.save(path, np.array(s, np.uint8))
    assert main(["solve", str(path)]) == 0
    out = capsys.readouterr().out
    assert "PRBS23" in out and "scrambler-additive" in out
