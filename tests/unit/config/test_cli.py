"""The command line: listing, showing a definition, receiving a capture"""
import json

import numpy as np
import pytest

import rpps as rp
from rpps import cli
from support.signals import shaped


def test_ls_and_show(capsys):
    assert cli.main(["ls", "crc"]) == 0
    assert "crc32_mpeg2" in capsys.readouterr().out
    assert cli.main(["show", "fec/block/rs/204_188"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# ") and '"base": "block"' in out
    assert cli.main(["show", "fec/block/bch/dvbs2_1_2"]) == 0
    assert capsys.readouterr().out.startswith("# a variant in ") # (a row of bch/dvbs2_normal, not a file)
    assert cli.main(["show", "nope/x"]) == 1

@pytest.mark.slow
def test_rx_decodes_a_capture(tmp_path, capsys):
    p = rp.profile.load("dvbs2/normal")
    t = rp.profile.table("dvbs2/modcod")
    frames = [{"modcod": 4, "pilots": 1}]*3
    data = np.random.default_rng(1).integers(0, 2, sum(t.row({**f, "short": 0})["kbch"] - 80 for f in frames)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=frames)
    pad = np.zeros(3000, np.complex64)
    (np.concatenate((pad, o.data, pad))*np.exp(0.7j)).astype(np.complex64).tofile(tmp_path/"cap.cf32")
    assert cli.main(["rx", "dvbs2/normal", str(tmp_path/"cap.cf32"), "--fs", str(o.Fs), "--out", str(tmp_path/"out.bin")]) == 0
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

def test_tx_writes_what_rx_receives(tmp_path, capsys):
    """BLE advertising packets out of `rpps tx`, back through the profile"""
    payload = bytes(range(60))
    (tmp_path/"in.bin").write_bytes(payload)
    assert cli.main(["tx", "ble/adv", str(tmp_path/"ble.cf32"), "--rate", "1e6", "--in", str(tmp_path/"in.bin"),
                     "--packet", "20", "--field", "type=2"]) == 0
    assert "samples at 4e+06 Hz" in capsys.readouterr().out
    x = np.fromfile(tmp_path/"ble.cf32", np.complex64)
    p = rp.profile.load("ble/adv")
    outs = p.rx()(rp.Data(x, Fs=4e6, sps=4, flush=True)) # one per packet (gap_s apart, the first after one too)
    bits = [np.asarray(o.data) < 0 if np.asarray(o.data).dtype.kind == "f" else np.asarray(o.data).astype(bool) for o in outs]
    got = np.packbits(np.concatenate(bits)).tobytes()
    assert got == payload

def test_rx_gives_the_branch_asked_for(tmp_path, capsys):
    """A profile that branches (lte/downlink): --branch picks whose bits go out"""
    from rpps.grid import lte
    sib = np.random.default_rng(2).integers(0, 2, 224).astype(np.uint8)
    x = lte.tx.channel(lte.tx.downlink(pci=10, ports=1, n_rb=6, frames=4, sib=sib), delay=700, snr_db=20)
    x.tofile(tmp_path/"lte.cf32")
    assert cli.main(["rx", "lte/downlink", str(tmp_path/"lte.cf32"), "--fs", "1.92e6", "--param", "n_rb=6", "--branch", "pdsch",
                     "--out", str(tmp_path/"sib.bin")]) == 0
    assert (tmp_path/"sib.bin").read_bytes() == np.packbits(np.tile(sib, 2)).tobytes() # Frames 0 and 2
    with pytest.raises(SystemExit, match="no branch"):
        cli.main(["rx", "lte/downlink", str(tmp_path/"lte.cf32"), "--fs", "1.92e6", "--param", "n_rb=6", "--branch", "nope"])

def test_rx_prints_the_messages(tmp_path, capsys):
    """--messages: every frame decoded, a JSON line each (the MIB, the CFI, the DCI, SIB1 of an LTE cell)"""
    import json
    from rpps.grid import lte
    sib = np.random.default_rng(2).integers(0, 2, 224).astype(np.uint8)
    x = lte.tx.channel(lte.tx.downlink(pci=10, ports=1, n_rb=6, frames=4, sib=sib), delay=700, snr_db=20)
    x.tofile(tmp_path/"lte.cf32")
    assert cli.main(["rx", "lte/downlink", str(tmp_path/"lte.cf32"), "--fs", "1.92e6", "--param", "n_rb=6", "--messages"]) == 0
    lines = [json.loads(ln) for ln in capsys.readouterr().out.splitlines()]
    kinds = {m["kind"] for m in lines}
    assert {"pbch", "pcfich", "pdcch", "pdsch"} <= kinds
    sib1 = [m for m in lines if m["kind"] == "pdsch" and m["crc_ok"]]
    assert sib1 and all(m["payload"] == np.packbits(sib).tobytes().hex() for m in sib1)

def test_rx_only_the_kinds_asked_for(tmp_path, capsys):
    import json
    from rpps.grid import lte
    x = lte.tx.channel(lte.tx.downlink(pci=10, ports=1, n_rb=6, frames=4), delay=700, snr_db=20)
    x.tofile(tmp_path/"lte.cf32")
    assert cli.main(["rx", "lte/downlink", str(tmp_path/"lte.cf32"), "--fs", "1.92e6", "--param", "n_rb=6", "--messages",
                     "--kinds", "pbch,pcfich"]) == 0
    kinds = {json.loads(ln)["kind"] for ln in capsys.readouterr().out.splitlines()}
    assert kinds == {"pbch", "pcfich"}

def test_fmt_takes_out_a_base_the_folder_says(tmp_path, capsys):
    d = tmp_path / "fec" / "block" / "rs"
    d.mkdir(parents=True)
    (d / "a.json").write_text(json.dumps({"base": "block", "type": "rs", "comment": "x", "n": 15, "k": 11, "m": 4}))
    (d / "b.json").write_text(json.dumps({"base": "conv", "comment": "in the wrong folder"}))
    assert cli.main(["fmt", str(tmp_path)]) == 1
    assert "base" not in json.loads((d / "a.json").read_text())
    assert "folder says 'block'" in capsys.readouterr().err and "base" in json.loads((d / "b.json").read_text())

def test_ls_tags_and_abstract_parents_and_steps(capsys):
    assert cli.main(["ls", "--tags"]) == 0
    out = capsys.readouterr().out
    tag = next(iter(rp.tags()))
    assert tag in out
    assert cli.main(["ls", "--tag", "dvb-s2"]) == 0
    assert "dvbs2" in capsys.readouterr().out
    assert cli.main(["ls", "crc", "--all"]) == 0
    assert "crc32_mpeg2" in capsys.readouterr().out
    assert cli.main(["steps"]) == 0
    out = capsys.readouterr().out
    assert "Step kinds" in out and "timing" in out

def test_survey_and_profile_a_capture(tmp_path, capsys):
    """A carrier in noise: survey lists it, profile writes one JSON line for it"""
    rng = np.random.default_rng(0)
    x, _ = shaped(rng, [1, 1j, -1, -1j], 2**15, 4, snr_db=20.0)
    x.tofile(tmp_path/"cap.cf32")
    args = [str(tmp_path/"cap.cf32"), "--fs", "1e6"]
    assert cli.main(["survey", *args]) == 0
    out = capsys.readouterr().out
    assert "1 signals" in out and "MS/s" in out
    assert cli.main(["profile", *args, "--signals", "0", "--blocks", "4", "--no-blind", "--json", str(tmp_path/"p.jsonl")]) == 0
    lines = (tmp_path/"p.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["file"].endswith("cap.cf32")

@pytest.mark.parametrize("fmt", ["bytes", "bits"])
def test_solve_reads_packed_and_unpacked_bit_files(fmt, tmp_path, capsys):
    bits = np.random.default_rng(1).integers(0, 2, 8000).astype(np.uint8)
    path = tmp_path/"x.bin"
    path.write_bytes((np.packbits(bits) if fmt == "bytes" else bits).tobytes())
    assert cli.main(["solve", str(path), "--format", fmt]) == 0
    out = capsys.readouterr().out
    assert "8000 bits" in out and "nothing found" in out

def test_solve_takes_symbols(tmp_path, capsys):
    """Complex symbols with no frame structure are sliced to bits"""
    rng = np.random.default_rng(2)
    sym = rp.symbol.load("qpsk").points[rng.integers(0, 4, 4000)]
    np.save(tmp_path/"sym.npy", sym)
    assert cli.main(["solve", str(tmp_path/"sym.npy")]) == 0
    out = capsys.readouterr().out
    assert "4000 symbols: no frame structure" in out and "8000 bits" in out

@pytest.fixture
def generated(tmp_path, capsys):
    """`rpps gen-ldpc` on a small address table: the config directory it wrote into"""
    (tmp_path/"tables.hh").write_text("""struct DVB_S2_TABLE_B5
{
\tstatic const int M = 2;
\tstatic const int N = 8;
\tstatic const int K = 4;
\tstatic constexpr int DEG[] = {3, 2, 0};
\tstatic constexpr int LEN[] = {1, 1};
\tstatic constexpr int POS[] = {0, 1, 2, 3, 1};
};
""")
    out = tmp_path/"cfg"/"fec"/"block"/"ldpc"
    assert cli.main(["gen-ldpc", str(tmp_path/"tables.hh"), "B5", "--out", str(out)]) == 0
    assert "dvbs2_64800_3_5 rows 2 k 4" in capsys.readouterr().out
    return tmp_path/"cfg"

def test_gen_ldpc_turns_address_tables_into_definitions(generated):
    obj = json.loads((generated/"fec"/"block"/"ldpc"/"dvbs2_64800_3_5.json").read_text())
    assert (obj["n"], obj["k"], obj["group"], obj["table"]) == (8, 4, 2, [[0, 1, 2], [3, 1]])

def test_a_generated_ldpc_definition_loads(generated, config_layer):
    config_layer(generated)
    code = rp.code.load("block/ldpc/dvbs2_64800_3_5")
    word = np.asarray(code.encoder.encode(np.array([1, 0, 1, 1], bool))).astype(bool)
    assert len(word) == 8 and word[:4].tolist() == [True, False, True, True]
