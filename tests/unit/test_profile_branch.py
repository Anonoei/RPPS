"""Profiles that fan out: a trunk, then branches reading the same blocks, a header in one driving another
(rpps.profile.branch)"""
import numpy as np
import pytest

import rpps as rp
from rpps.base.types import Rep
from rpps.errors import ConfigError
from rpps.profile.profile import Profile

def bits_of(d):
    x = np.asarray(d.data)
    return x < 0 if x.dtype.kind == "f" else x.astype(bool)

def channel_bits(bits, flush=True, stream=None):
    d = rp.Data(np.asarray(bits, bool), flush=flush, stream=stream)
    return d.derive(d.data, d.CAT, d.DT, rep=Rep.CHANNEL)

@pytest.fixture(autouse=True)
def codes_table(tmp_path, monkeypatch):
    """A table from the header's length field to a code name (a control's strings come from fields or a table's columns)"""
    d = tmp_path / "table" / "test_branch"
    d.mkdir(parents=True)
    (d / "codes.json").write_text('{"base": "table", "comment": "test", "key": "length", "columns": ["length", "code"], '
                                  '"rows": [[20, "hamming/15_11"]]}')
    monkeypatch.setattr(rp._config, "_EXTRA", [tmp_path] + list(rp._config._EXTRA))

def profile(order=("head", "body"), drive="fec", frame=None):
    """head: the BLE advertising header (CRC-24) of the bits; body: a Hamming code over the same bits, its code switched to
    (15, 11) by a header of 20 bytes in the head branch"""
    return Profile("test/branch", {
        "steps": [{"branch": "split", "order": list(order), "frame": frame, "branches": {
            "head": [{"code": "uncoded", "id": "unc"}, {"link": "header", "id": "hdr", "name": "ble_adv"}],
            "body": [{"code": "block", "id": "fec", "name": "hamming/7_4"}]}}],
        "controls": {"hdr": {"table": "test_branch/codes", "drive": {drive: {"name": "code"}}}}})

@pytest.fixture
def framed(rng):
    msg = rng.integers(0, 2, 8*20).astype(bool)
    return np.asarray((rp.Data(msg) * rp.link.load("header/ble_adv")).data).astype(bool) # 16 + 160 + 24 = 200 bits

def test_branches_read_the_same_block_and_give_each_their_output(framed):
    out = profile().rx()(channel_bits(framed))
    assert set(out) == {"head", "body"}
    assert out["head"].stream is not out["body"].stream # each branch is a stream of its own
    assert out["head"].stream.est["ble_adv"]["frames"] == 1

def test_a_header_in_one_branch_drives_the_next(framed):
    """The header's fields apply from its payload's first bit (16): the body decodes (7, 4) up to there (bits 14 and 15, a
    codeword cut short, dropped) and (15, 11) from there"""
    out = profile().rx()(channel_bits(framed))
    h = lambda name, b: bits_of(channel_bits(b) / rp.code.load(f"block/hamming/{name}"))
    want = np.concatenate((h("7_4", framed[:14]), h("15_11", framed[16:16 + (200 - 16)//15*15])))
    assert np.array_equal(bits_of(out["body"]), want)
    undriven = bits_of(channel_bits(framed[:len(framed)//7*7]) / rp.code.load("block/hamming/7_4"))
    assert not np.array_equal(bits_of(out["body"]), undriven[:len(want)])

def test_a_branch_cannot_drive_one_that_ran_before_it(framed):
    with pytest.raises(ConfigError, match="runs before it"):
        profile(order=("body", "head")).rx()

def test_branch_step_is_last_and_receive_only():
    with pytest.raises(ConfigError, match="the last"):
        Profile("t", {"steps": [{"branch": "b", "branches": {"a": [{"code": "uncoded"}]}}, {"code": "uncoded"}]})
    with pytest.raises(ConfigError, match="receive only"):
        profile().tx()
    with pytest.raises(ConfigError, match="several branches"):
        Profile("t", {"steps": [{"branch": "b", "branches": {"a": [{"code": "uncoded", "id": "u"}],
                                                           "c": [{"code": "uncoded", "id": "u"}]}}]})

def test_controls_name_steps_inside_branches():
    with pytest.raises(ConfigError, match="no such step"):
        profile(drive="nope")

@pytest.mark.parametrize("cut", [10, 16, 17, 100])
def test_the_block_cut_does_not_matter(framed, cut):
    """The body may have read part of the stream when the header's fields arrive: they apply at their own position"""
    whole = profile(frame=200).rx()(channel_bits(framed))
    rx = profile(frame=200).rx() # The header's frame is 200 bits: the branches read it whole
    a = rx(channel_bits(framed[:cut], flush=False))
    b = rx(channel_bits(framed[cut:], stream=a["body"].stream)) # (any stream: the runner keeps its branches' own)
    assert np.array_equal(np.concatenate((bits_of(a["body"]), bits_of(b["body"]))), bits_of(whole["body"]))
