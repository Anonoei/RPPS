"""5G NR downlink to SIB1 through LEO Doppler (rpps.grid.nr, profile nr/downlink): synthetic, from rpps.grid.nr.tx"""
import json
from pathlib import Path

import numpy as np
import pytest

import rpps as rp
from rpps.grid import nr
from rpps.grid.nr import phy

FS, NFFT, FC = 7.68e6, 512, 2.185e9
V = json.loads((Path(__file__).parent/"vectors"/"nr_py3gpp_ssb.json").read_text())

def unhex(h, n):
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), np.uint8))[:n]

def signs(h, n):
    return 1 - 2.0*unhex(h, n)

# ---- against py3gpp (tests/unit/vectors/nr_py3gpp_ssb.json: what it is in its "source")

def test_sync_signals_match_py3gpp():
    for n2, h in V["pss"].items():
        assert np.array_equal(phy.pss(int(n2)), signs(h, 127))
    for pci, h in V["sss"].items():
        assert np.array_equal(phy.sss(int(pci)//3, int(pci) % 3), signs(h, 127))
    for d in V["dmrs"]:
        assert np.allclose(phy.pbch_dmrs(d["pci"], d["ib"]), (signs(d["re"], 144) + 1j*signs(d["im"], 144))/np.sqrt(2))

@pytest.mark.parametrize("v", V["bch"], ids=lambda v: f"pci{v['pci']}_sfn{v['sfn']}")
def test_bch_matches_py3gpp(v):
    """MIB and timing bits interleaved and scrambled (38.212 7.1.1-7.1.2), CRC-24C, polar, rate matching: and back"""
    mib = unhex(v["mib"], 24)
    a = phy.payload_encode(mib, v["sfn"], v["hrf"], 0, v["pci"])
    assert np.array_equal(rp.code.load("block/polar/nr_pbch").encoder.encode(a).astype(np.uint8), unhex(v["e"], 864))
    m, sfn, hrf, _ = phy.payload_decode(a, v["pci"])
    assert np.array_equal(m, mib) and (sfn, hrf) == (v["sfn"], v["hrf"])

@pytest.mark.parametrize("v", V["pbch"], ids=lambda v: f"pci{v['pci']}_v{v['v']}")
def test_pbch_scrambling_matches_py3gpp(v):
    z = phy.qam(unhex(v["e"], 864) ^ phy.pbch_scramble(v["pci"], v["v"]), 2)
    assert np.allclose(z, (signs(v["re"], 432) + 1j*signs(v["im"], 432))/np.sqrt(2))

# ---- tables and maps

def test_coreset0_and_type0_tables():
    assert phy.coreset0(0) == {"index": 0, "pattern": 1, "rbs": 24, "symbols": 2, "offset": 0}
    assert phy.coreset0(14)["rbs"] == 96
    with pytest.raises(ValueError, match="reserved"):
        phy.coreset0(15)
    assert phy.type0_slots(0, 0, 2) == ((0, 1), 0, 0) and phy.type0_slots(0, 3, 2) == ((3, 4), 0, 0)
    assert phy.type0_slots(5, 1, 2)[0] == (5, 6) and phy.type0_slots(4, 1, 2)[2] == 0 # O = 5 ms (searchSpaceZero 4)

def test_cce_to_reg_interleaving_covers_coreset0():
    for n_rb, sym in ((24, 2), (48, 1), (96, 3)):
        regs = [r for c in phy.cce_regs(n_rb, sym, 17) for r in c]
        assert sorted(regs) == sorted((s, rb) for s in range(sym) for rb in range(n_rb))

def test_tbs():
    assert phy.tbs(12*12 - 12*3, 24, 2, 193/1024) == 984 # 108 REs a PRB, 24 PRBs, MCS 2: N_info 976.5 -> 984
    # Above 3824: N_info = 15600 x 6 x 948/1024 = 86652 -> N'_info 2^11 round(42.3) = 86016, C = 11 -> 88 ceil(86040/88) - 24
    assert phy.tbs(156, 100, 6, 948/1024) == 86040
    assert phy.dci_10_si_size(24) == 37

# ---- the profile

def downlink(frames, mcs=2, **kw):
    rng = np.random.default_rng(2)
    size = nr.tx.sib_size(24, 1, 2, mcs)
    sib = rng.integers(0, 2, size).astype(np.uint8)
    return nr.tx.downlink(pci=17, nfft=NFFT, frames=frames, sib=sib, ssb=(0, 1), mcs=mcs, k_ssb=2, **kw), sib

def receive(x, block=100000, **params):
    outs = list(rp.source.Memory(x, Fs=FS)(block) + rp.profile.load("nr/downlink", **params).rx())
    est = {k: outs[-1][k].stream.est[k] for k in ("pbch", "pdcch", "pdsch")}
    got = np.concatenate([np.asarray(o["pdsch"].data) for o in outs]).astype(np.uint8)
    return est, got, outs

def check(est, got, sib, blocks):
    assert (est["pbch"]["decoded"], est["pbch"]["failed"]) == (blocks, 0)
    assert (est["pdsch"]["decoded"], est["pdsch"]["failed"]) == (blocks, 0)
    assert np.array_equal(got, np.tile(sib, blocks))
    assert est["pdcch"]["last"]["rnti"] == phy.SI_RNTI

def test_terrestrial_cell():
    y, sib = downlink(10)
    x = nr.tx.awgn(np.concatenate((np.zeros(3000), y))*np.exp(2j*np.pi*3.3e3*np.arange(len(y) + 3000)/FS), 15, ref=y)
    est, got, _ = receive(x)
    check(est, got, sib, 5)
    assert est["pbch"]["last"]["sfn"] == 8 and est["pbch"]["last"]["k_ssb"] == 2

@pytest.mark.parametrize("t0", [-150.0, -1.0], ids=["far_46kHz", "closest_approach"])
def test_leo_pass(t0):
    """A 600 km orbit at 2.185 GHz: 150 s out the carrier is 46 kHz off and the samples 21 ppm short; at closest approach
    the offset sweeps at over 600 Hz/s"""
    y, sib = downlink(16)
    x = nr.tx.awgn(nr.tx.leo(y, FS, FC, t0=t0, delay=3000), 15, ref=y)
    d = rp.Data(x, Fs=FS, flush=True)
    d + nr.NrCell()
    cell = d.stream.est["cell"]
    t_end = t0 + len(x)/FS
    assert cell["cfo"] == pytest.approx(float(nr.tx.doppler(t_end, FC)), abs=60)
    assert cell["ppm"] == pytest.approx(-float(nr.tx.doppler(t_end, FC))/FC*1e6, abs=1.5)
    est, got, _ = receive(x)
    check(est, got, sib, 8)

def test_16qam_sib():
    y, sib = downlink(6, mcs=12)
    est, got, _ = receive(nr.tx.awgn(np.concatenate((np.zeros(1000), y)), 20, ref=y))
    check(est, got, sib, 3)
    assert est["pdsch"]["last"]["qm"] == 4

def test_the_block_cut_does_not_matter():
    y, sib = downlink(8)
    x = nr.tx.awgn(np.concatenate((np.zeros(1000), y)), 15, ref=y)
    a, b = receive(x, 100000), receive(x, 33333)
    for name in ("pbch", "pdcch", "pdsch"):
        assert np.array_equal(np.concatenate([np.asarray(o[name].data) for o in a[2]]),
                              np.concatenate([np.asarray(o[name].data) for o in b[2]]))

@pytest.mark.parametrize("sent", [True, False], ids=["phase_sent", "no_phase"])
def test_symbol_phase_either_way(sent):
    """The phase NR's upconversion gives each symbol (38.211 5.4), from the block's raster frequency: a cell that applies it
    and one that does not (as on air) both decode, the PDSCH trying both and the CRC choosing"""
    y, sib = downlink(6, fc=FC if sent else 0.0)
    x = nr.tx.awgn(np.concatenate((np.zeros(1000), y)), 15, ref=y)
    est, got, _ = receive(x, fc=FC)
    check(est, got, sib, 3)
    assert est["pdsch"]["last"]["phase"] is sent
    if sent: # Without fc the receiver cannot take it off
        assert receive(x)[0]["pdsch"]["decoded"] == 0
