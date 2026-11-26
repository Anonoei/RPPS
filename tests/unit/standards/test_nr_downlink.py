"""5G NR downlink to SIB1 through LEO Doppler (rpps.grid.nr, profile nr/downlink): synthetic, from rpps.grid.nr.tx"""
import json

import numpy as np
import pytest

import rpps as rp
from rpps.grid import nr
from rpps.grid.nr import phy
from support.signals import unhex
from support.paths import VECTORS as VECTOR_DIR

FS, NFFT, FC = 7.68e6, 512, 2.185e9
V = json.loads((VECTOR_DIR/"nr_py3gpp_ssb.json").read_text())

def signs(h, n):
    return 1 - 2.0*unhex(h, n)

# ---- against py3gpp (tests/vectors/nr_py3gpp_ssb.json: what it is in its "source")

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

def test_numerology():
    for nfft in (256, 512, 1024):
        for mu in (0, 1):
            long, cp = phy.cp_lengths(nfft, mu)
            n = 1 if mu else 2 # the long prefixes of a slot: one a half millisecond
            assert n*long + (14 - n)*cp + 14*nfft == phy.slot_len(nfft)
    assert [phy.ssb_place("A", i) for i in range(3)] == [(0, 2), (0, 8), (1, 2)]
    assert [phy.ssb_place("B", i) for i in (0, 1, 2, 4)] == [(0, 4), (0, 8), (1, 2), (2, 4)]
    assert [phy.ssb_place("C", i) for i in (1, 7)] == [(0, 8), (3, 8)]

@pytest.mark.parametrize("scs,common,min_bw", [(15e3, 15e3, 5), (15e3, 30e3, 5), (30e3, 15e3, 5), (30e3, 30e3, 5),
                                               (30e3, 15e3, 40), (30e3, 30e3, 40)])
def test_every_coreset0_table(scs, common, min_bw):
    rows = [phy.coreset0(i, scs, common, min_bw) for i in range(9)]
    for r in rows:
        assert r["pattern"] == 1 and r["rbs"] in (24, 48, 96) and r["symbols"] in (1, 2, 3) and r["offset"] >= 0

def test_interleaved_blocks_are_a_permutation():
    for n_rb in (24, 48, 96, 25):
        assert sorted(phy.vrb_interleaver(n_rb)) == list(range(n_rb))
    assert list(phy.vrb_interleaver(24)[:6]) == [0, 1, 12, 13, 2, 3] # bundles of 2, written in two rows

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
    return nr.tx.downlink(pci=17, nfft=NFFT, frames=frames, sib=sib, ssb=(0,), mcs=mcs, k_ssb=2, **kw), sib

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

# ---- 15 and 30 kHz

def cell(scs=30e3, common=30e3, nfft=512, ssb=(0,), tdra=0, mcs=2, min_bw=5, coreset0=0, frames=8, **kw):
    """A downlink at these spacings, received through a carrier offset and noise"""
    rb = phy.coreset0(coreset0, scs, common, min_bw)["rbs"]
    sib = np.random.default_rng(2).integers(0, 2, nr.tx.sib_size(rb, tdra + 1, 2, mcs)).astype(np.uint8)
    y = nr.tx.downlink(pci=17, nfft=nfft, frames=frames, sib=sib, ssb=ssb, scs=scs, scs_common=common, tdra=tdra, mcs=mcs,
                       min_bw=min_bw, coreset0=coreset0, k_ssb=kw.pop("k_ssb", 2), **kw)
    fs = scs*nfft
    x = nr.tx.awgn(np.concatenate((np.zeros(3000), y))*np.exp(2j*np.pi*3.3e3*np.arange(len(y) + 3000)/fs), 15, ref=y)
    return x, fs, sib

def receive_at(x, fs, scs, nfft, block=100000, **params):
    rx = rp.profile.load("nr/downlink", scs=scs, nfft=nfft, **params).rx()
    msgs, bits = [], []
    for d in rp.source.Memory(x, Fs=fs)(block):
        out = rx(d)
        msgs += rx.take_messages()
        bits.append(np.asarray(out["pdsch"].data).astype(np.uint8))
    est = {k: out[k].stream.est[k] for k in ("pbch", "pdcch", "pdsch")}
    return est, np.concatenate(bits), msgs

LAYOUTS = {
    "15_30": dict(scs=15e3, common=30e3, nfft=1024),
    "30_15": dict(scs=30e3, common=15e3),
    "30_30": dict(),
    "pattern_B": dict(pattern="B", ssb=(6,)),
    "half_frame_1_of_8": dict(ssb=(5,), hrf=1),
    "half_frame_1_of_4": dict(ssb=(2,), hrf=1, lmax=4),
    "coreset_after_symbol_0": dict(ssb=(1,), ss0=1, tdra=7, mcs=6, k_ssb=0),
    "interleaved": dict(nfft=768, coreset0=10, vrb=1),
    "40MHz_tables": dict(nfft=1024, min_bw=40, coreset0=6),
    "40MHz_30_15": dict(common=15e3, nfft=1024, min_bw=40, coreset0=1),
}

@pytest.mark.parametrize("name", LAYOUTS)
def test_spacings_and_layouts(name):
    kw = dict(LAYOUTS[name])
    scs, nfft = kw.get("scs", 30e3), kw.get("nfft", 512)
    x, fs, sib = cell(**kw)
    params = {k: kw[k] for k in ("pattern", "min_bw") if k in kw}
    est, got, msgs = receive_at(x, fs, scs, nfft, **params)
    blocks = len(got)//len(sib)
    assert blocks >= 3 and np.array_equal(got, np.tile(sib, blocks))
    assert est["pbch"]["failed"] == 0 and est["pdsch"]["failed"] == 0
    mib = next(m for m in msgs if m.kind == "pbch" and m.crc_ok).fields
    assert mib["scs"] == scs and mib["scs_common"] == kw.get("common", 30e3)
    assert mib["i_ssb"] == kw.get("ssb", (0,))[0] and mib["hrf"] == kw.get("hrf", 0)

def test_a_block_without_sib1():
    """k_SSB above 23: the MIB is read, no CORESET 0 is searched (a non-standalone cell's block)"""
    y = nr.tx.downlink(pci=17, nfft=512, frames=6, scs=30e3)
    k = nr.tx.downlink(pci=17, nfft=512, frames=6, scs=30e3, k_ssb=30)
    assert len(y) == len(k)
    est, got, _ = receive_at(nr.tx.awgn(np.concatenate((np.zeros(1000), k)), 20, ref=k), 15.36e6, 30e3, 512)
    assert est["pbch"]["decoded"] == 3 and est["pdcch"]["no_sib1"] > 0 and not len(got)

def test_leo_pass_at_30_khz():
    sib = np.random.default_rng(2).integers(0, 2, nr.tx.sib_size(24, 1, 2, 2)).astype(np.uint8)
    y = nr.tx.downlink(pci=17, nfft=512, frames=16, sib=sib, scs=30e3, k_ssb=2)
    x = nr.tx.awgn(nr.tx.leo(y, 15.36e6, FC, t0=-150.0, delay=3000), 15, ref=y)
    est, got, _ = receive_at(x, 15.36e6, 30e3, 512)
    check(est, got, sib, 8)

@pytest.mark.parametrize("scs,common,nfft", [(30e3, 30e3, 512), (15e3, 30e3, 1024)])
def test_the_block_cut_does_not_matter_at_30_khz(scs, common, nfft):
    x, fs, sib = cell(scs, common, nfft, frames=6)
    a, b = receive_at(x, fs, scs, nfft, 100000), receive_at(x, fs, scs, nfft, 33333)
    assert len(a[1]) and np.array_equal(a[1], b[1])
    assert [(m.kind, m.payload) for m in a[2]] == [(m.kind, m.payload) for m in b[2]]
