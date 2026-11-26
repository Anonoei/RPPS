"""LTE downlink to SIB1 (rpps.grid.lte, profile lte/downlink): a synthetic downlink from rpps.grid.lte.tx, received"""
import numpy as np
import pytest

import rpps as rp
from rpps.grid import lte
from rpps.grid.lte import phy

FS = {6: 1.92e6, 15: 3.84e6, 25: 7.68e6, 50: 15.36e6}

def received(n_rb, ports, frames=6, pci=77, cfo=3.1e3, delay=1234, snr_db=15.0, block=20000, seed=1, sib_len=224, x=None):
    rng = np.random.default_rng(seed)
    sib = rng.integers(0, 2, sib_len).astype(np.uint8)
    if x is None:
        x = lte.tx.channel(lte.tx.downlink(pci=pci, ports=ports, n_rb=n_rb, frames=frames, sib=sib), cfo=cfo, fs=FS[n_rb],
                           delay=delay, snr_db=snr_db, seed=seed)
    outs = list(rp.source.Memory(x, Fs=FS[n_rb])(block) + rp.profile.load("lte/downlink", n_rb=n_rb).rx())
    return outs, sib

def joined(outs, name):
    return np.concatenate([np.asarray(o[name].data) for o in outs]).astype(np.uint8)

def est(outs, name):
    return outs[-1][name].stream.est[name]

# ---- the resource-element maps

def test_maps_have_the_standard_sizes():
    for pci in (0, 1, 77, 503):
        k, l = phy.pbch_res(pci, 50)
        assert len(k) == 240 and set(l) == {7, 8, 9, 10} # 72 subcarriers x 4 symbols, 48 reference REs left out
        assert len(phy.pcfich_res(pci, 50, 4)[0]) == 16
    # 50 RB, CFI 1, Ng 1 (7 PHICH groups): 100 REGs in symbol 0, 4 PCFICH, 21 PHICH: 75 REGs, 8 CCEs
    assert len(phy.control_regs(25, 50, 1, phy.phich_groups(1.0, 50), 4)) == 75
    assert phy.dci_1a_size(50) == 27 and phy.dci_1a_size(6) == 21 # (26 and 20 are ambiguous sizes: one more)

@pytest.mark.parametrize("n_rb", [6, 15, 50])
def test_riv_round_trip(n_rb):
    for first in range(n_rb):
        for count in range(1, n_rb - first + 1):
            assert phy.riv_decode(lte.tx.riv(first, count, n_rb), n_rb) == (first, count)

def test_pdsch_avoids_reference_and_sync_signals():
    blocks = (np.arange(20, 30),)*2
    k, l = phy.pdsch_res(blocks, 1, 25, 0, 50, 4)
    crs = phy.crs_mask(25, 50, 4)
    assert not crs[l, k].any() and l.min() == 1
    centre = (k >= 264) & (k < 336)
    assert not (centre & np.isin(l, (5, 6, 7, 8, 9, 10))).any() # Subframe 0: PSS, SSS and PBCH
    k5, l5 = phy.pdsch_res(blocks, 1, 25, 5, 50, 4)
    assert len(k5) > len(k) # Subframe 5: no PBCH

def test_diversity_undone(rng):
    """Each port through a gain of its own: undo_diversity gives the symbols back, scaled by the channel's power"""
    for ports in (1, 2, 4):
        x = (rng.choice([-1, 1], 64) + 1j*rng.choice([-1, 1], 64))/np.sqrt(2)
        y = lte.tx.diversity(x, ports)
        h = np.exp(2j*np.pi*rng.random(4))*(0.5 + rng.random(4))
        G = np.zeros((14, 64), complex)
        G[3] = np.sum(h[:ports, None]*y, axis=0)
        H = np.broadcast_to(h[:, None, None], (4, 14, 64))
        z = phy.undo_diversity(G, H, np.arange(64), np.full(64, 3), ports)
        assert np.allclose(z/np.abs(z), x/np.abs(x))

# ---- the cell

def test_cell_search_finds_the_cell_and_its_offset():
    x = lte.tx.channel(lte.tx.downlink(pci=301, ports=1, n_rb=6, frames=4), cfo=-11.2e3, fs=FS[6], delay=5000, snr_db=10)
    d = rp.Data(x, Fs=FS[6], flush=True)
    out = d + lte.LteCell()
    e = d.stream.est["cell"]
    assert e["pci"] == 301 and e["cfo"] == pytest.approx(-11.2e3, abs=30)
    assert len(out) == 4*10*phy.subframe_len(128) # From the first subframe 0 (the delay's 5000 samples left behind)

# ---- the profile

@pytest.mark.parametrize("n_rb,ports", [(6, 1), (15, 2), (25, 4)])
def test_profile_receives_mib_cfi_dci_and_sib1(n_rb, ports):
    outs, sib = received(n_rb, ports)
    pbch = est(outs, "pbch")
    assert (pbch["decoded"], pbch["failed"]) == (6, 0)
    assert pbch["last"] == {"ports": ports, "n_rb": n_rb, "phich_ng": 1.0, "phich_duration": 0, "sfn": 5}
    assert est(outs, "pcfich")["decoded"] == est(outs, "pcfich")["frames"] # (from the first MIB on)
    assert est(outs, "pdcch")["last"]["rnti"] == phy.SI_RNTI
    pdsch = est(outs, "pdsch")
    assert (pdsch["frames"], pdsch["decoded"]) == (3, 3) # Frames 0, 2, 4: only the subframes a DCI allocated
    got = joined(outs, "pdsch")
    assert np.array_equal(got, np.tile(sib, 3))
    mibs = joined(outs, "pbch").reshape(-1, 24)
    assert [int("".join(map(str, m[6:14])), 2) for m in mibs] == [0, 0, 0, 0, 1, 1] # The SFN's 8 MSBs: frames 0-5

def test_the_block_cut_does_not_matter():
    a, _ = received(15, 2, block=20000)
    b, _ = received(15, 2, block=7777)
    for name in ("pbch", "pcfich", "pdcch", "pdsch"):
        assert np.array_equal(joined(a, name), joined(b, name))

def test_reacquired_after_a_fade():
    """The signal gone for 150 ms (30 half-frames: lost after 20): acquired again, the subframes still in their places"""
    rng = np.random.default_rng(4)
    sib = rng.integers(0, 2, 224).astype(np.uint8)
    y = lte.tx.downlink(pci=77, ports=2, n_rb=6, frames=30, sib=sib)
    x = lte.tx.channel(y, fs=FS[6], delay=300, snr_db=20, seed=4)
    n = phy.subframe_len(128)
    a, b = 300 + 70*n, 300 + 220*n
    s = np.std(x[:300])/np.sqrt(2) # (the delay's samples: the noise alone)
    x[a:b] = s*(rng.standard_normal(b - a) + 1j*rng.standard_normal(b - a))
    outs, _ = received(6, 2, x=x, block=50000)
    pbch = est(outs, "pbch")
    assert pbch["last"]["sfn"] == 29
    assert pbch["decoded"] >= 30 - 16 - 2 # All but the frames faded (and the two after, while it reacquires)
    assert joined(outs, "pbch").reshape(-1, 24).shape[0] == pbch["decoded"]

@pytest.mark.parametrize("cut", [0, 17, 829, 832, 9590])
def test_a_capture_starting_anywhere(cut):
    """Cut into the first subframe 0 (or its PSS): the reception starts at the next whole one"""
    x = lte.tx.channel(lte.tx.downlink(pci=77, ports=2, n_rb=6, frames=8), fs=FS[6], snr_db=20)[cut:]
    outs, _ = received(6, 2, x=x, block=30000)
    pbch = est(outs, "pbch")
    assert (pbch["decoded"], pbch["failed"], pbch["last"]["sfn"]) == (8 if cut == 0 else 7, 0, 7)

def messages(x, n_rb, block=20000, **params):
    """({channel: its messages}, the transport blocks' bits) of the profile over x"""
    rx, msgs, bits = rp.profile.load("lte/downlink", n_rb=n_rb, **params).rx(), [], []
    for d in rp.source.Memory(x, Fs=FS[n_rb])(block):
        bits.append(np.asarray(rx(d)["pdsch"].data).astype(np.uint8))
        msgs += rx.take_messages()
    return {k: [m for m in msgs if m.kind == k] for k in ("pbch", "pcfich", "pdcch", "pdsch")}, np.concatenate(bits)

# ---- TDD, formats 1A and 1C, distributed blocks, extended PHICH

def test_distributed_blocks_are_a_permutation_per_slot():
    """36.211 6.2.3.2: every bandwidth and gap maps the virtual blocks to as many physical ones, in each slot"""
    for n_rb in range(6, 111):
        for second in (False, True) if n_rb >= 50 else (False,):
            m = phy.dvrb_map(n_rb, second)
            assert m.shape == (2, phy.gaps(n_rb, second)[1])
            for slot in m:
                assert len(set(slot)) == len(slot) and slot.min() >= 0 and slot.max() < n_rb

def test_dci_sizes_and_tables():
    assert [phy.dci_1a_size(n, "tdd") for n in (6, 25, 50, 100)] == [23, 27, 29, 31]
    assert [phy.dci_1c_size(n) for n in (6, 25, 50, 100)] == [8, 12, 13, 15] # 36.212: N_VRB,gap1 in steps
    t = phy.tbs_table("lte/tbs_1c")
    assert len(t) == 32 and t[0] == [40] and t[31] == [1736]
    assert phy.gaps(50) == (27, 46, 46) and phy.gaps(50, True) == (9, 36, 18) # Table 6.2.3.2-1

CASES = [ # (N_RB, ports, duplex, DCI, distributed, extended PHICH)
    (6, 1, "fdd", "1C", False, False), (15, 2, "tdd", "1A", False, False), (25, 1, "fdd", "1A", True, False),
    (25, 4, "tdd", "1C", False, False), (25, 2, "fdd", "1A", False, True), (25, 4, "tdd", "1A", True, True),
    (50, 2, "tdd", "1A", True, False), (50, 4, "fdd", "1C", False, True),
]

@pytest.mark.parametrize("n_rb,ports,duplex,fmt,distributed,extended", CASES)
def test_profile_receives_every_layout(n_rb, ports, duplex, fmt, distributed, extended):
    sib = np.random.default_rng(1).integers(0, 2, 224).astype(np.uint8)
    gap = {"gap": 1} if n_rb >= 50 and (distributed or fmt == "1C") else {}
    y = lte.tx.downlink(pci=77, ports=ports, n_rb=n_rb, frames=6, sib=sib, duplex=duplex, fmt=fmt,
                        distributed=distributed, extended=extended, **gap)
    x = lte.tx.channel(y, cfo=2.1e3, fs=FS[n_rb], delay=1234, snr_db=18, seed=2)
    m, got = messages(x, n_rb)
    mib = [b for b in m["pbch"] if b.crc_ok]
    assert mib and all(b.fields["duplex"] == duplex and b.fields["ports"] == ports for b in mib)
    assert m["pdsch"] and all(b.crc_ok for b in m["pdsch"])
    assert {(d.fields["format"], d.fields.get("distributed", 1)) for d in m["pdcch"]} == {(fmt, int(distributed or fmt == "1C"))}
    assert len(got) >= 2*len(sib) and np.array_equal(got, np.tile(sib, len(got)//len(sib)))

# ---- cells sharing a carrier

def two_cells(duplex, weak_db, delay):
    sibs = [np.random.default_rng(i).integers(0, 2, 224).astype(np.uint8) for i in (1, 2)]
    parts = [lte.tx.channel(lte.tx.downlink(pci=pci, ports=2, n_rb=25, frames=8, sib=s, duplex=duplex, first=first),
                            cfo=cfo, fs=FS[25], delay=d, seed=seed)
             for pci, s, first, cfo, d, seed in ((77, sibs[0], 0, 2e3, 1234, 2), (201, sibs[1], 10, 1.7e3, delay, 3))]
    x = np.zeros(max(map(len, parts)), complex)
    for part, g in zip(parts, (1.0, 10**(weak_db/20))):
        x[:len(part)] += g*part
    rng = np.random.default_rng(0)
    x += 0.03*np.std(parts[0])*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    return x.astype(np.complex64), sibs

@pytest.mark.parametrize("duplex,weak_db,delay", [("fdd", -6, 1234), ("tdd", -6, 3000)])
def test_two_cells_on_a_carrier(duplex, weak_db, delay):
    x, sibs = two_cells(duplex, weak_db, delay)
    found = lte.search(x, FS[25])
    assert [c["pci"] for c in found] == [77, 201] and {c["duplex"] for c in found} == {duplex}
    assert found[1]["level_db"] == pytest.approx(weak_db, abs=2)
    for pci, sib in zip((77, 201), sibs):
        m, got = messages(x, 25, 40000, pci=pci)
        assert m["pbch"] and {b.fields["pci"] for b in m["pbch"]} == {pci}
        assert len(got) >= 2*len(sib) and np.array_equal(got, np.tile(sib, len(got)//len(sib)))

def test_noise_holds_no_cell(rng):
    x = rng.standard_normal(80000) + 1j*rng.standard_normal(80000)
    assert lte.search(x, 1.92e6) == []
