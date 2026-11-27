"""DVB SI and MPEG PSI sections (header/link/dvb_si): each table read from sections built byte by byte from the standards,
built back, and received out of transport streams over DVB-S2 (dvbs2/si) and DVB-S (dvbs/si) however they are cut"""
import collections
import shutil
import subprocess

import numpy as np
import pytest

import rpps as rp
from rpps.errors import ConfigError
from support import ts as T

TSHARK = shutil.which("tshark")
KNOWN = {"timing": {"method": "known"}}

def as_bits(b):
    return np.unpackbits(np.frombuffer(b, np.uint8)).astype(bool)

@pytest.fixture(scope="module")
def si():
    return rp.link.load("header/dvb_si")

def read(h, section):
    """(header fields, payload values, CRC ok or None) of one whole section"""
    bits = as_bits(section)
    fields, _ = h.parse(bits)
    hb, n, fb = h.need(bits), h.payload_bits(fields), h.frame_fcs(fields)
    assert hb + n + fb == len(bits)
    ok = None if not fb else np.array_equal(h.fcs.field(bits[:hb + n]), bits[hb + n:])
    return fields, h.payload_values(bits[hb:hb + n], fields), ok

def body(d):
    """A descriptor's body"""
    return d["body"]

SAT = T.satellite_delivery(11778000000, 192, 27500000)
SECTIONS = {
    "NIT": T.nit(0x1234, "Test net", [(1, 2, [SAT])]),
    "SDT": T.sdt(1, 2, [(100, 4, [T.service(1, b"Prov", b"\x15Caf\xc3\xa9 TV")])]),
    "EIT_pf": T.eit_pf(100, 1, 2, [(7, "2026-11-26", "20:15:00", "01:30:00", [T.short_event("eng", "News", "Today")])]),
    "TDT": T.tdt("2026-11-26", "12:34:56"),
    "TOT": T.tot("2026-11-26", "12:34:56", [T.local_time_offset("DEU", 60, "2027-03-28", "01:00:00", 120)]),
    "PAT": T.pat(1, [(0, 0x10), (100, 0x100)]),
    "PMT": T.pmt(100, 0x101, [(27, 0x101), (3, 0x102)]),
}

def test_network_time_and_service_tables(si):
    f, v, ok = read(si, SECTIONS["NIT"])
    assert f["table_id.name"] == "NIT" and f["table_id_extension"] == 0x1234 and ok
    nit = v["nit"]
    assert body(nit["descriptors"][0]) == {"network_name": "Test net"}
    ts = nit["transport_streams"][0]
    assert (ts["transport_stream_id"], ts["original_network_id"]) == (1, 2)
    sat = body(ts["descriptors"][0])
    assert sat["frequency"] == 11778000000 and sat["symbol_rate"] == 27500000 and sat["orbital_position"] == 192
    assert sat["west_east.name"] == "east" and sat["modulation_system.name"] == "DVB-S2" and sat["fec_inner.name"] == "3/4"
    assert sat["modulation_type.name"] == "QPSK" and sat["polarization.name"] == "linear horizontal"

    f, v, ok = read(si, SECTIONS["TDT"])
    assert f["table_id.name"] == "TDT" and ok is None and "table_id_extension" not in f # (no section syntax, no CRC)
    assert v["tdt"] == {"utc_date": "2026-11-26", "utc_time": "12:34:56"}

    f, v, ok = read(si, SECTIONS["TOT"])
    assert f["table_id.name"] == "TOT" and f["ssi"] == 0 and ok # (no section syntax, and a CRC all the same)
    off = body(v["tot"]["descriptors"][0])["offsets"][0]
    assert v["tot"]["utc_time"] == "12:34:56" and off["country_code"] == "DEU" and off["local_time_offset"] == "01:00"
    assert off["time_of_change_date"] == "2027-03-28" and off["next_time_offset"] == "02:00"

    f, v, ok = read(si, SECTIONS["SDT"])
    svc = v["sdt"]["services"][0]
    assert ok and svc["service_id"] == 100 and svc["running_status.name"] == "running"
    assert body(svc["descriptors"][0]) == {"service_type": 1, "service_type.name": "digital television",
                                           "provider_name_length": 4, "provider_name": "Prov",
                                           "service_name_length": 9, "service_name": "Café TV"}

    f, v, ok = read(si, SECTIONS["EIT_pf"])
    ev = v["eit"]["events"][0]
    assert ok and (ev["start_date"], ev["start_time"], ev["duration"]) == ("2026-11-26", "20:15:00", "01:30:00")
    se = body(ev["descriptors"][0])
    assert (se["language"], se["event_name"], se["text"]) == ("eng", "News", "Today")

def test_program_tables(si):
    _, v, ok = read(si, SECTIONS["PAT"])
    assert ok and [(p["program_number"], p["pid"]) for p in v["pat"]["programs"]] == [(0, 0x10), (100, 0x100)]
    _, v, ok = read(si, SECTIONS["PMT"])
    pmt = v["pmt"]
    assert ok and pmt["pcr_pid"] == 0x101 and pmt["descriptors"] == []
    assert [(s["stream_type.name"], s["elementary_pid"]) for s in pmt["streams"]] == [("H.264", 0x101), ("MPEG-1 audio", 0x102)]
    assert body(pmt["streams"][1]["descriptors"][0]) == {"component_tag": 2}

def test_unknown_descriptors_are_kept_as_bytes(si):
    sec = T.long_section(0x40, 1, T.loop([T.descriptor(0x4A, b"\x01\x02\x03")]) + b"\xf0\x00")
    _, v, ok = read(si, sec)
    d = v["nit"]["descriptors"][0]
    assert ok and d["tag.name"] == "linkage" and body(d) == {"data": b"\x01\x02\x03"}

def strip_lengths(v):
    """The values without the lengths a definition works out on build"""
    if isinstance(v, list):
        return [strip_lengths(x) for x in v]
    if isinstance(v, dict):
        return {k: strip_lengths(x) for k, x in v.items() if not (k == "length" or k.endswith("_length"))}
    return v

@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_sections_build_back(si, name):
    """Header and tables built from what was read, the lengths and the CRC worked out: the same bytes"""
    f, v, _ = read(si, SECTIONS[name])
    scope = {k: x for k, x in f.items() if not isinstance(x, (dict, list))}
    payload = si.payload_schema.build(strip_lengths(v), scope)
    h = rp.link.load("header/dvb_si")
    h.control(**{k: x for k, x in f.items() if k != "section_length" and "." not in k})
    out = rp.Data(payload, flush=True) * h
    assert np.packbits(np.asarray(out.data, bool)).tobytes() == SECTIONS[name]

def test_a_conditional_fcs_is_not_framed():
    with pytest.raises(ConfigError):
        rp.link.Header("x", [{"name": "a", "bits": 8}], fcs={"crc": "crc32_mpeg2", "if": "a"}, framed=True)

# ---- out of transport streams

def multiplex(rng, bad_nit=False):
    """PSI/SI on their PIDs (each section twice), a PMT on 0x100, video on 0x101, nulls; the second NIT's CRC broken"""
    nit = SECTIONS["NIT"]
    nit2 = nit[:12] + bytes([nit[12] ^ 0x20]) + nit[13:] if bad_nit else nit
    tsx = T.mux(rng,
                T.ts_packets([SECTIONS["PAT"]]*2, 0x00, rng),
                T.ts_packets([nit, SECTIONS["SDT"], nit2, SECTIONS["SDT"]], 0x10, rng),
                T.ts_packets([SECTIONS["EIT_pf"]]*2, 0x12, rng),
                T.ts_packets([SECTIONS["TDT"], SECTIONS["TOT"]]*2, 0x14, rng),
                T.ts_packets([SECTIONS["PMT"]]*2, 0x100, rng),
                T.ts_stream(rng, 12, 0x101))
    return tsx

def tables(msgs):
    return collections.Counter((m.fields["table_id.name"], m.crc_ok) for m in msgs if m.kind == "si")

WANT = {("PAT", True): 2, ("PMT", True): 2, ("NIT", True): 2, ("SDT", True): 2, ("EIT_pf", True): 2, ("TDT", None): 2,
        ("TOT", True): 2}

def test_dvbs2_service_information_however_cut():
    rng = np.random.default_rng(11)
    tsx = multiplex(rng) + [T.NULL]*2
    p = rp.profile.load("dvbs2/si", pids=[0x100], sync=KNOWN)
    got = []
    for block in (None, 7919):
        out, msgs = T.through_dvbs2(p, tsx, block)
        assert tables(msgs) == WANT
        est = out.stream.est["sections"]
        assert est["lost"] == 0 and est["misaligned"] == 0
        got.append([(m.fields["table_id.name"], m.sample, m.payload) for m in msgs if m.kind == "si"])
    assert got[0] == got[1]
    tdt = next(m for m in msgs if m.kind == "si" and m.fields["table_id.name"] == "TDT")
    assert tdt.fields["payload"]["tdt"]["utc_time"] == "12:34:56"
    _, msgs = T.through_dvbs2(rp.profile.load("dvbs2/si", sync=KNOWN), tsx) # Without the PMT's PID: no PMT
    assert ("PMT", True) not in tables(msgs) and tables(msgs)[("NIT", True)] == 2

def test_a_broken_section_crc_is_told():
    rng = np.random.default_rng(12)
    tsx = multiplex(rng, bad_nit=True) + [T.NULL]*2
    _, msgs = T.through_dvbs2(rp.profile.load("dvbs2/si", sync=KNOWN), tsx)
    t = tables(msgs)
    assert t[("NIT", True)] == 1 and t[("NIT", False)] == 1 and t[("SDT", True)] == 2

def test_dvbs_service_information():
    rng = np.random.default_rng(13)
    tsx = [T.NULL]*16 + multiplex(rng)
    tsx += [T.NULL]*(-len(tsx) % 8 + 24) # (the deinterleaver's and decoder's latency)
    known = {**KNOWN, "carrier": {"method": "none"}}
    data = rp.Data(np.frombuffer(b"".join(tsx), np.uint8), Fs=1e6, sps=1, flush=True)
    w = rp.profile.load("dvbs/qpsk", sync=known).tx()(data)
    p = rp.profile.load("dvbs/si", pids=[0x100], sync=known)
    got = []
    for block in (len(w.data), 6007): # (the alignment publishing packet starts, however cut)
        rx, st = p.rx(), None
        for i in range(0, len(w.data), block):
            d = rp.Data(w.data[i:i + block], Fs=w.Fs, sps=w.sps, delay=w.delay, stream=st, flush=i + block >= len(w.data))
            st = d.stream
            out = rx(d)
        msgs = rx.take_messages()
        assert tables(msgs) == WANT
        assert out.stream.est["sections"]["lost"] == 0
        got.append([(m.fields["table_id.name"], m.sample, m.payload) for m in msgs if m.kind == "si"])
    assert got[0] == got[1]

@pytest.mark.skipif(TSHARK is None, reason="tshark not installed")
def test_wireshark_reads_the_tables(tmp_path):
    rng = np.random.default_rng(14)
    tsx = multiplex(rng) + [T.NULL]*2
    p = rp.profile.load("dvbs2/si", sync=KNOWN)
    _, msgs = T.through_dvbs2(p, tsx)
    with rp.source.PcapSink(tmp_path/"si.pcapng") as s:
        p.capture.packets([m for m in msgs if m.kind == "ts"]) >> s
    rows = subprocess.run([TSHARK, "-r", str(tmp_path/"si.pcapng"), "-T", "fields", "-e", "mpeg_sect.tid"],
                          capture_output=True, text=True, check=True).stdout.split()
    assert {"0x40", "0x70", "0x73"} <= {t for r in rows for t in r.split(",")}
