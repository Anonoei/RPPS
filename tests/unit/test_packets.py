"""Packets that cross frames (rpps.link.packets): CCSDS space packets out of TM transfer frames (ccsds/tm_packets), DVB-S2
transport stream packets out of BBFRAMEs in TS mode (dvbs2/ts), however the stream is cut, a lost frame or a bad packet
dropping only what it should"""
import shutil
import struct
import subprocess

import numpy as np
import pytest

import rpps as rp
from rpps import _config
from rpps.base.types import Rep
from rpps.errors import ConfigError

TSHARK = shutil.which("tshark")

# ---- CCSDS TM space packets

FRAME = 223 # (RS depth 1: a transfer frame is one codeword's data)

def space_packet(apid, seq, data):
    return struct.pack(">HHH", apid, (3 << 14) | seq, len(data) - 1) + data

def frames_for(packets, vcid, ocf=False):
    """Transfer frames of one virtual channel carrying the packets back to back (CCSDS 132.0-B-3 4.1.2), the last
    filled with an idle packet"""
    stream = b"".join(packets)
    dfl = FRAME - 6 - (4 if ocf else 0)
    starts = np.cumsum([0] + [len(p) for p in packets[:-1]])
    out, pos, k = [], 0, 0
    while pos < len(stream):
        chunk = stream[pos:pos + dfl]
        firsts = [s - pos for s in starts if pos <= s < pos + dfl]
        if len(chunk) < dfl:
            chunk += space_packet(2047, 0, bytes(dfl - len(chunk) - 6))
        hdr = struct.pack(">HBBH", (42 << 4) | (vcid << 1) | ocf, k, k, (3 << 11) | (firsts[0] if firsts else 2047))
        out.append(hdr + chunk + (b"OCF!" if ocf else b""))
        pos, k = pos + dfl, k + 1
    return out

def channel_bits(frames):
    """Transfer frames through ccsds/tm_rs's code steps: the channel bits a receiver decodes"""
    bits = np.unpackbits(np.frombuffer(b"".join(frames), np.uint8)).astype(bool)
    d = rp.Data(bits, Fs=1e6, sps=1, flush=True)
    for how, op in rp.profile.load("ccsds/tm_rs", depth=1).tx():
        if how == "*" and op.CATEGORY == rp.Category.CODE:
            d = d*op
    return np.asarray(d.data).copy()

def receive(p, x, cuts, start=Rep.CHANNEL):
    rx, st, i, msgs = p.rx(start=start), None, 0, []
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], flush=k == len(cuts) - 1, stream=st)
        st, i = d.stream, i + n
        rx(d.derive(x[i - n:i], rp.Category.CODE, rp.dtype.BITS, rep=start))
        msgs += rx.take_messages()
    return msgs, st

@pytest.fixture(scope="module")
def tm():
    rng = np.random.default_rng(1)
    a = [space_packet(100, i, rng.bytes(int(rng.integers(1, 300)))) for i in range(12)]
    b = [space_packet(200, i, rng.bytes(int(rng.integers(1, 120)))) for i in range(6)]
    fa, fb = frames_for(a, 1), frames_for(b, 2, ocf=True)
    frames = [f for i in range(max(len(fa), len(fb))) for f in (fa[i:i + 1] + fb[i:i + 1])]
    return a, b, fa, frames

def test_space_packets_across_frames(tm):
    a, b, _, frames = tm
    x = channel_bits(frames)
    p = rp.profile.load("ccsds/tm_packets", depth=1)
    for cuts in ([len(x)], [1000, 3, 17000, len(x) - 18003]):
        msgs, st = receive(p, x, cuts)
        space = [m for m in msgs if m.kind == "space"]
        assert [m.payload for m in space if m.fields["channel"] == 1] == a # Virtual channels kept apart
        assert [m.payload for m in space if m.fields["channel"] == 2] == b # (the OCF passed over)
        assert st.est["space"]["idle"] == 2 and st.est["space"]["lost"] == 0
        assert all(m.parts == {"header": (0, 48), "payload": (48, 8*len(m.payload) - 48)} for m in space)

def test_a_lost_frame_drops_only_what_crossed_it(tm):
    a, b, fa, frames = tm
    msgs, st = receive(rp.profile.load("ccsds/tm_packets", depth=1), (x := channel_bits([f for f in frames if f is not fa[2]])),
                       [len(x)])
    dfl = FRAME - 6
    starts = np.cumsum([0] + [len(q) for q in a])
    crossed = {i for i in range(len(a)) if starts[i] < 3*dfl and starts[i + 1] > 2*dfl}
    assert crossed == {2, 3}
    got = [m.payload for m in msgs if m.kind == "space" and m.fields["channel"] == 1]
    assert got == [q for i, q in enumerate(a) if i not in crossed] and st.est["space"]["lost"] == 1

@pytest.mark.skipif(TSHARK is None, reason="tshark is not installed")
def test_wireshark_reads_space_packets(tm, tmp_path):
    a, b, _, frames = tm
    p = rp.profile.load("ccsds/tm_packets", depth=1)
    x = channel_bits(frames)
    msgs, _ = receive(p, x, [len(x)])
    with rp.source.PcapSink(tmp_path/"s.pcapng") as s:
        p.capture.packets(msgs) >> s
    out = subprocess.run([TSHARK, "-r", str(tmp_path/"s.pcapng"), "-Y", "ccsds", "-T", "fields", "-e", "ccsds.apid", "-e",
                          "ccsds.seqnum"], capture_output=True, text=True, check=True).stdout.split()
    rows = list(zip(out[::2], out[1::2]))
    assert sorted(rows) == sorted([("100", str(i)) for i in range(len(a))] + [("200", str(i)) for i in range(len(b))])

# ---- DVB-S2 transport stream

UPL = 1504

def bbframes(p, ts, nframes, dfl, corrupt=None):
    """BBFRAMEs in TS mode (EN 302 307-1 5.1.4-5.1.6): user packets back to back, each one's sync byte the CRC-8 of the
    one before, SYNCD where the first one starting in each data field is"""
    crc8 = rp.link.load("crc/crc8_dvbs2").encoder
    ups, prev = [], None
    for t in ts:
        ups.append(bytes([0 if prev is None else crc8.compute(prev[1:])]) + t[1:])
        prev = t
    if corrupt is not None: # A bit of one packet flipped after its check was made
        ups[corrupt] = ups[corrupt][:50] + bytes([ups[corrupt][50] ^ 1]) + ups[corrupt][51:]
    stream = np.unpackbits(np.frombuffer(b"".join(ups), np.uint8)).astype(bool)
    hdr = next(op for how, op in p.tx() if str(op) == "Header:dvbs2_bb")
    frames = []
    for f in range(nframes):
        syncd = (-f*dfl) % UPL
        values = {"ts_gs": 3, "sis": 1, "ccm": 1, "issyi": 0, "npd": 0, "ro": 0, "matype2": 0, "upl": UPL, "dfl": dfl,
                  "sync": 0x47, "syncd": syncd if syncd < dfl else 65535}
        frames.append(np.concatenate((hdr.build(values), stream[f*dfl:(f + 1)*dfl])))
    return np.concatenate(frames)

def phy_tx(bb, nframes):
    """dvbs2/normal without its BBHEADER step: BBFRAMEs made here (a packet corrupted, null packets deleted), sent"""
    from rpps.profile.profile import Profile
    obj = _config.resolve(_config.read(_config.find("profile", "dvbs2/normal")), {}, "dvbs2/normal")
    obj["steps"] = [st for st in obj["steps"] if st.get("id") != "bbheader"]
    obj["parallel"] = None
    obj.pop("capture")
    obj["controls"]["pls"]["frame"] = {"calc": "kbch"} # (the BBFRAME whole: no header step adds 80 bits)
    for e in obj["controls"].values():
        e.get("drive", {}).pop("bbheader", None)
    phy = Profile("phy", obj, params=_config.declared(obj))
    return phy.tx()(rp.Data(bb, Fs=1e6, sps=1, flush=True), fields=[{"modcod": 4, "pilots": 1}]*nframes)

@pytest.mark.parametrize("corrupt", [None, 30])
def test_dvbs2_transport_stream(corrupt, tmp_path):
    rng = np.random.default_rng(3)
    p = rp.profile.load("dvbs2/ts", sync={"timing": {"method": "known"}})
    dfl = rp.profile.table("dvbs2/modcod").row({"modcod": 4, "pilots": 1, "short": 0})["kbch"] - 80
    nframes = 4
    ts = [bytes([0x47, 0x01, 0x00, 0x10 | (k & 15)]) + rng.bytes(184) for k in range(nframes*dfl//UPL + 2)]
    o = phy_tx(bbframes(p, ts, nframes, dfl, corrupt), nframes)
    rx = p.rx()
    out = rx(rp.Data(np.concatenate((o.data, np.zeros(200, np.complex64))), Fs=o.Fs, sps=4, delay=o.delay, flush=True))
    got = [m for m in rx.take_messages() if m.kind == "ts"]
    whole = nframes*dfl//UPL # The packets wholly inside the 4 frames
    assert len(got) == whole and all(m.payload == ts[i] for i, m in enumerate(got) if m.crc_ok is not False)
    assert [m.crc_ok for m in got] == [i != corrupt for i in range(whole - 1)] + [None] # (the last: nothing after it)
    assert out.stream.est["ts"]["failed"] == (corrupt is not None) and got[0].check == "crc8_dvbs2"
    packets = p.capture.packets(got)
    assert {k.linktype.name for k in packets} == {"mpeg2_ts"} and all(k.data[0] == 0x47 for k in packets)
    if TSHARK:
        with rp.source.PcapSink(tmp_path/"t.pcapng") as s:
            packets >> s
        cc = subprocess.run([TSHARK, "-r", str(tmp_path/"t.pcapng"), "-T", "fields", "-e", "mp2t.cc"], capture_output=True,
                            text=True, check=True).stdout.split()
        assert cc[:5] == ["0", "1", "2", "3", "4"]

def test_step_errors():
    from rpps.link.packets import Packets
    with pytest.raises(ConfigError, match="unknown keys"):
        Packets({"pointer": "0", "length": "8", "nope": 1})
    with pytest.raises(ConfigError, match="needs"):
        Packets({"length": "8"})
    with pytest.raises(ConfigError, match="no parameter"):
        Packets({"pointer": "0", "length": "8", "params": ["fhp"]}).control(vcid=1)

# ---- DVB-S2 transmitting transport streams, null-packet deletion

def ts_stream(rng, n, pid=0x100):
    return [bytes([0x47, pid >> 8, pid & 255, 0x10 | (k & 15)]) + rng.bytes(184) for k in range(n)]

def receive_ts(p, o):
    rx = p.rx()
    out = rx(rp.Data(np.concatenate((o.data, np.zeros(200, np.complex64))), Fs=o.Fs, sps=4, delay=o.delay, flush=True))
    return out, rx.take_messages()

def test_dvbs2_transmits_transport_streams():
    """dvbs2/ts packs a transport stream into TS-mode BBFRAMEs itself (SYNCD, the CRC-8s), however the stream is cut"""
    rng = np.random.default_rng(5)
    p = rp.profile.load("dvbs2/ts", sync={"timing": {"method": "known"}})
    ts = ts_stream(rng, 3*32128//UPL + 1)
    bits = np.unpackbits(np.frombuffer(b"".join(ts), np.uint8)).astype(bool)
    outs = []
    for cuts in ([len(bits)], [5, 1500, 30000, 1, len(bits) - 31506]):
        tx, st, i, ys = p.tx(), None, 0, []
        for k, n in enumerate(cuts):
            d = rp.Data(bits[i:i + n], Fs=1e6, sps=1, flush=k == len(cuts) - 1, stream=st)
            st, i = d.stream, i + n
            ys.append(tx(d, fields={"modcod": 4, "pilots": 1}))
        outs.append(np.concatenate([y.data for y in ys]))
        syncd = [f["syncd"] for s, f, _ in st.control.fields("ts")]
        assert syncd == [0, 32128 % UPL and UPL - 32128 % UPL, UPL - 2*32128 % UPL, 65535]
    assert np.array_equal(outs[0], outs[1])
    out, msgs = receive_ts(p, ys[-1].derive(outs[1], ys[-1].CAT, ys[-1].DT))
    got = [m for m in msgs if m.kind == "ts"]
    assert [m.payload for m in got] == ts and [m.crc_ok for m in got] == [True]*(len(ts) - 1) + [None]

def npd_bbframes(p, ts, nframes, dfl):
    """BBFRAMEs in TS mode with null-packet deletion (EN 302 307-1 annex D.3): null packets taken out, each useful
    packet followed by the count of those deleted before it (DNP), its sync byte the CRC-8 of the one before"""
    crc8 = rp.link.load("crc/crc8_dvbs2").encoder
    ups, prev, dnp = [], None, 0
    for t in ts:
        if t[1:3] == b"\x1f\xff":
            dnp += 1
            continue
        up = t + bytes([dnp])
        ups.append(bytes([0 if prev is None else crc8.compute(prev[1:])]) + up[1:])
        prev, dnp = up, 0
    upl = 189*8
    stream = np.unpackbits(np.frombuffer(b"".join(ups), np.uint8)).astype(bool)
    hdr = next(op for how, op in p.tx() if str(op) == "Header:dvbs2_bb")
    frames = []
    for f in range(nframes):
        syncd = (-f*dfl) % upl
        values = {"ts_gs": 3, "sis": 1, "ccm": 0, "npd": 1, "upl": upl, "dfl": dfl, "sync": 0x47,
                  "syncd": syncd if syncd < dfl else 65535}
        frames.append(np.concatenate((hdr.build(values), stream[f*dfl:(f + 1)*dfl])))
    return np.concatenate(frames), nframes*dfl//upl # (the packets wholly inside the frames)

def test_dvbs2_null_packet_deletion():
    """Null packets deleted at the transmitter are put back where they were, the DNP bytes taken off"""
    rng = np.random.default_rng(6)
    p = rp.profile.load("dvbs2/ts", sync={"timing": {"method": "known"}})
    dfl = rp.profile.table("dvbs2/modcod").row({"modcod": 4, "pilots": 1, "short": 0})["kbch"] - 80
    null = bytes([0x47, 0x1F, 0xFF, 0x10]) + b"\xff"*184
    ts = []
    for k in range(120):
        ts += [null]*int(rng.integers(0, 4)) + ts_stream(rng, 1, 0x100 + k % 3)
    bb, whole = npd_bbframes(p, ts, 4, dfl)
    out, msgs = receive_ts(p, phy_tx(bb, 4))
    got = [m for m in msgs if m.kind == "ts"]
    useful = [t for t in ts if t[1:3] != b"\x1f\xff"]
    assert [m.payload for m in got] == useful[:whole] and all(m.crc_ok for m in got[:-1])
    assert all("dnp" in m.fields for m in got)
    k = ts.index(useful[whole - 1])
    given = np.packbits(np.asarray(out.data)).tobytes()
    assert given == b"".join(ts[:k + 1]) and out.stream.est["ts"]["inserted"] == k + 1 - whole

# ---- MPE and ULE in a transport stream

def ipv4(rng, n):
    h = bytearray(struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + n, 1, 0, 64, 17, 0, bytes([10, 0, 0, 1]), bytes([10, 0, 0, 2])))
    s = sum(struct.unpack(">10H", bytes(h)))
    s = (s & 0xFFFF) + (s >> 16)
    h[10:12] = struct.pack(">H", ~((s & 0xFFFF) + (s >> 16)) & 0xFFFF)
    return bytes(h) + rng.bytes(n)

def mpe_section(ip, k):
    crc = rp.link.load("crc/crc32_mpeg2").encoder
    body = bytes([0x3E, 0xB0 | (len(ip) + 13) >> 8, (len(ip) + 13) & 255, 0x02, 0x00, 0xC1, k, 0, 0x00, 0x00, 0x5E, 0x01]) + ip
    return body + crc.compute(body).to_bytes(4, "big")

def ule_sndu(ip):
    crc = rp.link.load("crc/crc32_mpeg2").encoder
    body = struct.pack(">HH", 0x8000 | (len(ip) + 4), 0x0800) + ip # (D = 1: no destination address)
    return body + crc.compute(body).to_bytes(4, "big")

def ts_packets(units, pid, rng):
    """Units (sections, SNDUs) in the payloads of transport packets of one PID (ISO/IEC 13818-1 2.4.4.2): a pointer
    field where one starts, some packets with an adaptation field, 0xFF to fill the last"""
    stream = b"".join(units)
    starts = set(np.cumsum([0] + [len(u) for u in units[:-1]]).tolist())
    out, pos, cc = [], 0, 0
    while pos < len(stream):
        af = b"" if rng.random() < 0.7 else bytes([7, 0]) + b"\xff"*6
        room = 184 - len(af)
        first = min((s for s in starts if pos <= s < pos + room), default=None)
        if first is not None:
            room -= 1
            first = min((s for s in starts if pos <= s < pos + room), default=None)
        data = stream[pos:pos + room]
        payload = (bytes([first - pos]) if first is not None else b"") + data
        payload += b"\xff"*(184 - len(af) - len(payload))
        afc = 3 if af else 1
        out.append(bytes([0x47, (first is not None) << 6 | pid >> 8, pid & 255, afc << 4 | cc]) + af + payload)
        pos, cc = pos + len(data), (cc + 1) % 16
    return out

def mux(rng, *streams):
    """Transport packets of several PIDs, each in its order, mixed, with null packets"""
    null = bytes([0x47, 0x1F, 0xFF, 0x10]) + b"\xff"*184
    queues = [list(s) for s in streams]
    out = []
    while any(queues):
        q = [q for q in queues if q][int(rng.integers(0, sum(1 for q in queues if q)))]
        out.append(q.pop(0))
        if rng.random() < 0.2:
            out.append(null)
    return out

def through_dvbs2(p, ts):
    """Transport packets sent by dvbs2/ts and received by profile p"""
    tx = rp.profile.load("dvbs2/ts").tx()
    bits = np.unpackbits(np.frombuffer(b"".join(ts), np.uint8)).astype(bool)
    o = tx(rp.Data(bits, Fs=1e6, sps=1, flush=True), fields={"modcod": 4, "pilots": 1})
    return receive_ts(p, o)

@pytest.mark.parametrize("kind", ["mpe", "ule"])
def test_ip_over_transport_streams(kind, tmp_path):
    rng = np.random.default_rng(7)
    ips = [ipv4(rng, int(rng.integers(0, 600))) for _ in range(40)]
    units = [mpe_section(ip, k) for k, ip in enumerate(ips)] if kind == "mpe" else [ule_sndu(ip) for ip in ips]
    psi = ts_packets([bytes([0, 0xB0, 13, 0, 1, 0xC1, 0, 0, 0, 1, 0xE1, 0, 0, 0, 0, 0])]*6, 0, rng) # (a PAT-like section)
    other = ts_stream(rng, 12, 0x101)
    ts = mux(rng, ts_packets(units, 0x100, rng), psi, other)
    if len(ts) % 2: # (the last packet's check needs one after it)
        ts.append(other[0])
    ts.append(other[0])
    p = rp.profile.load(f"dvbs2/{kind}", pid=0x100, sync={"timing": {"method": "known"}})
    out, msgs = through_dvbs2(p, ts)
    got = [m for m in msgs if m.kind == kind]
    assert [m.payload for m in got] == ips and all(m.crc_ok for m in got)
    est = out.stream.est["sections" if kind == "mpe" else "sndu"]
    assert est["lost"] == 0 and est["misaligned"] == 0
    if kind == "mpe": # Every PID's sections (the default): the others' are not MPE, and not kept
        _, msgs = through_dvbs2(rp.profile.load("dvbs2/mpe", sync={"timing": {"method": "known"}}), ts)
        assert [m.payload for m in msgs if m.kind == "mpe"] == ips
    # A transport packet lost: only the units that crossed it go
    ts2 = [t for t in ts if t is not ts[len(ts)//2]]
    out, msgs = through_dvbs2(p, ts2)
    got2 = [m.payload for m in msgs if m.kind == kind and m.crc_ok]
    assert len(ips) - 3 <= len(got2) < len(ips) and all(g in ips for g in got2)
    if TSHARK:
        with rp.source.PcapSink(tmp_path/"ip.pcapng") as s:
            p.capture.packets(got) >> s
        rows = subprocess.run([TSHARK, "-r", str(tmp_path/"ip.pcapng"), "-T", "fields", "-e", "ip.len", "-e",
                               "ip.checksum.status"], capture_output=True, text=True, check=True).stdout.split("\n")
        assert [r.split("\t")[0] for r in rows if r] == [str(len(ip)) for ip in ips]
