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
    """dvbs2/normal without its BBHEADER step (its transmitter makes generic streams): BBFRAMEs made here, sent"""
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
