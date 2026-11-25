"""pcapng out and back (rpps.source.PcapSink, Pcap), as Wireshark reads it (when tshark is installed), with RPPS's own
header for tools/wireshark/rpps.lua, and through the command line (rpps rx --pcap, rpps tx --from-pcap)"""
import io
import shutil
import struct
import subprocess

import pytest

import rpps as rp
from rpps import cli
from rpps.link.capture import Packet, linktype, exported_pdu
from support.paths import ROOT

TSHARK = shutil.which("tshark")
needs_tshark = pytest.mark.skipif(TSHARK is None, reason="tshark is not installed")
LUA = ROOT/"tools"/"wireshark"/"rpps.lua"

ETH = bytes.fromhex("ffffffffffff020000000001") + b"\x08\x06" + bytes(28) # (an ARP-sized frame)

def packets():
    return [Packet(linktype("ethernet"), ETH, "demo/x", "frame", "crc16_x25", True, 10, 1.5, 433.92e6, 12.0, {"a": 1}),
            Packet(linktype("ethernet"), ETH, "demo/x", "frame", "crc16_x25", False, 20, 2.5),
            Packet(linktype("upper_pdu"), exported_pdu(b"hi", "data", "demo/x hdr", "ok addr=7"), "demo/x", "hdr", None,
                   None, 30, 3.0)]

def test_written_and_read_back():
    buf = io.BytesIO()
    sink = rp.source.PcapSink(buf)
    packets() >> sink
    assert sink.count == 3 and sink.per == {"rpps:demo/x:frame": 2, "rpps:demo/x:hdr": 1}
    raw = buf.getvalue()
    assert struct.unpack("<II", raw[:8])[0] == 0x0A0D0D0A and len(raw) % 4 == 0
    cap = rp.source.Pcap(raw)
    recs = list(cap)
    assert [r.dlt for r in recs] == [1, 1, 252] and [r.data for r in recs[:2]] == [ETH, ETH]
    assert [r.time for r in recs] == [1.5, 2.5, 3.0] and [r.interface for r in recs] == [0, 0, 1]
    assert recs[0].comment == "demo/x frame, ok (crc16_x25), freq=433.920000 MHz, q=12.0 dB, sample=10, a=1"
    assert recs[1].flags == 1 | 1 << 24 and recs[0].flags == 1 # Inbound; the second's CRC error flag
    assert cap.interfaces == [(1, "rpps:demo/x:frame"), (252, "rpps:demo/x:hdr")]

def test_wall_clock_and_bad_clock():
    import time
    buf = io.BytesIO()
    t0 = time.time()
    rp.source.PcapSink(buf, clock="wall").write(packets()[0])
    assert next(iter(rp.source.Pcap(buf.getvalue()))).time >= t0
    with pytest.raises(ValueError, match="clock"):
        rp.source.PcapSink(buf, clock="gps")

def test_classic_pcap_is_read(tmp_path):
    hdr = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    rec = struct.pack("<IIII", 7, 500000, len(ETH), len(ETH)) + ETH
    (tmp_path/"c.pcap").write_bytes(hdr + rec)
    (r,) = list(rp.source.Pcap(tmp_path/"c.pcap"))
    assert (r.dlt, r.data, r.time) == (1, ETH, 7.5)

def test_named_pipes_are_windows_only():
    import sys
    if sys.platform == "win32":
        pytest.skip("a Windows named pipe waits for a reader")
    with pytest.raises(OSError, match="FIFO"):
        rp.source.PcapSink("pipe:rpps")

@needs_tshark
def test_wireshark_reads_it(tmp_path):
    with rp.source.PcapSink(tmp_path/"p.pcapng") as s:
        packets() >> s
    out = subprocess.run([TSHARK, "-r", str(tmp_path/"p.pcapng"), "-T", "fields", "-e", "frame.interface_name", "-e",
                          "frame.protocols", "-e", "frame.packet_flags_crc_error", "-e", "_ws.col.Protocol", "-e",
                          "_ws.col.Info", "-e", "frame.comment"], capture_output=True, text=True, check=True).stdout
    rows = [line.split("\t") for line in out.splitlines()]
    assert [r[0] for r in rows] == ["rpps:demo/x:frame"]*2 + ["rpps:demo/x:hdr"]
    assert rows[0][1].startswith("eth:ethertype") and [r[2] for r in rows] == ["False", "True", "False"]
    assert rows[2][1:] == ["exported_pdu:data", "False", "demo/x hdr", "ok addr=7", "demo/x hdr, unchecked, sample=30"]
    assert "FAILED (crc16_x25)" in rows[1][5]

@needs_tshark
def test_the_lua_dissector(tmp_path):
    p = [Packet(linktype("ethernet"), ETH, "ble/adv", "pdu", "crc24_ble", ok, 4160 + i, 0.004, 2.402e9, 12.5,
                {"type.name": name}) for i, (ok, name) in enumerate([(True, "ADV_IND"), (False, "ADV_NONCONN_IND")])]
    with rp.source.PcapSink(tmp_path/"l.pcapng", rpps=True) as s:
        p >> s
    assert {r.dlt for r in rp.source.Pcap(tmp_path/"l.pcapng")} == {147}
    run = lambda *a: subprocess.run([TSHARK, "-X", f"lua_script:{LUA}", "-r", str(tmp_path/"l.pcapng"), *a],
                                    capture_output=True, text=True, check=True).stdout
    rows = [line.split("\t") for line in run("-T", "fields", "-e", "frame.protocols", "-e", "rpps.profile", "-e",
                                             "rpps.crc_ok", "-e", "rpps.field", "-e", "rpps.sample", "-e", "_ws.col.Info").splitlines()]
    assert rows[0][:5] == ["rpps:eth:ethertype:arp", "ble/adv", "1", "type.name=ADV_IND", "4160"]
    assert rows[1][2] == "0" and "[check FAILED]" in rows[1][5]
    assert run("-Y", 'rpps.crc_ok == 0 && rpps.field == "type.name=ADV_NONCONN_IND"', "-T", "fields", "-e",
               "frame.number").split() == ["2"]

def test_cli_rx_to_pcap_and_back_out(tmp_path, capsys):
    """rpps tx the data, rpps rx it to a pcapng, rpps tx --from-pcap the packets, rpps rx them again: the same packets"""
    call = lambda c, ssid, last: bytes(ord(ch) << 1 for ch in c.ljust(6)) + bytes([0x60 | ssid << 1 | last])
    (tmp_path/"in.bin").write_bytes(call("APRS", 0, 0) + call("N0CALL", 7, 1) + b"\x03\xf0" + b"!4903.50N/07201.75W-")
    prm = ["--param", "centre=0", "--param", "sps=10"]
    assert cli.main(["tx", "ax25/afsk1200", str(tmp_path/"a.cf32"), "--rate", "1200", "--in", str(tmp_path/"in.bin")] + prm) == 0
    rx = ["--fs", "12000", "--sps", "10"] + prm
    assert cli.main(["rx", "ax25/afsk1200", str(tmp_path/"a.cf32"), "--pcap", str(tmp_path/"a.pcapng")] + rx) == 0
    assert cli.main(["tx", "ax25/afsk1200", str(tmp_path/"b.cf32"), "--rate", "1200", "--from-pcap", str(tmp_path/"a.pcapng"),
                     "--gap", "0.05"] + prm) == 0
    assert cli.main(["rx", "ax25/afsk1200", str(tmp_path/"b.cf32"), "--pcap", str(tmp_path/"b.pcapng")] + rx) == 0
    a, b = ([r.data for r in rp.source.Pcap(tmp_path/f"{k}.pcapng")] for k in "ab")
    assert a == b == [(tmp_path/"in.bin").read_bytes()]
    assert "1 messages" in capsys.readouterr().out

def test_cli_refuses_what_it_cant_do(tmp_path):
    with pytest.raises(SystemExit, match="one of them"):
        cli.main(["rx", "ax25/afsk1200", "--fs", "48000"])
    with pytest.raises(SystemExit, match="stdout"):
        cli.main(["rx", "ax25/afsk1200", "x.cf32", "--fs", "48000", "--messages", "--pcap", "-"])
    with pytest.raises(SystemExit, match="one of them"):
        cli.main(["tx", "ax25/afsk1200", "--rate", "1200", "--random", "10"])
