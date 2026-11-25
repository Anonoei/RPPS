"""Linux TUN/TAP (rpps.source.Tun): packets onto an interface and the system's packets off it, in a network namespace of
our own (`unshare -rn`: no root needed) where the system allows one"""
import shutil
import subprocess
import sys
import textwrap

import pytest

import rpps as rp
from rpps.errors import ConfigError

def _namespace():
    if not sys.platform.startswith("linux") or shutil.which("unshare") is None:
        return False
    r = subprocess.run(["unshare", "-rn", sys.executable, "-c", "import os; os.close(os.open('/dev/net/tun', os.O_RDWR))"],
                       capture_output=True)
    return r.returncode == 0

needs_namespace = pytest.mark.skipif(not _namespace(), reason="no user + network namespace with /dev/net/tun here")

def in_namespace(code):
    r = subprocess.run(["unshare", "-rn", sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout

SNIFF = """
import os, socket, rpps as rp
from rpps.link.capture import Packet, linktype
def sniff(name):
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
    s.bind((name, 0))
    s.settimeout(5)
    return s
def recv(s):
    while True: # (what arrives on the interface: not what the system sends out of it, IPv6 solicitations)
        d, a = s.recvfrom(2048)
        if a[2] != socket.PACKET_OUTGOING:
            return d, a
IP = bytes.fromhex("4500001c00004000401100000a0000010a000002") + bytes(8)
"""

@needs_namespace
def test_packets_onto_interfaces_of_their_link_type():
    out = in_namespace(SNIFF + """
    with rp.source.Tun("rpps0") as t:                                   # raw IP
        s = sniff("rpps0")
        [Packet(linktype("ipv4"), IP)] >> t
        d, a = recv(s)
        print(d == IP, a[3])
    mac = bytes.fromhex("41c8001234ffff3412") + b"hi"
    with rp.source.Tun("rpps1", linktype="ieee802_15_4_nofcs") as t:    # another link type (TUNSETLINK)
        s = sniff("rpps1")
        fcs = bytes(2) # (802.15.4 with FCS: converted, its FCS checked and taken off; a wrong one is skipped)
        crc = rp.link.load("crc/crc16_kermit").encoder.compute(mac).to_bytes(2, "little")
        t.write([Packet(linktype("ieee802_15_4_nofcs"), mac), Packet(linktype("ipv4"), IP),
                 Packet(linktype("ieee802_15_4"), mac + crc), Packet(linktype("ieee802_15_4"), mac + fcs)])
        d, a = recv(s)
        d2, _ = recv(s)
        print(d == mac == d2, a[3], a[1], t.skipped)
    eth = bytes.fromhex("ffffffffffff020000000001") + b"\\x08\\x00" + IP
    with rp.source.Tun("rpps2", mode="tap") as t:                        # Ethernet
        s = sniff("rpps2")
        t.write(eth)
        d, a = recv(s)
        print(d == eth, a[3])
    """.replace("\n    ", "\n"))
    assert out.split("\n")[:3] == ["True 65534", "True 804 246 {'ipv4': 1, 'ieee802_15_4': 1}", "True 1"] # ARPHRD_NONE, _IEEE802154, _ETHER

@needs_namespace
def test_the_systems_packets_off_a_tun():
    out = in_namespace(SNIFF + """
    with rp.source.Tun("rpps3") as t:
        os.system("ip addr add 10.9.9.1/24 dev rpps3")
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        u.sendto(b"to the air", ("10.9.9.2", 5000))
        got = [p for p in t.packets(timeout=1.0) if p[:1] == b"\\x45"]
        print(len(got), got[0][-10:] == b"to the air", got[0][9])
    """.replace("\n    ", "\n"))
    assert out.split() == ["1", "True", "17"] # One UDP datagram, as raw IPv4

def test_what_an_interface_cannot_carry():
    import sys
    if not sys.platform.startswith("linux"):
        with pytest.raises(OSError, match="pcapng"):
            rp.source.Tun("rpps0")
        return
    with pytest.raises(ConfigError, match="TAP carries Ethernet"):
        rp.source.Tun("rpps0", mode="tap", linktype="raw")
    with pytest.raises(ConfigError, match="no interface type"):
        rp.source.Tun("rpps0", linktype="ble_ll")
    with pytest.raises(ConfigError, match="mode"):
        rp.source.Tun("rpps0", mode="tunnel")
